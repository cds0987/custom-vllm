"""
TASK A2: last attempt at a custom quantized Qwen3.5-9B checkpoint, switching
BOTH variables suspected of costing quantize_awq_9b.py's v3 checkpoint the
SWE-bench ppl comparison against RedHatAI/Qwen3.5-9B-quantized.w4a16 (v3
scored ppl=6.9958 vs RedHatAI's 5.1578 and the plain GGUF Q4_K_M baseline's
5.6475 -- a WARN/FAIL despite v3's decode throughput beating both).

This is deliberately NOT a controlled experiment -- changing two variables
at once trades diagnostic cleanliness for the best realistic shot at beating
RedHatAI's number in one more ~75-90 minute run, per the explicit decision
to stop the custom-quantize line after this attempt regardless of outcome.

Variable 1 -- algorithm: GPTQModifier instead of AWQModifier. RedHatAI's
published quantization_config (read directly from their config.json) uses
actorder="static" and format="pack-quantized" -- both hallmarks of GPTQ, not
AWQ (AWQ does not have an activation-reordering concept; it protects salient
channels via smoothing scales instead). GPTQModifier's llm-compressor default
IS actorder="static" already, so this recipe just uses the modifier's
defaults rather than overriding them.

Variable 2 -- calibration domain: v3 (like the 2B script before it)
calibrated on HuggingFaceH4/ultrachat_200k alone -- general chat, zero code.
The SWE-bench ppl gate scores real GitHub PR patches: pure code-domain text.
A calibration set with no code exposure has no way to learn good scales for
whatever activation distributions code text produces, and RedHatAI's actual
calibration set is unknown but their ppl edge on code-domain content is
exactly what a code-aware calibration set would predict. This recipe mixes
~50% code (iamtarun/python_code_instructions_18k_alpaca -- small, fast to
download, Alpaca-schema instruction/input/output triples) with ~50% ultrachat
chat data, 256 samples total, so the checkpoint isn't overfit to one domain
at the cost of the other (bench_load's decode-speed probes and the 5-probe
quality gate are general-chat-shaped, not code-shaped, so pure-code
calibration would risk trading one domain's ppl for the other's).

Scheme, ignores, and the two hard-won empirical lessons all carry over
unchanged from quantize_awq_9b.py v3 (see that script's docstring for the
full failure history if either of these needs re-litigating):
  - int4, group_size=32, symmetric -- finer than RedHatAI's 128, GPTQ
    convention (group + activation reordering is the standard high-fidelity
    GPTQ recipe shape, matching what RedHatAI likely ran).
  - `re:.*linear_attn.*` stays fully excluded (fp16) BY DEFAULT. Quantizing
    ANY of GDN's four input projections used to break vLLM's
    merged-parameter load path unconditionally (v2's AssertionError in
    load_merged_column_weight) -- see --quantize-gdn below, now that
    scripts/patch_vllm_gdn_quant_load.py fixes the loader for the case
    that matters: merging shards that share one quantization scheme.
  - lm_head, conv1d, norm, embed all excluded, same reasoning as v3 (not
    nn.Linear, or a near-zero-size/quality tradeoff not worth touching).

--quantize-gdn (default OFF, opt-in): narrows the GDN exclusion down to
just conv1d/norms and quantizes the four in_proj_* GDN projections too.
Requires scripts/patch_vllm_gdn_quant_load.py to be applied on the serving
side (it patches MergedColumnParallelLinear.weight_loader_v2 to handle
compressed-tensors packed shards, plus a weight_shape correctness fix --
see that script's docstring for the full root-cause writeup). That loader
fix only makes output-channel concatenation of independently-quantized
shards valid when every shard merged into ONE vLLM parameter shares the
exact same scheme (num_bits, group_size) -- concatenation happens along N
(the output/channel axis), while compressed-tensors' int4/int8 packing
lives along K (the input axis, confirmed by reading
CompressedTensorsWNA16.create_weights: weight_packed's packed_dim=1 !=
output_dim=0), so packed_K and the per-group scale width only match across
shards when the scheme matches. vLLM fuses GDN's four HF projections into
TWO params:
    in_proj_qkv + in_proj_z -> in_proj_qkvz  (must share ONE scheme)
    in_proj_b   + in_proj_a -> in_proj_ba    (must share ONE scheme)
but the two PAIRS are separate vLLM params, so they may use different
schemes from each other. This recipe therefore uses two mutually exclusive
config_groups when --quantize-gdn is set:
  - group_default (int4, group_size=32, the same scheme as everything
    else): q/k/v/o_proj, gate/up/down_proj, in_proj_qkv, in_proj_z,
    out_proj -- i.e. every Linear in the model except in_proj_b/in_proj_a.
  - group_ba (int8, group_size=128): in_proj_b and in_proj_a only. Kept at
    a coarser/higher-bit scheme deliberately -- vLLM's own comment on
    in_proj_ba's construction notes "ba_proj doesn't support blockwise fp8
    quantization" (qwen_gdn_linear_attn.py), and b/a feed the GDN gating
    nonlinearities (sigmoid/exp) directly, which are far more sensitive to
    quantization noise than a linear projection whose output goes through
    a norm; this project's fp8 sensitivity finding (STATUS.md: "toàn bộ
    độ nhạy fp8 của Qwen3.5 nằm ở đường GDN in_proj_ba") independently
    points the same way. Both groups use targets that explicitly name
    Linear submodules by suffix rather than a blanket ["Linear"] target,
    so the two config_groups can never both match the same module --
    avoids depending on unspecified group-precedence behavior when
    targets overlap.

--quantize-gdn-int8 (default OFF, opt-in, mutually exclusive with
--quantize-gdn): TASK G2a, a milder variant born from --quantize-gdn's
result (SWE-bench ppl ratio 1.156 vs the RedHatAI champion -- a WARN, not
a PASS, despite a big conc32 speed win). Hypothesis: int4 g32 on
in_proj_qkv/in_proj_z (the mixed scheme --quantize-gdn used) is the
quality-costly half of that recipe, since q/k/v feed attention-like score
computation and z is the gate multiplier -- both plausibly as noise-
sensitive as b/a already were shown to be (see --quantize-gdn's docstring
above and STATUS.md's fp8-sensitivity finding). This variant keeps ALL
FOUR GDN in_proj_* projections at the SAME coarser int8 g128 scheme
(instead of splitting qkv/z into int4 g32 and b/a into int8 g128), while
leaving every other Linear (attention proj, MLP) at the champion's int4
g32. Since compressed-tensors packing is along K (see above), int8 g128
on both fused pairs (in_proj_qkvz AND in_proj_ba) is still a same-scheme
merge on each pair, so the loader patch's constraint is satisfied exactly
as before -- no further loader changes needed. Expected outcome: GDN bytes
still shrink 2x vs fp16 (not 4x like int4), so decode throughput should
land between the fp16-GDN champion (563 tok/s @conc32) and the int4-GDN
attempt (768 tok/s @conc32); ppl should sit much closer to the champion
since int8 g128 is a materially gentler quantization than int4 g32 on the
projections that turned out to matter.

offload_hessians=True is passed unconditionally: GPTQ's Hessian accumulation
(the real thing this knob controls, unlike AWQModifier which has no Hessian
concept at all) is the memory-heavy step or a 9B model on a 23GB L4, and
llm-compressor's own docs recommend this default-on for anything above
~7B when calibration is not already comfortably fitting in VRAM headroom.

Usage:
    source /tmp/vllm_env.sh
    python scripts/quantize_gptq_9b.py
    # writes the checkpoint to /content/qwen35-9b-gptq (or $GPTQ_OUTPUT_DIR)

    python scripts/quantize_gptq_9b.py --quantize-gdn
    # also quantizes in_proj_qkv/in_proj_z (int4 g32) and in_proj_b/
    # in_proj_a (int8 g128). Apply scripts/patch_vllm_gdn_quant_load.py
    # on the serving side before loading this checkpoint.

Then scripts/fix_qwen35_hf_checkpoint.py on the output before serving, same
two mechanical fixes (language_model prefix + mrope_section) as every other
checkpoint this project has quantized from HF-saved Qwen3.5 weights.

--- Calibration speed knobs (measured on Colab L4, latency-bound) ---

Measured baseline: 256 samples x 2048 max_seq_len, batch_size=1 (the
defaults below, unchanged) -- calibration ran ~297s/layer x 33 layers
(~160 min total). nvidia-smi during that run showed ~12% GPU utilization
at 99% max clocks and no thermal throttling: this is NOT compute-bound,
it's per-sample Python/tokenize/Hessian-update overhead between forwards
dominating wall time. The fix is batching forwards, not a faster GPU.

llm-compressor's oneshot() has a genuine (non-patched) calibration-batching
path: `batch_size` and `data_collator` are real DatasetArguments fields
(llmcompressor/args/dataset_arguments.py), plumbed straight through
oneshot()'s kwargs into the calibration DataLoader
(llmcompressor/datasets/utils.py:make_dataset_splits / get_calibration_dataloader).
With the default data_collator="truncation", a batch's sequences are
truncated down to the SHORTEST sequence's length in that batch (not
padded) -- see DataCollatorWithTruncation in that file -- so batching does
NOT introduce padding tokens or a fabricated attention_mask into the
Hessian statistics; it only discards the tail of longer samples in a
batch, exactly like max_seq_length truncation already does at the single-
sample level. When batch_size > 1, `--calib-shuffle` should stay off
(shuffle_calibration_samples=False) so oneshot uses its LengthAwareSampler
(same file) to group similarly-long samples into the same batch --
minimizes how much of the longer samples' tails get truncated away.
Passing shuffle_calibration_samples=True with batch_size>1 instead uses a
plain RandomSampler and llm-compressor itself warns this "can lead to
unoptimal batching" / "delete a large number of tokens". This script's own
dataset-level `.shuffle(seed=42)` calls already randomize sample order
before calibration ever sees the data, so turning off oneshot's *internal*
shuffle-for-batching does not make the calibration set itself less random
-- it only controls how the 256 (or --num-samples) selected examples get
grouped into batches.

Recommended combinations (bs = --calib-batch-size). The "wall time" column
matches this script's own pre-flight estimate formula (~160 min *
ceil(num_samples / bs) / 256) so the printed estimate and this table never
disagree; that formula only scales with forward-CALL count, it does not
model per-forward compute growing with batch size or max_seq_len, so
treat every non-bs=1 estimate as a LOWER BOUND, not a promise:

  num_samples x max_seq_len, bs   | wall time (L4)          | quality
  ---------------------------------|--------------------------|-------------------
  256 x 2048, bs=1 (DEFAULT)       | ~160 min -- MEASURED     | verified baseline
                                    |                          | (every prior ppl
                                    |                          | number used this)
  128 x 1024, bs=8                 | ~10 min lower bound,     | UNVERIFIED -- run
                                    | budget ~15-25 min        | a 128-vs-256 ppl
                                    | wall-clock in practice   | A/B before trusting
                                    |                          | this for a real
                                    |                          | checkpoint

  Only the 256x2048/bs=1 row is what every existing SWE-bench ppl number
  in this project's history (v3's 6.9958, the fp16-GDN G run, G2a, etc.)
  was measured against. Smaller/batched calibration changes what the
  Hessian sees (fewer samples, shorter truncated-off tails, coarser
  activation statistics) and per llm-compressor's own docs GPTQ's
  Hessian-based scale/zero-point solve is more calibration-sensitive than
  most PTQ methods -- do not point a real quality-gated checkpoint run at
  the 128x1024/bs=8 row (or anything more aggressive) without first
  running both configurations back-to-back and comparing SWE-bench ppl.
  If they land close, adopt the fast config going forward; if not, the
  256x2048/bs=1 default remains the only trusted path.

Flags (all default to the exact prior behavior -- no flag = no change):
    --calib-batch-size N   (default 1)  -- passed straight to oneshot's
                            genuine batch_size kwarg above.
    --num-samples N        (default 256, this script's prior hardcoded
                            NUM_CALIBRATION_SAMPLES)
    --max-seq-len N         (default 2048, this script's prior hardcoded
                            MAX_SEQUENCE_LENGTH)

Before running, the script prints an expected-time estimate derived from
the measured ~1s/sample-forward, ~297s/(256/1)-layer baseline above, so
whoever kicks off a run can see roughly what they're buying before they
wait for it.
"""

import argparse
import os

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

parser = argparse.ArgumentParser()
gdn_group = parser.add_mutually_exclusive_group()
gdn_group.add_argument(
    "--quantize-gdn",
    action="store_true",
    default=False,
    help=(
        "Quantize GDN's in_proj_qkv/in_proj_z/in_proj_b/in_proj_a too, "
        "mixed scheme (qkv/z at int4 g32, b/a at int8 g128) -- TASK G's "
        "recipe (default: keep the whole linear_attn block fp16). Requires "
        "scripts/patch_vllm_gdn_quant_load.py on the serving side. See "
        "this script's module docstring for the scheme constraints."
    ),
)
gdn_group.add_argument(
    "--quantize-gdn-int8",
    action="store_true",
    default=False,
    help=(
        "TASK G2a: quantize all four GDN in_proj_* projections at a single "
        "uniform int8 g128 scheme (milder than --quantize-gdn's mixed "
        "int4/int8 split -- see module docstring). Requires "
        "scripts/patch_vllm_gdn_quant_load.py on the serving side."
    ),
)
parser.add_argument(
    "--calib-batch-size",
    type=int,
    default=1,
    help=(
        "TASK L: calibration forward batch size, passed straight to "
        "llm-compressor oneshot()'s genuine `batch_size` dataset argument "
        "(default 1 -- identical to every prior run's behavior). >1 uses "
        "oneshot's default data_collator='truncation', which truncates a "
        "batch's sequences down to the shortest member -- no padding, no "
        "fabricated attention_mask, so Hessian statistics stay clean. See "
        "the module docstring's speed-knob table before using >1 on a "
        "quality-gated run: only bs=1 has a verified SWE-bench ppl number."
    ),
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=256,
    help="Number of calibration samples (default 256, the verified value).",
)
parser.add_argument(
    "--max-seq-len",
    type=int,
    default=2048,
    help="Max calibration sequence length (default 2048, the verified value).",
)
args = parser.parse_args()

MODEL_ID = "Qwen/Qwen3.5-9B"
NUM_CALIBRATION_SAMPLES = args.num_samples
MAX_SEQUENCE_LENGTH = args.max_seq_len
CALIB_BATCH_SIZE = args.calib_batch_size
OUTPUT_DIR = os.environ.get("GPTQ_OUTPUT_DIR", "/content/qwen35-9b-gptq")

# TASK L, knob 3: rough pre-flight time estimate so whoever kicks off a run
# sees what they're buying before waiting for it. Anchored to the measured
# L4 baseline (256 samples, batch_size=1, max_seq_len=2048 -> ~160 min
# total, i.e. ~1s of wall time per calibration *sample-forward* at that
# per-layer overhead). Batching amortizes the per-forward Python/Hessian
# overhead across CALIB_BATCH_SIZE samples per forward call, so the
# forward-call count (and thus wall time) scales with
# num_samples / batch_size, not num_samples alone; shortening max_seq_len
# is not modeled here (compute-bound tail, not the latency-bound overhead
# this estimate targets) so it is intentionally left out of the scaling.
_BASELINE_SAMPLES = 256
_BASELINE_MINUTES = 160
_est_forward_calls = -(-NUM_CALIBRATION_SAMPLES // CALIB_BATCH_SIZE)  # ceil div
_baseline_forward_calls = _BASELINE_SAMPLES  # bs=1 baseline: 1 call/sample
_est_minutes = _BASELINE_MINUTES * _est_forward_calls / _baseline_forward_calls
print(
    f"[quantize_gptq_9b] calibration plan: num_samples={NUM_CALIBRATION_SAMPLES}, "
    f"batch_size={CALIB_BATCH_SIZE}, max_seq_len={MAX_SEQUENCE_LENGTH} "
    f"-> ~{_est_forward_calls} calibration forward calls/layer, expected "
    f"~{_est_minutes:.0f} min on L4 at the measured ~1s/sample-forward rate "
    f"(extrapolated from the 256x2048/bs=1 baseline; NOT compute-model-"
    f"adjusted for max_seq_len changes). See module docstring for which "
    f"combinations have a verified SWE-bench ppl number vs. which are "
    f"speed-only extrapolations."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

n_each = NUM_CALIBRATION_SAMPLES // 2

chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
chat_ds = chat_ds.shuffle(seed=42).select(range(n_each))


def preprocess_chat(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


chat_ds = chat_ds.map(preprocess_chat, remove_columns=chat_ds.column_names)

code_ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
code_ds = code_ds.shuffle(seed=42).select(range(n_each))


def preprocess_code(example):
    messages = [
        {"role": "user", "content": (example["instruction"] + "\n\n" + example["input"]).strip()},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


code_ds = code_ds.map(preprocess_code, remove_columns=code_ds.column_names)

ds = concatenate_datasets([chat_ds, code_ds]).shuffle(seed=42)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

if args.quantize_gdn:
    # Two mutually exclusive, name-based target lists -- see the
    # --quantize-gdn section of the module docstring for why in_proj_b/
    # in_proj_a need their own (coarser) scheme while everything else,
    # including in_proj_qkv/in_proj_z, shares group_default.
    config_groups = {
        "group_default": QuantizationScheme(
            targets=[
                "re:.*\\.(q_proj|k_proj|v_proj|o_proj"
                "|gate_proj|up_proj|down_proj"
                "|in_proj_qkv|in_proj_z|out_proj)$"
            ],
            weights=QuantizationArgs(
                num_bits=4,
                group_size=32,
                strategy="group",
                symmetric=True,
            ),
        ),
        "group_ba": QuantizationScheme(
            targets=["re:.*\\.(in_proj_b|in_proj_a)$"],
            weights=QuantizationArgs(
                num_bits=8,
                group_size=128,
                strategy="group",
                symmetric=True,
            ),
        ),
    }
    # Only conv1d/norm/embed (not nn.Linear -- irrelevant to quantization
    # anyway) plus lm_head stay excluded; every in_proj_* is now covered by
    # one of the two config_groups above instead of being ignored wholesale.
    ignore = [
        "lm_head",
        "re:.*conv1d.*",
        "re:.*norm.*",
        "re:.*embed.*",
    ]
elif args.quantize_gdn_int8:
    # TASK G2a: uniform int8 g128 across ALL FOUR GDN in_proj_* projections
    # (both fused pairs), everything else stays at the champion's int4 g32.
    # Each fused pair (in_proj_qkvz, in_proj_ba) is still internally
    # same-scheme, so the merged-column loader patch's constraint holds.
    config_groups = {
        "group_default": QuantizationScheme(
            targets=[
                "re:.*\\.(q_proj|k_proj|v_proj|o_proj"
                "|gate_proj|up_proj|down_proj|out_proj)$"
            ],
            weights=QuantizationArgs(
                num_bits=4,
                group_size=32,
                strategy="group",
                symmetric=True,
            ),
        ),
        "group_gdn_int8": QuantizationScheme(
            targets=[
                "re:.*\\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)$"
            ],
            weights=QuantizationArgs(
                num_bits=8,
                group_size=128,
                strategy="group",
                symmetric=True,
            ),
        ),
    }
    ignore = [
        "lm_head",
        "re:.*conv1d.*",
        "re:.*norm.*",
        "re:.*embed.*",
    ]
else:
    config_groups = {
        "group_default": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                group_size=32,
                strategy="group",
                symmetric=True,
            ),
        ),
    }
    ignore = [
        "lm_head",
        "re:.*linear_attn.*",
        "re:.*conv1d.*",
        "re:.*norm.*",
        "re:.*embed.*",
    ]

recipe = [
    GPTQModifier(
        config_groups=config_groups,
        ignore=ignore,
        offload_hessians=True,
    ),
]

oneshot(
    model=MODEL_ID,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir=OUTPUT_DIR,
    batch_size=CALIB_BATCH_SIZE,
    # dataset-level .shuffle(seed=42) calls above already randomized sample
    # order/selection; disabling oneshot's *internal* shuffle-for-batching
    # when batch_size > 1 switches it from RandomSampler to
    # LengthAwareSampler, which groups similarly-long samples into the same
    # batch so the default truncation-based collator (see module docstring)
    # discards as little of each sample's tail as possible. At batch_size=1
    # this has no effect (LengthAwareSampler with batch_size=1 samples in
    # descending-length order, RandomSampler samples randomly, and no
    # truncation-to-shortest-in-batch happens either way).
    shuffle_calibration_samples=(CALIB_BATCH_SIZE == 1),
)

print("DONE_GPTQ_QUANTIZE_9B")
