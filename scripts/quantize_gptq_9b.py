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
args = parser.parse_args()

MODEL_ID = "Qwen/Qwen3.5-9B"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
OUTPUT_DIR = os.environ.get("GPTQ_OUTPUT_DIR", "/content/qwen35-9b-gptq")

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
)

print("DONE_GPTQ_QUANTIZE_9B")
