"""
Custom Qwen3.5-9B AWQ checkpoint, targeting a beat of RedHatAI's generic
RedHatAI/Qwen3.5-9B-quantized.w4a16 recipe.

RedHatAI's recipe (read directly from their published config.json's
quantization_config): a single config_group, every nn.Linear (GDN
in_proj_a/in_proj_b included, undifferentiated) at int4, group_size=128,
symmetric, strategy=group, format=pack-quantized, actorder=static -- i.e.
plain GPTQ-style uniform 4-bit with a 128-wide group and activation
reordering, no per-layer sensitivity handling.

Our edge, per this project's own accumulated knowledge (STATUS.md's kernel
profile + the GGUF format-sweep quality table): not every Linear in this
architecture is equally forgiving of 4-bit noise, and finer groups buy back
precision at a small size cost. This recipe differentiates:

  group_gdn:     GDN's four input projections (in_proj_a, in_proj_b,
                 in_proj_qkv, in_proj_z) -- int8, group_size=128
  group_default: every other Linear -- int4, group_size=32 (finer than
                 RedHatAI's 128, more scale granularity per weight)

NOTE (found empirically, not part of the original design): a first attempt
scoped group_gdn to ONLY in_proj_a/in_proj_b (matching the task spec's
literal wording) and crashed inside AWQModifier's grid search with
`RuntimeError: The size of tensor a (32) must match the size of tensor b
(128) at non-singleton dimension 1`. Root cause, confirmed against the AWQ
mappings TEST 10 already recorded in recipe.yaml for the 2B checkpoint:
AWQModifier's auto-resolved mapping smooths all FOUR of a GDN layer's input
projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) jointly under one
`smooth_layer: input_layernorm` / `balance_layers: [...]` group, because
they all read the same input_layernorm output. The grid search that picks
the shared smoothing scale fake-quantizes every balance_layer with each
trial scale to measure reconstruction error -- which requires every
balance_layer in the group to share one quantization scheme. Ours didn't:
in_proj_a/b were group_gdn (g128) while in_proj_qkv/z fell through to
group_default (g32), so the fake-quantize step tried to apply a g128 scale
tensor to a g32-shaped weight chunk mid grid-search. Splitting a
jointly-smoothed AWQ mapping across two different quantization group_sizes
is a structural conflict, not a config typo -- the fix widens group_gdn to
cover all four co-smoothed tensors instead of trying to isolate two of them.

Two things this recipe explicitly does NOT change from the 2B script
(scripts/quantize_awq_2b.py), verified against Qwen/Qwen3.5-9B's actual
model.safetensors.index.json before writing this docstring (never assumed):

  - GDN naming is confirmed HF-side as
    model.language_model.layers.N.linear_attn.in_proj_a.weight /
    ...in_proj_b.weight (separate tensors, matching the 2B checkpoint's
    layout) -- hence the group_gdn regex now covers all four sibling
    tensors, `re:.*linear_attn\\.in_proj_(a|b|qkv|z)$`, per the note above.
    This is a wider net than the 2B script's `re:.*linear_attn.*` exclusion
    (which kept ALL of linear_attn fp16, avoiding the fused
    in_proj_qkvz/in_proj_ba layout question entirely) -- here all four input
    projections DO get quantized, at int8 rather than being skipped. Verify
    PPL wasn't in fact traded away by this widening of scope -- that's
    exactly the swebench-ppl gate at the bottom of the runbook this script's
    docstring points to.)
  - The checkpoint has a substantial vision tower: 333 of Qwen/Qwen3.5-9B's
    775 safetensors keys are model.visual.* (confirmed via the same index.json
    read -- Qwen/Qwen3.5-2B carries one too, 297/632 keys, and the 2B AWQ
    recipe quantized it anyway with the plain "everything Linear" rule
    without incident). Left in scope here for consistency with that
    precedent; vLLM serves this checkpoint through the text-only
    Qwen3_5ForCausalLM path (see patch_vllm_qwen35_registry.py) so the vision
    tower's quantized weights are simply unused dead weight on disk, not a
    correctness risk.

conv1d/A_log/dt_bias/norm/embed weights are not nn.Linear, so
QuantizationModifier's targets=["Linear"] already leaves them alone --
same reasoning as the 2B script. lm_head is explicitly ignored (kept
fp16) as before.

Memory note: no Hessian computation happens in this pipeline (that is a
GPTQModifier concept; this recipe pairs AWQModifier's activation-weighted
scale search with plain RTN quantization via QuantizationModifier, same as
the 2B script -- there is no offload_hessians knob to reach for here).
llm-compressor's oneshot() defaults to pipeline="independent" with
sequential_offload_device="cpu", i.e. layer-by-layer sequential processing
with CPU offload already built in -- no extra flags needed for a 9B model on
a 23GB L4 as long as the default pipeline is left alone. If calibration OOMs
regardless, the first lever is num_calibration_samples (256 -> 128), not a
pipeline change.

Usage:
    source /tmp/vllm_env.sh   # LD_LIBRARY_PATH for CUDA runtime libs
    python scripts/quantize_awq_9b.py
    # writes the checkpoint to /content/qwen35-9b-awq (or $AWQ_OUTPUT_DIR)

Then run scripts/fix_qwen35_hf_checkpoint.py on the output before serving
(same two mechanical bugs as the 2B checkpoint: language_model prefix
mismatch + mrope_section false-positive) -- see that script's docstring and
STATUS.md's TEST 10 notes for the full runbook this one continues.
"""

import os

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

MODEL_ID = "Qwen/Qwen3.5-9B"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT_DIR", "/content/qwen35-9b-awq")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Dict insertion order = resolution priority: group_gdn's specific regex is
# checked before group_default's blanket "Linear" catch-all, so all four
# in_proj_* land in the int8 group and everything else Linear falls through
# to int4. Must cover all four siblings (a/b/qkv/z), not just a/b -- see the
# module docstring's NOTE for why splitting AWQ's jointly-smoothed group
# across two quantization group_sizes crashes the scale grid search.
config_groups = {
    "group_gdn": QuantizationScheme(
        targets=[r"re:.*linear_attn\.in_proj_(a|b|qkv|z)$"],
        weights=QuantizationArgs(
            num_bits=8,
            group_size=128,
            strategy="group",
            symmetric=True,
        ),
    ),
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

recipe = [
    AWQModifier(),
    QuantizationModifier(
        config_groups=config_groups,
        ignore=["lm_head", "re:.*conv1d.*", "re:.*norm.*", "re:.*embed.*"],
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

print("DONE_AWQ_QUANTIZE_9B")
