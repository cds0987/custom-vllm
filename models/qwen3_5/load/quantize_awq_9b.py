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
precision at a small size cost. FINAL shipped recipe (v3, after two
empirical dead ends recorded below): a single config_group, int4
group_size=32 (finer than RedHatAI's 128) on every Linear EXCEPT GDN's
linear_attn.* (kept fp16, same scope as the 2B script -- see the v2 NOTE
below for why) and lm_head. GDN was never the target of the "our edge"
premise anyway: STATUS.md's kernel profile already measured GDN/fla at
2-12% of decode CUDA time, so leaving it fp16 costs essentially nothing,
and the genuine differentiation from RedHatAI survives untouched on the
part of the model that actually dominates decode time (attention Q/K/V/O
and MLP gate/up/down, all now g32 instead of their g128).

NOTE v1->v2 (found empirically, not part of the original design): a first
attempt tried giving GDN's in_proj_a/in_proj_b their own int8 group_size=128
scheme (matching the task spec's literal wording) and crashed inside
AWQModifier's grid search with `RuntimeError: The size of tensor a (32) must
match the size of tensor b (128) at non-singleton dimension 1`. Root cause,
confirmed against the AWQ mappings TEST 10 already recorded in recipe.yaml
for the 2B checkpoint: AWQModifier's auto-resolved mapping smooths all FOUR
of a GDN layer's input projections (in_proj_qkv, in_proj_z, in_proj_b,
in_proj_a) jointly under one `smooth_layer: input_layernorm` /
`balance_layers: [...]` group, because they all read the same
input_layernorm output. The grid search that picks the shared smoothing
scale fake-quantizes every balance_layer with each trial scale to measure
reconstruction error -- which requires every balance_layer in the group to
share one quantization scheme. Splitting a jointly-smoothed AWQ mapping
across two different quantization group_sizes is a structural conflict, not
a config typo. v2's fix: widen the GDN group to int8 g128 across all four
co-smoothed tensors instead of trying to isolate two of them.

NOTE v2->v3 (found empirically, one level deeper): v2's quantize step
completed cleanly and produced a checkpoint, but vLLM crashed AT SERVE TIME
loading it -- `AssertionError` in vllm/model_executor/parameter.py:175,
inside `load_merged_column_weight`. Root cause: vLLM's
Qwen3_5Model.hf_to_vllm_mapper doesn't just rename GDN's four input
projections, it MERGES pairs of them into fused parameters via
orig_to_new_stacked (in_proj_qkv+in_proj_z -> in_proj_qkvz,
in_proj_b+in_proj_a -> in_proj_ba). That merge path is built to concatenate
RAW (unquantized) shards into one destination tensor at load time -- it
cannot reassemble two INDEPENDENTLY quantized-and-packed compressed-tensors
modules (each with its own separately-computed weight_packed/weight_scale/
weight_shape from having been quantized as a standalone Linear); the shapes
don't line up and the assert fires before a single token generates. This is
exactly the constraint the 2B script's docstring already flagged when it
excluded linear_attn.* wholesale ("avoiding the fused in_proj_qkvz/
in_proj_ba layout mismatch entirely") -- it wasn't merely conservative, it
was load-bearing: quantizing ANY of GDN's four input projections at all,
regardless of bit width, breaks vLLM's merged-load path for this
architecture as it exists today. v3's fix: revert to the 2B script's full
`re:.*linear_attn.*` exclusion.

Two things this recipe carries over unchanged from the 2B script
(scripts/quantize_awq_2b.py), verified against Qwen/Qwen3.5-9B's actual
model.safetensors.index.json before writing this docstring (never assumed):

  - GDN naming is confirmed HF-side as
    model.language_model.layers.N.linear_attn.in_proj_a.weight /
    ...in_proj_b.weight (separate tensors, matching the 2B checkpoint's
    layout, plus sibling in_proj_qkv/in_proj_z tensors) -- all four covered
    by the `re:.*linear_attn.*` ignore, per the v3 NOTE above.
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

# SECOND empirical finding (v2 of this recipe, after the group_gdn-covers-all-
# four fix): quantizing the GDN input projections at ALL -- even uniformly at
# int8, all four siblings together -- crashes vLLM at SERVE time, not quantize
# time. vLLM's Qwen3_5Model.hf_to_vllm_mapper doesn't just rename these
# tensors, it MERGES pairs of them into fused parameters via
# orig_to_new_stacked: in_proj_qkv+in_proj_z -> in_proj_qkvz,
# in_proj_b+in_proj_a -> in_proj_ba (see vllm/model_executor/models/qwen3_5.py).
# That merge goes through MergedColumnParallelLinear's weight_loader_v2 ->
# load_merged_column_weight, which asserts the incoming shard's shape matches
# the destination parameter's shape -- a check written for concatenating RAW
# (unquantized) shards into one merged tensor at load time, not for
# reassembling two INDEPENDENTLY quantized-and-packed compressed-tensors
# modules (each with its own weight_packed/weight_scale/weight_shape from
# being quantized as a standalone Linear). The shapes don't line up and it
# throws `AssertionError` in parameter.py:175, before a single token is ever
# generated. This is exactly the constraint the 2B script's docstring already
# flagged when it excluded linear_attn.* wholesale ("avoiding the fused
# in_proj_qkvz/in_proj_ba layout mismatch entirely") -- it wasn't merely
# conservative, it was load-bearing. Reverting to that same full exclusion
# here. GDN is not the bottleneck to chase anyway: STATUS.md's kernel profile
# already measured GDN/fla at 2-12% of decode CUDA time, so fp16 GDN costs
# essentially nothing. The genuine point of differentiation from RedHatAI's
# recipe survives unchanged: int4 group_size=32 (vs their 128) on every OTHER
# Linear -- attention Q/K/V/O and the MLP gate/up/down projections.
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

recipe = [
    AWQModifier(),
    QuantizationModifier(
        config_groups=config_groups,
        ignore=[
            "lm_head",
            "re:.*linear_attn.*",
            "re:.*conv1d.*",
            "re:.*norm.*",
            "re:.*embed.*",
        ],
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
