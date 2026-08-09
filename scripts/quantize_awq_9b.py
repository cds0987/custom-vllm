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

  group_gdn:     GDN's in_proj_a / in_proj_b -- int8, group_size=128
  group_default: every other Linear          -- int4, group_size=32 (finer
                  than RedHatAI's 128, more scale granularity per weight)

Two things this recipe explicitly does NOT change from the 2B script
(scripts/quantize_awq_2b.py), verified against Qwen/Qwen3.5-9B's actual
model.safetensors.index.json before writing this docstring (never assumed):

  - GDN naming is confirmed HF-side as
    model.language_model.layers.N.linear_attn.in_proj_a.weight /
    ...in_proj_b.weight (separate tensors, matching the 2B checkpoint's
    layout) -- hence the group_gdn regex `re:.*linear_attn\\.in_proj_[ab]$`.
    in_proj_qkv and in_proj_z are separate tensors too but are NOT part of
    this group -- they fall through to group_default like every other
    Linear, same as the 2B recipe's blanket int4 treatment of everything
    outside its `re:.*linear_attn.*` exclusion. (The 2B script excluded ALL
    of linear_attn.* from quantization entirely, staying fp16, because its
    goal was avoiding the fused in_proj_qkvz/in_proj_ba layout mismatch
    entirely. This script takes the differentiated approach instead:
    in_proj_a/b get their own int8 group rather than being skipped, and
    in_proj_qkv/in_proj_z quantize under the same int4 default as everything
    else. Verify PPL wasn't in fact traded away by this widening of scope --
    that's exactly the swebench-ppl gate at the bottom of the runbook this
    script's docstring points to.)
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
# checked before group_default's blanket "Linear" catch-all, so in_proj_a/b
# land in the int8 group and everything else Linear falls through to int4.
config_groups = {
    "group_gdn": QuantizationScheme(
        targets=[r"re:.*linear_attn\.in_proj_[ab]$"],
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
