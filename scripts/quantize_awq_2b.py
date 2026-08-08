"""
TEST 10: build a Qwen3.5-2B AWQ W4A16 checkpoint with llm-compressor.

Rationale: no 2B AWQ checkpoint exists on the Hub for Qwen3.5 (only 4B/9B,
e.g. QuantTrio/Qwen3.5-4B-AWQ). This script produces one via the official
AWQ algorithm (activation-weighted scale search + RTN quantize), as opposed
to the project's own transcode_gguf_to_gptq.py which is a noisier RTN-only
path with no calibration. AWQ output serves through vLLM via Marlin, the
same fast int4-tensor-core kernel that made QuantTrio's 4B AWQ hit
757 tok/s @ conc32 on L4 -- if the 2B version clears that same path and
holds quality, it becomes a genuine "low-bit + speed, no compromise"
candidate to sit alongside (or replace) the GGUF three-way-dispatch champion.

Recipe:
  1. AWQModifier() -- transform-only step. Auto-resolves smooth/balance
     layer mappings from the model architecture (Qwen3_5ForCausalLM is
     natively registered in llm-compressor's mapping table, including the
     hybrid GDN blocks), searches activation-weighted per-channel scales via
     grid search over calibration data, and folds them into the weights.
     Does not quantize on its own -- must be paired with a QuantizationMixin
     modifier below to actually produce a compressed checkpoint.
  2. QuantizationModifier(scheme="W4A16", targets=["Linear"],
     ignore=["lm_head", "re:.*linear_attn.*"]) -- does the actual RTN
     quantization to 4-bit weights / 16-bit activations on every nn.Linear
     except lm_head (kept fp16, standard practice -- output head is small
     and quantizing it costs quality for near-zero size/speed win) and any
     linear_attn.* module.

     The linear_attn exclusion needs explaining since it's the one
     Qwen3.5-specific choice in this recipe: HF's Qwen3.5 implementation
     names the hybrid GDN block's input projections in_proj_a / in_proj_b
     (separate tensors), but vLLM's Qwen3.5 model code (and this repo's GGUF
     path) expects the FUSED in_proj_ba layout AWQ/vLLM's own upstream
     example for the sibling Qwen3-Next-Thinking model uses this exact
     ignore pattern for exactly this reason -- quantizing linear_attn here
     would produce a checkpoint whose fused/unfused layout does not line up
     with what vLLM's loader expects. Conv1d, A_log, dt_bias, and norm
     weights are not nn.Linear so QuantizationModifier's targets=["Linear"]
     already leaves them alone; no separate ignore needed for those. This is
     conservative v1 scope: it leaves the whole hybrid-GDN mixer stack (attn
     Q/K/V/O aside -- those ARE plain nn.Linear and DO get quantized) at
     fp16 rather than attempting the fused-layout quantization this repo's
     GGUF transcode path (scripts/transcode_gguf_to_gptq.py) already had to
     solve by hand.

Calibration: 256 samples from HuggingFaceH4/ultrachat_200k (chat-formatted
via the model's own template), truncated to 2048 tokens. 256 samples /
2048 tokens is llm-compressor's own common default for W4A16 AWQ recipes --
enough for the activation-scale grid search to converge without the
calibration pass itself becoming the long pole (it is already the single
slowest step: one forward pass per sample, sequential, on a 2B model this
still runs in single-digit minutes on an L4).

Usage:
    source /tmp/vllm_env.sh   # LD_LIBRARY_PATH for CUDA runtime libs
    python scripts/quantize_awq_2b.py
    # writes the AWQ checkpoint to /content/qwen35-2b-awq (or $AWQ_OUTPUT_DIR)

Then serve with vLLM; it should auto-select Marlin (verify "MarlinLinearKernel"
in the startup log, matching the QuantTrio/Qwen3.5-4B-AWQ precedent already
measured in this project) -- see STATUS.md TEST 10 notes for the exact serve
invocation and results.

Known environment wrinkle: `pip install llmcompressor` currently pulls in
compressed-tensors 0.17.1, one patch ahead of the 0.17.0 vllm 0.26.0 pins.
This has not broken anything observed so far (quantize + save + serve all
worked), but if a future vllm/llmcompressor bump does break compatibility,
pin compressed-tensors back to whatever vllm's requirements file specifies
after the quantize step (the saved checkpoint's format does not depend on
which compressed-tensors version wrote it, only on vllm's ability to read it
at serve time).
"""

import os

from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.transformers import oneshot

MODEL_ID = "Qwen/Qwen3.5-2B"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT_DIR", "/content/qwen35-2b-awq")

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

recipe = [
    AWQModifier(),
    QuantizationModifier(
        ignore=["lm_head", "re:.*linear_attn.*"],
        scheme="W4A16",
        targets=["Linear"],
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

print("DONE_AWQ_QUANTIZE")
