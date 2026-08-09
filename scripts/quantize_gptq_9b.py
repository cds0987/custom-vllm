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
  - `re:.*linear_attn.*` stays fully excluded (fp16). Quantizing ANY of
    GDN's four input projections, at ANY bit width or algorithm, breaks
    vLLM's merged-parameter load path for this architecture (v2's
    AssertionError in load_merged_column_weight) -- this is a hard
    constraint of the serving code, not something GPTQ escapes by being a
    different algorithm from AWQ.
  - lm_head, conv1d, norm, embed all excluded, same reasoning as v3 (not
    nn.Linear, or a near-zero-size/quality tradeoff not worth touching).

offload_hessians=True is passed unconditionally: GPTQ's Hessian accumulation
(the real thing this knob controls, unlike AWQModifier which has no Hessian
concept at all) is the memory-heavy step or a 9B model on a 23GB L4, and
llm-compressor's own docs recommend this default-on for anything above
~7B when calibration is not already comfortably fitting in VRAM headroom.

Usage:
    source /tmp/vllm_env.sh
    python scripts/quantize_gptq_9b.py
    # writes the checkpoint to /content/qwen35-9b-gptq (or $GPTQ_OUTPUT_DIR)

Then scripts/fix_qwen35_hf_checkpoint.py on the output before serving, same
two mechanical fixes (language_model prefix + mrope_section) as every other
checkpoint this project has quantized from HF-saved Qwen3.5 weights.
"""

import os

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

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
    GPTQModifier(
        config_groups=config_groups,
        ignore=[
            "lm_head",
            "re:.*linear_attn.*",
            "re:.*conv1d.*",
            "re:.*norm.*",
            "re:.*embed.*",
        ],
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
