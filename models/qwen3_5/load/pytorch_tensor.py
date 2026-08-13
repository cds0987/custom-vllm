"""Load path #3: PYTORCH TENSOR — serve a safetensors checkpoint as-is.

For checkpoints already in HF safetensors form (bf16, or pre-quantized
compressed-tensors/GPTQ/AWQ). No conversion; vLLM's native loaders handle it.
This is the 27B path: apolo13x/Qwen3.5-27B-quantized.w4a16 ships GDN already
quantized, so it serves directly (ppl 4.1484 — no graft needed).

One patch matters here: engine/vllm/patches/patch_vllm_gdn_quant_load.py
teaches vLLM to load QUANTIZED GDN in_proj weights (upstream assumes bf16).
Frames that keep GDN in bf16 (RedHat 9B style) also load fine, but then
path #1 (gguf_to_marlin) exists to quantize the GDN and win back VRAM/quality.
"""

ADAPTER = {
    "axis": "load",
    "variant": "pytorch_tensor",
    "requires": {},
    "input": "HF safetensors dir (bf16 or compressed-tensors/GPTQ/AWQ)",
    "output": "served directly, no artifact",
    "tradeoff": "no prep, quality/speed is whatever the checkpoint author achieved",
}
