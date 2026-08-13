"""Load path #2: PURE GGUF — serve any unsloth/llama.cpp GGUF file directly.

No conversion step: vllm-gguf-plugin reads the GGUF at load time. This is the
slowest path (~2x slower decode than Marlin: mixed Triton/dequant kernels) but
the only one that accepts ANY quant flavor (Q3_K, IQ4, UD-*...) with zero prep.

    vllm serve <file.gguf> --load-format gguf ...

Requires engine=vllm: the 14 patch_gguf_* patches in ../engine/vllm/patches
exist precisely to make this path work for Qwen3.5's hybrid GDN tensors.
Pick this path for: quick quality A/B of a new GGUF quant before investing in
conversion. Pick gguf_to_marlin (path #1) for anything you intend to serve.
"""

ADAPTER = {
    "axis": "load",
    "variant": "pure_gguf",
    "requires": {"engine": "vllm"},  # needs vllm-gguf-plugin + its patch set
    "input": "any .gguf file",
    "output": "served directly, no artifact",
    "tradeoff": "zero prep, ~2x slower decode than Marlin paths",
}
