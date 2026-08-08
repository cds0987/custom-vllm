"""
Opt-in: route GGUF matmuls through dequantise + cuBLAS instead of the fused
Triton kernels.

_fused_mul_mat_gguf() picks between three strategies:

    mmvq   (ggml_mul_mat_vec_a8)  for small batches
    mmq    (ggml_mul_mat_a8)      for larger ones
    dequant + `x @ weight.T`      only for types the first two don't cover

K-quants are members of both MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES, so the
dequant path is unreachable for them even though it is often much faster here:
without llama.cpp's CUDA kernels the plugin's fused ops fall back to Triton
(ops.ggml_mul_mat_a8 -> ggml_mul_mat_a8_triton), whose per-call overhead
dominates decode. Measured on a T4 with Qwen3.5-2B, one token at a time:

    GGUF Q4_K_M via Triton fused kernels    0.92 tok/s
    same model, fp16 safetensors, cuBLAS   55.1  tok/s

That 60x gap is kernel overhead, not arithmetic — throughput scaled almost
linearly with batch size, i.e. each decode step paid a near-constant cost
regardless of how much work it carried.

Dequantising into a scratch buffer and calling cuBLAS keeps the weights 4-bit
in VRAM (only the transient buffer is fp16), so it trades a little bandwidth
per step for a well-optimised GEMM. Whether that wins depends on the shape mix,
hence the env switch rather than an unconditional change:

    CUSTOM_VLLM_GGUF_DEQUANT=1 vllm serve ...
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: optional dequant + cuBLAS instead of the fused Triton kernels ---"

ANCHOR = (
    "    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:\n"
)
PATCH = (
    f"    {PATCH_MARKER}\n"
    "    if _CUSTOM_VLLM_PREFER_DEQUANT and qweight_type in DEQUANT_TYPES:\n"
    "        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]\n"
    "        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)\n"
    "        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)\n"
    "        return x @ weight.T\n"
    + ANCHOR
)

FLAG_ANCHOR = "def _fused_mul_mat_gguf(\n"
FLAG_PATCH = (
    f"{PATCH_MARKER}\n"
    "import os as _custom_vllm_os\n\n"
    "_CUSTOM_VLLM_PREFER_DEQUANT = (\n"
    '    _custom_vllm_os.environ.get("CUSTOM_VLLM_GGUF_DEQUANT", "0") == "1"\n'
    ")\n\n\n"
    + FLAG_ANCHOR
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/quantization/linear.py")
if not matches:
    raise SystemExit(f"vllm_gguf_plugin/quantization/linear.py not found under {site_packages}")
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
else:
    for anchor in (ANCHOR, FLAG_ANCHOR):
        if anchor not in src:
            raise SystemExit(f"Anchor not found in {path}; plugin source may have changed")
    src = src.replace(FLAG_ANCHOR, FLAG_PATCH, 1).replace(ANCHOR, PATCH, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
