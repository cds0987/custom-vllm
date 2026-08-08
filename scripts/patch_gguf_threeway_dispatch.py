"""
Opt-in third dispatch lane: keep the Triton GEMM for mid-size batches even
when the compiled CUDA extension is available.

TEST 8 (L4, sm89) measured the three kernel families against each other once
_C_gguf (llama.cpp's CUDA mmvq/mmq ported into the plugin) was built from
sdist and actually loaded:

    decode tok/s     conc1   conc4   conc16  conc32
    CUDA kernels     128.8   332.4   503.7   539.8
    Triton fused      33.0   127.9   475.8   846.8
    (prefill: CUDA mmq collapses entirely — 403 vs 10,496 total tok/s —
     already routed away by the hybrid-dispatch patch's >=1024 threshold)

Each family owns one batch regime: CUDA mmvq wins small M by 2.6-3.9x
(it is a real GEMV; the Triton path pays a full BLOCK_M=32 tile at M=1),
Triton mmq wins mid M (2 warps but better tiling than the ported mmq at
conc32, +57%), dequant+cuBLAS owns prefill. The stock dispatch only knows
two lanes: mmvq_safe splits vec-vs-matrix, and with CUDA enabled BOTH
branches prefer the CUDA op — so mid-M silently takes the losing kernel.

This patch adds the missing lane: under CUSTOM_VLLM_GGUF_TRITON_MID=1, the
MMQ branch calls ops.ggml_mul_mat_a8_triton directly instead of
ops.ggml_mul_mat_a8, bypassing the CUDA preference for mid-size batches
only. Combined with CUSTOM_VLLM_GGUF_HYBRID=1 the full routing becomes:

    M <= mmvq_safe (2-6)      CUDA mmvq        (3.9x at conc1)
    mmvq_safe < M < 1024      Triton mmq       (846.8 at conc32)
    M >= 1024                 dequant+cuBLAS   (10,496 tok/s prefill)

Default off: without the env var the MMQ branch is byte-identical to the
stock behavior, so A/B testing stays honest. Requires no particular patch
ordering beyond running after the plugin install; anchors on the stock MMQ
branch, which none of the other patches rewrite.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: third lane — Triton for mid-M even when CUDA kernels exist ---"

ANCHOR = """    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
"""

PATCH = f"""    elif qweight_type in MMQ_QUANT_TYPES:
        {PATCH_MARKER}
        if _CUSTOM_VLLM_TRITON_MID:
            y = ops.ggml_mul_mat_a8_triton(qweight, x, qweight_type, qweight.shape[0])
        else:
            y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
"""

FLAG_ANCHOR = "def _fused_mul_mat_gguf(\n"
FLAG_PATCH = (
    f"{PATCH_MARKER}\n"
    "import os as _custom_vllm_os_mid\n\n"
    "_CUSTOM_VLLM_TRITON_MID = (\n"
    '    _custom_vllm_os_mid.environ.get("CUSTOM_VLLM_GGUF_TRITON_MID", "0") == "1"\n'
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
