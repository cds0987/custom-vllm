"""
Opt-in: unsloth-style scratch-buffer reuse for the Triton GGUF dequant path,
so the prefill dequant+cuBLAS route (patch_gguf_prefer_dequant.py /
patch_gguf_hybrid_dispatch.py) stops allocating a fresh fp16 weight tensor on
every single forward call.

Where the allocation actually happens
--------------------------------------
The hot path added by the two patches above is, in
vllm_gguf_plugin/quantization/linear.py:

    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
    return x @ weight.T

For a 4096x4096 layer in fp16 that's a fresh ~32MB tensor per call, discarded
right after the GEMM reads it. ops.ggml_dequantize() itself
(vllm_gguf_plugin/ops.py) is just a dispatcher — it does no allocation of its
own:

    def ggml_dequantize(W, quant_type, m, n, dtype=None):
        if _cuda_kernel_available("ggml_dequantize", quant_type):
            return torch.ops._C_gguf.ggml_dequantize(W, quant_type, m, n, dtype)
        return ggml_dequantize_triton(W, quant_type, m, n, dtype)

The CUDA branch is a compiled custom op (torch::empty happens inside C++);
there is no `out=` parameter on its schema, and patching a compiled extension
is out of scope for a source patch. But per patch_gguf_prefer_dequant.py's
own measurements, the deployments this repo targets (T4/L4 Colab installs of
vllm-gguf-plugin) run *without* that compiled extension — "without llama.cpp's
CUDA kernels the plugin's fused ops fall back to Triton" — so the branch that
actually matters here is ggml_dequantize_triton(), which dispatches by quant
type to one of 20 per-type wrapper functions (q4_k, q6_k, iq4_nl, ...). Every
one of those 20 wrappers is a thin shim that calls a single shared function,
run_dequantize_kernel(), in
vllm_gguf_plugin/triton/dequantize/utils.py:

    def run_dequantize_kernel(kernel, W, m, n, dtype, quant_type, extra_args=()):
        W, total, dtype = validate_dequant_args(W, quant_type, m, n, dtype)
        Y = torch.empty((m, n), device=W.device, dtype=dtype)
        ...
        kernel[grid](W, Y, total, *extra_args, BLOCK_SIZE=..., num_warps=4)
        return Y

That `Y = torch.empty(...)` is the one and only allocation site for every
Triton dequant kernel in the plugin. It is a far better patch target than the
linear.py call site: linear.py and ops.ggml_dequantize both just forward
arguments, so patching them cannot avoid the allocation, only relocate where
it's requested from. This file is the actual "wrapper level" — the thing
unsloth's own buffer trick wraps in kernels/utils.py:611-632 (module-level
buffer store, lazily created, grown via `resize_`, sliced+viewed instead of
reallocated) is exactly this kind of shared dequant-into-Y launcher.

Why per-(device, shape, dtype), not one global buffer
-------------------------------------------------------
Unsloth keeps exactly one buffer per device (WEIGHT_BUFFERS[device_index]),
sized to the largest weight seen and resized up as needed, because bitsandbytes
dequant calls in their training loop are effectively serialized one at a time.
vLLM is not: a single forward pass walks many layers with different weight
shapes (q_proj, k_proj, gate_proj, down_proj, ...), and a single global buffer
sized for the largest of them would still be *correct* under vLLM's default
single-CUDA-stream execution (see below), but it throws away the whole benefit
of GGUF's low VRAM footprint by making every dequant scratch allocation as
large as the biggest layer in the model. So this patch keys the cache by
(device, m, n, dtype) instead: one buffer per distinct weight shape actually
seen, grown lazily and only up to that shape's own size. In practice GGUF
models reuse only a handful of distinct (m, n) pairs (attention proj shapes,
MLP shapes), so the cache stays small and bounded regardless of layer count.

Stream-safety reasoning (why reuse across calls is sound here)
------------------------------------------------------------------
Buffer W's storage is filled by the Triton dequant kernel and then
immediately read by the caller's `x @ weight.T` GEMM, both issued on the same
CUDA stream, both inside the same Python call to run_dequantize_kernel(). The
next call that reuses the *same* (device, shape, dtype) key — say, a
different layer with an identically-shaped weight — enqueues its own dequant
kernel after that GEMM on that same stream. CUDA stream ordering guarantees
the GEMM reading the old contents has already been enqueued (and vLLM issues
these calls synchronously from a single Python thread, so it has also already
been *issued*) before the next dequant kernel touching the same memory is
enqueued; the stream will not reorder them. So a call sequence
dequant(W1)->gemm(W1)->dequant(W2, same shape, reuses W1's buffer)->gemm(W2)
cannot corrupt gemm(W1)'s read: gemm(W1) is already ahead of dequant(W2) in
the same stream's queue. This would NOT be safe if dequant and its GEMM could
be issued on independent streams that race with each other (e.g. CUDA-graph
multi-stream capture, or manual multi-stream inference) — this patch assumes
vLLM's default single-stream forward pass and does not attempt to detect or
guard against multi-stream execution. Anyone enabling CUDA graphs or a custom
multi-stream executor for the GGUF path should leave this patch off.

Env kill-switch
----------------
CUSTOM_VLLM_GGUF_DEQUANT_BUFFER=1   enable buffer reuse (default: off)

Left off by default so it's a clean A/B knob against the unpatched
torch.empty()-per-call baseline, same as the other opt-in dequant patches in
this repo. With it unset, run_dequantize_kernel() behaves exactly as upstream.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: unsloth-style dequant scratch-buffer reuse ---"

FLAG_ANCHOR = "def run_dequantize_kernel(\n"
FLAG_PATCH = (
    f"{PATCH_MARKER}\n"
    "import os as _custom_vllm_os\n\n"
    "_CUSTOM_VLLM_GGUF_DEQUANT_BUFFER = (\n"
    '    _custom_vllm_os.environ.get("CUSTOM_VLLM_GGUF_DEQUANT_BUFFER", "0") == "1"\n'
    ")\n\n"
    "_CUSTOM_VLLM_DEQUANT_BUFFERS: dict = {}\n\n\n"
    "def _custom_vllm_dequant_buffer(m, n, device, dtype):\n"
    "    # Grow-only scratch buffer keyed per (device, shape, dtype), mirroring\n"
    "    # unsloth's WEIGHT_BUFFER pattern (kernels/utils.py:611-632) but keyed\n"
    "    # instead of global, since vLLM interleaves many differently-shaped\n"
    "    # weights per forward pass. Safe under single-stream execution: see\n"
    "    # module docstring in patch_gguf_dequant_buffer.py.\n"
    "    total = m * n\n"
    "    key = (device, dtype, m, n)\n"
    "    buf = _CUSTOM_VLLM_DEQUANT_BUFFERS.get(key)\n"
    "    if buf is None:\n"
    "        buf = torch.empty(total, device=device, dtype=dtype)\n"
    "        _CUSTOM_VLLM_DEQUANT_BUFFERS[key] = buf\n"
    "    elif buf.numel() < total:\n"
    "        buf.resize_(total)\n"
    "    return buf[:total].view(m, n)\n\n\n"
    + FLAG_ANCHOR
)

BODY_ANCHOR = (
    "    W, total, dtype = validate_dequant_args(W, quant_type, m, n, dtype)\n"
    "    Y = torch.empty((m, n), device=W.device, dtype=dtype)\n"
)
BODY_PATCH = (
    "    W, total, dtype = validate_dequant_args(W, quant_type, m, n, dtype)\n"
    f"    {PATCH_MARKER}\n"
    "    if _CUSTOM_VLLM_GGUF_DEQUANT_BUFFER:\n"
    "        Y = _custom_vllm_dequant_buffer(m, n, W.device, dtype)\n"
    "    else:\n"
    "        Y = torch.empty((m, n), device=W.device, dtype=dtype)\n"
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/triton/dequantize/utils.py")
if not matches:
    raise SystemExit(
        f"vllm_gguf_plugin/triton/dequantize/utils.py not found under {site_packages}"
    )
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
else:
    for anchor in (FLAG_ANCHOR, BODY_ANCHOR):
        if anchor not in src:
            raise SystemExit(f"Anchor not found in {path}; plugin source may have changed")
    src = src.replace(FLAG_ANCHOR, FLAG_PATCH, 1).replace(BODY_ANCHOR, BODY_PATCH, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
