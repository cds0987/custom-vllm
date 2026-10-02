# [PR] Route large-batch GGUF matmuls through dequantize + dense GEMM (opt-in)

## Tóm tắt cho người không chuyên

Plugin GGUF nhân ma trận nén bằng kernel "fused" (đọc thẳng trọng số nén). Cách
đó hợp với lúc sinh từng chữ (decode, ít hàng), nhưng lúc đọc prompt dài
(prefill, hàng nghìn hàng) thì giải nén rồi nhân ma trận dày nhanh hơn nhiều
lần. Bản vá thêm MỘT biến môi trường, mặc định tắt: phép nhân nào có số hàng
từ ngưỡng trở lên thì đi đường giải nén, còn lại giữ nguyên.

**Trạng thái (2026-10-02):** đã viết lại trên `main` của plugin (`e2b8ad5`),
CHƯA nộp. Bản vá: `upstream/patches/06-dense-gemm-large-batch.patch`. Nhánh
local: `out/vllm-gguf-plugin` @ `gguf-dense-gemm-large-batch` (chưa commit vì
upstream yêu cầu `Signed-off-by`). **Chặn nộp:** số đo bên dưới lấy trên plugin
0.0.4 — phải đo lại trên `main` (việc GPU, cần user duyệt) rồi mới điền vào.

**Target repo:** vllm-project/vllm-gguf-plugin
**Closes:** vllm-project/vllm#55578 (feature request "Add --gguf-dequant-on-load
option to unlock cuBLAS tensor cores for prefill-heavy workloads", open, không
ai nhận, không có PR — kiểm 2026-10-02)
**Liên quan, không trùng:** PR #89 (giới hạn workspace dequant cho weight cao),
PR #141/#142 (đồng bộ kernel CUDA với llama.cpp mới), PR #135 (dispatch MoE).

---

Everything below is the PR description, written for the plugin maintainers.

## Summary

Adds `VLLM_GGUF_DENSE_GEMM_MIN_ROWS` (default `0` = off). When set, a quantized
matmul whose input has at least that many rows is run as "dequantize the
weight, then `x @ W.T`" instead of through `mmq`. Smaller batches keep the
fused kernels. No behavior change unless the variable is set.

Closes vllm-project/vllm#55578.

## Why

`_fused_mul_mat_gguf` already has a dequantize + dense GEMM branch, but it is
only reachable for i-matrix types: standard and K quants are in
`MMQ_QUANT_TYPES`, so every batch above `mmvq_safe` goes to `mmq` regardless of
size. That is the right kernel for decode and the wrong one for prefill:

- Decode steps have at most `--max-num-seqs` rows and are bound by reading the
  packed weight. Fused kernels win.
- A prefill chunk has up to `--max-num-batched-tokens` rows and is compute
  bound. A dense GEMM wins by a wide margin.

#55578 asks for dequantizing at load time. That gets the prefill speed but
gives up the memory saving and the decode speed of the fused kernels. Choosing
per call by row count keeps both: the packed weight stays resident, and only
the layer being multiplied is expanded, transiently.

## Measurements

<!-- TODO before filing: re-measure on plugin main + current vLLM. -->

Measured on an L4 (sm89), Qwen3.5-2B Q4_K_M, plugin 0.0.4, vLLM 0.26/0.27.
Decode is tok/s at concurrency 32; prefill is sustained tok/s on 12k-token
prompts.

| path | decode @32 | prefill |
|---|---|---|
| fused (Triton) | 872 | ~3,500 |
| dequantize + dense, every call | 674 | ~8,580 |
| **this PR, threshold 1024** | **852** | **~8,858** |

With the CUDA extension built and loaded, the gap on prefill is larger: `mmq`
403 tok/s vs 10,496 tok/s through the dense path on the same prompts.

## Choosing the threshold

`x.shape[0]` is the number of tokens in the engine step, so decode and prefill
occupy separate ranges: at most `--max-num-seqs` rows for a pure decode step,
up to `--max-num-batched-tokens` for a prefill chunk. Any value between the two
works; we used 1024. The README section added here says so.

## Cost

The dense path allocates one dequantized weight per call, so peak transient
memory is the largest quantized layer in the model dtype. This is the same
allocation the existing i-matrix branch already makes; PR #89 bounds it for
tall weights and composes with this change.

## Changes

- `quantization/linear.py`: the env var, a `_dequantize_mul_mat_gguf` helper
  shared with the existing i-matrix branch, and one extra branch at the top of
  the dispatch.
- `tests/test_kernels.py`: `test_dense_gemm_dispatch` checks that batches at or
  above the threshold call only `ggml_dequantize`, that smaller ones never call
  it, and that both match the dense reference; plus a default-off check.
- `README.md`: a "Performance tuning" section.

## Test status

- `ruff check`, `ruff format`, `typos`: clean (ruff 0.14.0, as pinned).
- Routing logic checked on CPU with stubbed kernels (Q8_0, Q4_0; thresholds 0,
  1024, negative; rows 1 to 4096): 18/18.
- `tests/test_kernels.py` needs a GPU and has NOT been run yet.
