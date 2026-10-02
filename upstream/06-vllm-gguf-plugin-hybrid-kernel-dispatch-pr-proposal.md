# [PR] Route large-batch GGUF matmuls through dequantize + dense GEMM (opt-in)

## Tóm tắt cho người không chuyên

Plugin GGUF nhân ma trận nén bằng kernel "fused" (đọc thẳng trọng số nén). Cách
đó hợp với lúc sinh từng chữ (decode, ít hàng), nhưng lúc đọc prompt dài
(prefill, hàng nghìn hàng) thì giải nén rồi nhân ma trận dày nhanh hơn nhiều
lần. Bản vá thêm MỘT biến môi trường, mặc định tắt: phép nhân nào có số hàng
từ ngưỡng trở lên thì đi đường giải nén, còn lại giữ nguyên.

**Trạng thái (2026-10-02):** viết lại trên `main` của plugin (`e2b8ad5`), đã chạy
test GPU (73/73) và đo A/B trên L4 với vLLM 0.30.0. CHƯA nộp — chờ user đọc từng
dòng bản vá và ra lệnh. Bản vá: `upstream/patches/06-dense-gemm-large-batch.patch`.
Số thô: `upstream/bench/results/2026-10-02-l4-qwen3-8b-q4km.md`. Commit nộp phải
có `Signed-off-by` của user.

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

NVIDIA L4, vLLM 0.30.0, plugin `main` @ `e2b8ad5` + this change built from source
(`_C_gguf` loaded), `unsloth/Qwen3-8B-GGUF:Q4_K_M`,
`--max-num-seqs 64 --max-num-batched-tokens 8192 --no-enable-prefix-caching`.
The only difference between the two columns is the environment variable.

Prefill, tok/s (one request at a time, fresh random prompt, `max_tokens=1`,
median of 5):

| prompt tokens | unset | `=1024` | speedup |
|---|---|---|---|
| ~2,100 | 213 | 2,709 | 12.7x |
| ~8,350 | 210 | 2,337 | 11.1x |
| ~12,200 | 208 | 2,715 | 13.1x |

A ~12.2k-token prompt goes from 58.7 s to 4.5 s.

Decode, total tok/s (closed loop, 256 tokens per request):

| concurrency | unset | `=1024` |
|---|---|---|
| 1 | 41.7 | 42.5 |
| 4 | 111.2 | 111.4 |
| 16 | 173.3 | 172.3 |
| 32 | 179.9 | 180.0 |

GPU memory after the run: 19,998 MiB vs 20,034 MiB. No OOM in either log.

Output check on the path this PR changes: a 5-digit code buried in the middle of
~3.1k and ~9.2k-token prompts is recalled 6/6 with the variable unset and 6/6
with it set. The continuation after the code is byte-identical in 4 of 6; the
other 2 differ in wording, which is expected when the prefill runs on a
different kernel.

Not measured: other GPUs, other quant types, MoE, mixed prefill+decode load.

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

- `pytest tests/test_kernels.py -k dense_gemm`: 73 passed (L4, CUDA extension loaded).
- `ruff check`, `ruff format`, `typos`: clean (ruff 0.14.0, as pinned).
- Benchmark and needle check: `VLLM_GGUF_DENSE_GEMM_MIN_ROWS` 0 vs 1024, numbers above.

## Duplicate check

Searched open PRs in this repo for dequant / cuBLAS / prefill / dispatch on
2026-10-02: none address dense-path routing for standard and K quants. #89 bounds
the dequantization workspace, #135 is MoE dispatch, #141 replaces the CUDA
kernels. vllm-project/vllm#55578 is the matching feature request and has no PR.

## AI assistance

This change was written with AI assistance (Claude). I have reviewed every
changed line and ran the tests and benchmarks above myself.
