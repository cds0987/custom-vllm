# Bản nháp bình luận cho vllm-gguf-plugin PR #141 — CHƯA GỬI, chờ user duyệt

Thay cho việc nộp PR riêng (xem `06-...md` và `bench/results/`). Nội dung dưới
đây là tiếng Anh, viết cho tác giả và maintainer của PR #141.

---

Independent data point for this PR, in case it helps review. NVIDIA L4 (sm89),
vLLM 0.30.0, torch 2.13.0+cu132, `unsloth/Qwen3-8B-GGUF:Q4_K_M`,
`--max-num-seqs 64 --max-num-batched-tokens 8192 --no-enable-prefix-caching`.
`main` is `e2b8ad5`, this PR is `aa09d65`, both built from source.

| | main | this PR |
|---|---|---|
| prefill, ~2.1k-token prompt (tok/s) | 213 | 2,440 |
| prefill, ~8.3k-token prompt | 210 | 2,095 |
| prefill, ~12.2k-token prompt | 208 | 1,998 |
| decode, concurrency 1 (tok/s) | 41.7 | 47.6 |
| decode, concurrency 4 | 111.2 | 123.1 |
| decode, concurrency 16 | 173.3 | 516.0 |
| decode, concurrency 32 | 179.9 | 918.6 |

Prefill is one request at a time with a fresh random prompt and `max_tokens=1`
(median of 5); decode is closed-loop with 256 fixed-length tokens per request.
No errors or degenerate outputs in either arm. This also largely addresses
vllm-project/vllm#55578.

One observation for a possible follow-up: on this GPU, sending large batches to
the `DENSE_BLAS` route this PR already ships is still somewhat faster than MMQ
for prefill. A local experiment that calls `ops.ggml_dense_blas` when
`x.shape[0] >= 1024` gave 2,526 / 2,537 / 2,286 tok/s on the three prompt
lengths above (+4% / +21% / +14%), with decode unchanged. Only measured on L4
with Q4_K_M, so I don't know how it holds on other architectures or quant types.

Bench script: https://github.com/cds0987/custom-vllm/tree/master/upstream/bench

(The measurements were run with AI assistance; I checked the method and the numbers.)
