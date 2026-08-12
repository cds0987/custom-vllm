# [Question/Feature] FlashInfer backend hardcodes `use_cascade_attention() -> False` ("doesn't work, disable it for now") — cascade/shared-prefix attention unavailable whenever FlashInfer is selected

## Tóm tắt cho người không chuyên

Có một kỹ thuật tăng tốc gọi là "cascade attention", giúp các yêu cầu dùng
chung một đoạn văn bản mở đầu (ví dụ cùng một bộ hướng dẫn hệ thống) không
phải tính toán lại phần chung đó nhiều lần. vLLM có sẵn cơ chế bật/tắt kỹ
thuật này, nhưng với backend tăng tốc phổ biến nhất trên GPU đời mới
(FlashInfer), tính năng này bị khoá cứng về "tắt" ngay trong mã nguồn, kèm
một dòng ghi chú ngắn gọn "chưa hoạt động, tạm tắt" — không rõ tình trạng
sửa tới đâu. Với các hệ thống dùng chung tiền tố văn bản dài (ngữ cảnh
128K), đây là một khoản tăng tốc tiềm năng đang bị bỏ lỡ hoàn toàn mà không
có cách nào bật lại từ phía người dùng.

**Target repo:** vllm-project/vllm
**Type:** Question about status/roadmap, filed alongside a feature-availability report (not a bug in the sense of incorrect behavior — the hardcoded `False` is presumably intentional given the inline comment, but its scope, reasoning, and fix timeline are opaque to an outside user)
**Severity:** Medium (forecloses a real, measurable optimization for shared-long-prefix workloads on the majority-common attention backend; not a correctness issue)
**Affected versions:** vllm 0.26, `FlashInferBackend` (`vllm/v1/attention/backends/flashinfer.py`, `use_cascade_attention()`, observed around lines 1518-1524)
**Local fix:** None — not something a downstream user can patch around; filed as-is.
**Duplicate check:** No existing issue found specifically asking about FlashInfer's `use_cascade_attention()` hardcoded `False` and its fix status. Related but distinct from any generic "prefix caching" issues — prefix *caching* (KV block reuse across requests) is unaffected and works normally; this is specifically about *cascade attention* (a distinct optimization: a single batched attention computation that reads the shared prefix once for the whole batch, rather than each request in the batch separately re-reading the same KV blocks from the cache), which is a separate mechanism layered on top of prefix caching.

## Summary

We were validating a shared-system-prompt production scenario (many concurrent requests sharing a long, cached prefix — e.g. a common instruction/skills preamble in front of per-request content) at 128K total context on an L4 GPU, and found cascade attention structurally unavailable, behind two independent gates:

1. `ModelConfig.disable_cascade_attn` defaults to `True` (the `--no-disable-cascade-attn` CLI flag exists to opt in).
2. Even after opting in, `FlashInferBackend.use_cascade_attention()` unconditionally returns `False`, with an inline comment reading (as observed in the source): `"Cascade attention doesn't work, disable it for now"`. This is a hardcoded return, not conditional on any runtime check — opting in via the CLI flag has no effect when FlashInfer is the active attention backend.

FlashInfer is not an edge-case backend on this stack — it is the *only* backend our own duplicate-checked A/B testing (`STATUS.md`'s attention-backend investigation) found viable at all once fp8 KV cache is in use (`FLASH_ATTN` raises `ValueError: kv_cache_dtype not supported` for fp8 KV at startup in this build), and vLLM's own auto-selection only offers `['FLASHINFER', 'TRITON_ATTN']` as a result. So for any fp8-KV-cache deployment on this class of GPU, cascade attention is unconditionally unavailable, regardless of the CLI opt-in flag's existence.

## Why we're asking rather than just reporting a bug

The inline comment (`"disable it for now"`) reads as an intentional, temporary stopgap rather than an oversight, so we are not confident this is unintentional/unreported — but as outside users we have no visibility into what "doesn't work" refers to (correctness bug? perf regression? incomplete kernel support for some head configuration?), nor any tracking issue we could find describing the plan or timeline to re-enable it. We're filing this partly as a bug-shaped observation (the CLI flag `--no-disable-cascade-attn` silently has no effect for FlashInfer users, which is itself a minor documentation/UX gap even if the underlying disablement is deliberate) and partly as a direct question for maintainers: is there a tracking issue for FlashInfer cascade attention, and is `FlashAttention`'s cascade support (which we understand does work, per the codebase, but which we could not use because of the fp8-KV-cache incompatibility noted above) intended as the long-term answer for cascade-attention users, or is FlashInfer cascade support planned?

## Impact on our use case

At 128K total context with a long shared prefix (~120K tokens common across concurrent sessions, ~2-4K unique suffix per request), the lack of cascade attention means each concurrent request's attention computation re-reads the full shared-prefix KV blocks independently rather than the batch sharing one read — a real, currently unrecoverable cost at higher concurrency (we measured per-user decode throughput dropping meaningfully as concurrency increased while scanning the same 120K shared context, consistent with this). We worked around it at the application layer (front-loading the shared prefix once per session so prefix-cache hits keep TTFT low, rather than relying on cascade attention's batched-read optimization at the attention-kernel level), but a working cascade-attention path on FlashInfer would be a strict improvement for this and similar shared-prefix production patterns.

## What we're not claiming

We are not asserting cascade attention is easy to fix, nor that the hardcoded `False` is wrong — the maintainers clearly know about the limitation (it's commented, not silently absent). This report exists to (a) confirm from an external user's measurement that the gap has real, non-trivial cost for a legitimate production pattern (shared long prefix, many concurrent sessions), and (b) ask for visibility into status/plan, since we could not find one via search.
