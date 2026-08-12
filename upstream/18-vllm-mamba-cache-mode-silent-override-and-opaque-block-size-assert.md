# [Bug/UX] Two silent/opaque configuration failure modes for hybrid (Mamba/GDN) models: `--mamba-cache-mode align` is silently downgraded to `none` when prefix caching is off, and the `block_size <= max_num_batched_tokens` assert gives no actionable message

## Tóm tắt cho người không chuyên

Hai vấn đề nhỏ nhưng gây mất thời gian khi cấu hình mô hình "lai" (có phần
nhớ kiểu Mamba): (1) người dùng đặt một chế độ quản lý bộ nhớ cụ thể
("align"), nhưng nếu quên bật kèm một cờ khác (bật cache tiền tố), hệ thống
âm thầm đổi lại về chế độ mặc định "không có gì" — chỉ có một dòng log dễ bị
bỏ qua báo việc này, không phải lỗi hay cảnh báo rõ ràng. (2) một điều kiện
bắt buộc khác (kích thước khối bộ nhớ phải nhỏ hơn một con số cấu hình khác)
báo lỗi ngay lúc khởi động nhưng không giải thích tại sao hay phải sửa cờ
nào — người dùng phải tự đọc mã nguồn để hiểu. Cả hai đều dễ sửa bằng cách
làm rõ thông báo.

**Target repo:** vllm-project/vllm
**Severity:** Low-Medium (no incorrect behavior once the flags are set correctly — both are pure UX/diagnostics gaps that cost real debugging time, not correctness bugs)
**Affected versions:** vllm 0.26
**Local fix:** None (both are logging/error-message clarity gaps, not something we patch around locally — we simply learned the two rules the hard way and now pass the flags correctly)
**Duplicate check:** No existing issue found for either specific gap.

## Issue 1 — `--mamba-cache-mode align` silently reverts to `none` when prefix caching is disabled

While validating a shared-prefix / prefix-caching production scenario for a hybrid Mamba/attention model (Qwen3.5) at 32K-65K context, we found that in this build, prefix caching is **not** enabled by default (contrary to what we had assumed from earlier, unrelated testing on a different config) — and, critically, if `--enable-prefix-caching` is left off while `--mamba-cache-mode align` is explicitly passed, vLLM does not raise an error and does not silently keep the requested mode. Instead it emits a log line:

```
Mamba cache mode is set to 'none' when prefix caching is disabled
```

and proceeds with `mamba_cache_mode=none` regardless of what the user asked for. This is a downgrade of an explicit user setting to a value the user did not request, communicated only via an INFO-level (or similar) log line easy to miss in a busy startup log, rather than:
- a startup error refusing to proceed with an inconsistent flag combination, or
- a WARNING-level message making clear that the user's explicit `--mamba-cache-mode align` choice is being **overridden**, not merely defaulted.

We only caught this because we were specifically grepping startup logs for confirmation the flag took effect (a rebuild reflex from an earlier debugging session) — a normal user relying on the CLI flag they passed would have no reason to suspect it was silently ignored, and would only notice via a much later, harder-to-diagnose symptom (e.g. unexpectedly poor prefix-cache-hit behavior for mamba-state-bearing sessions).

**Suggested fix:** promote this to a `logger.warning(...)` (not just info-level), and/or have `--mamba-cache-mode align` with `--enable-prefix-caching` unset raise a clear `ValueError` at argument-validation time (`vllm serve --mamba-cache-mode align` without `--enable-prefix-caching` is presumably never a combination the user actually wants, since `align` mode's entire purpose is to synchronize mamba-state eviction with prefix-cache block eviction) rather than silently substituting a different value and continuing.

## Issue 2 — `block_size <= max_num_batched_tokens` assert gives no actionable message

Separately, while tuning `--max-num-batched-tokens` down for a latency-sensitive shared-prefix workload (chasing a lower `ITL p95` by shrinking the prefill chunk budget), we hit a hard `AssertionError` at server startup for `--max-num-batched-tokens` values of 1024 and 512:

```
AssertionError
```

(no accompanying message text distinguishing this assert from any other in the startup path). Root-caused, by reading the scheduler/mamba-cache config source directly (not from any error text), to an assertion that mamba's own cache block size must be `<=` `max_num_batched_tokens` when `mamba_cache_mode=align` — and the specific model's mamba block size (Qwen3.5, this checkpoint) is **1056**, meaning any `--max-num-batched-tokens` value below 1056 (1024 and 512 both fail this way) trips the assert. `--long-prefill-token-threshold`, if set, is subject to the same floor.

This is an entirely legitimate constraint — we're not disputing that the assert should exist — but as filed it gives a user no way to know: (a) which constraint failed, (b) what the model's actual mamba block size is, or (c) which of the several related CLI flags (`--max-num-batched-tokens`, `--long-prefill-token-threshold`) needs to change to satisfy it. We only resolved this by reading vLLM's own scheduler/config source, which is a much higher bar than a CLI user should have to clear for what is, in the end, a one-line numeric constraint.

**Suggested fix:** give this assert (and its sibling for `--long-prefill-token-threshold`) an explicit message, e.g. `f"max_num_batched_tokens ({value}) must be >= mamba cache block_size ({block_size}) when mamba_cache_mode=align"`, so the fix is discoverable from the error text alone.

## Why we're filing these together

Both surfaced during the same investigation (tuning scheduler flags for a hybrid-model shared-prefix production scenario) and share a common theme: hybrid-model-specific configuration constraints exist and are enforced correctly, but neither failure mode communicates *what* was enforced or *why* to the user experiencing it — one fails by silently substituting a different value, the other fails by raising with no message at all. Both cost meaningful debugging time that a one-line message improvement would have eliminated.
