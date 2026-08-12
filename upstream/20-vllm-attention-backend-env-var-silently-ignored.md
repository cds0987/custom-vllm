# [Bug/UX] `VLLM_ATTENTION_BACKEND` environment variable is silently ignored in vllm 0.26 — logs "Unknown ... variable" instead of erroring, giving no indication the override had zero effect

## Tóm tắt cho người không chuyên

Một biến môi trường từng dùng để ép vLLM chọn một cơ chế tính attention cụ
thể đã bị bỏ (thay bằng một cờ dòng lệnh mới), nhưng nếu người dùng vẫn đặt
biến môi trường cũ theo thói quen hoặc theo tài liệu cũ, hệ thống chỉ in một
dòng log mơ hồ ("biến không xác định") rồi lặng lẽ dùng backend mặc định —
không phải backend người dùng tưởng mình đã chọn. Nhóm nghiên cứu đã tự mắc
bẫy này: hai lượt so sánh "A/B" tưởng là hai backend khác nhau hoá ra chạy
cùng một backend suốt, khiến kết luận đo đạc ban đầu bị sai cho tới khi phát
hiện ra.

**Target repo:** vllm-project/vllm
**Severity:** Medium (no crash, no data corruption — but a silent no-op on a variable that clearly looks like a functioning configuration override, which actively produces wrong conclusions in exactly the kind of A/B backend comparison it would be used for)
**Affected versions:** vllm 0.26
**Local fix:** None needed once discovered — switched to the replacement CLI flag (`--attention-backend`, per `vllm serve --help=AttentionConfig`). Filed as a UX/logging gap, not something we patch around.
**Duplicate check:** No existing issue found specifically about `VLLM_ATTENTION_BACKEND` being silently ignored (vs. erroring or warning loudly) in favor of the newer `--attention-backend` CLI flag.

## Summary

In vllm 0.26, setting the `VLLM_ATTENTION_BACKEND` environment variable — a long-standing, widely-documented-in-the-ecosystem way to force a specific attention backend — has no effect. The backend selection machinery has moved to a CLI flag (`--attention-backend`), and the environment variable path now only produces a log line along the lines of `Unknown ... variable` and proceeds with auto-selected (or otherwise default) backend behavior, rather than either (a) still honoring the old variable for backward compatibility, or (b) raising a clear, unambiguous error/warning that the variable is deprecated/removed and no longer has any effect.

## How we discovered this (and why it's dangerous)

We were running a deliberate A/B comparison between two attention backends on the same hardware/model/workload, setting `VLLM_ATTENTION_BACKEND=FLASH_ATTN` for one run and leaving it unset (or set to a different value) for the other, expecting genuinely different backend behavior between the two runs. Both runs in fact used the *same* backend throughout (auto-selected, since the env var override never took effect) — meaning our first two "A/B" measurement passes were comparing a configuration against itself, and any conclusion drawn from that comparison was invalid. We only caught this by noticing the log line and separately confirming, via `vllm serve --help=AttentionConfig`, that the actual override mechanism had moved to a CLI flag.

This is exactly the failure mode that is hardest to catch: the tool does not error, does not crash, and produces plausible-looking (if uniformly wrong) results either way — there is no signal from the output alone that anything is misconfigured. Anyone following slightly outdated documentation, a cached shell profile, an old benchmarking script, or general ecosystem knowledge (the env var name is used the same way across several other inference engines) who sets `VLLM_ATTENTION_BACKEND` expecting it to work will silently get vLLM's default/auto-selected backend instead, with no indication their override was ignored unless they are specifically watching startup logs for an "Unknown ... variable"-shaped message and know to connect that to this specific variable.

## Suggested fix

At minimum, promote the "Unknown ... variable" log line to a `logger.warning` (if it isn't already) that explicitly names the variable, states that it has no effect, and points at the replacement (`--attention-backend`) — e.g. `"VLLM_ATTENTION_BACKEND is set but no longer has any effect in this version; use --attention-backend instead."` A generic "Unknown ... variable" message (if that is an accurate paraphrase of the actual text — we did not capture the exact string verbatim during our own run and would recommend re-confirming the precise wording before filing) does not make clear to a user (a) that this specific, commonly-used variable used to work and has been intentionally superseded, versus e.g. a typo'd/unsupported variable name being silently ignored as a matter of general policy, or (b) what the correct replacement mechanism is. If backward compatibility is feasible, honoring the old variable (with a deprecation warning) for at least one release cycle would avoid silently breaking existing scripts/tooling entirely.

## Caveat

We did not capture the exact verbatim log text or the precise vllm sub-version at the moment this was observed (this surfaced as a side-finding during broader attention-backend investigation, not as a dedicated repro exercise) — before filing, we'd recommend re-running `VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve <any model>` against current vllm `main` to capture the exact message text and confirm the behavior still reproduces as described.
