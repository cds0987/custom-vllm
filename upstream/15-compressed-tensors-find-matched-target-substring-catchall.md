# [Bug] `find_matched_target` matches a catch-all `targets: ["Linear"]` group via SUBSTRING match on the module's class name before fused-component reconciliation ever runs — silently drops per-projection config for fused layers

## Tóm tắt cho người không chuyên

Khi một checkpoint đã nén khai báo "áp dụng cách nén X cho riêng lớp mạng
gộp Y", hệ thống lại kiểm tra sai thứ tự: nó tìm xem TÊN LỚP có "chứa" một
từ khoá chung chung ("Linear") hay không TRƯỚC khi kiểm tra tên cụ thể của
lớp gộp đó — mà gần như lớp nào cũng có chữ "Linear" trong tên (ví dụ
"MergedColumnParallelLinear"). Vì vậy nhóm cấu hình dành riêng cho lớp gộp
luôn bị bỏ qua âm thầm, hệ thống áp nhầm cách nén mặc định, và chỉ phát hiện
ra rất muộn — khi hai nửa của lớp gộp bị nén theo hai kiểu khác nhau và phép
ghép nổ lỗi khó hiểu. Cách phòng tránh (đã kiểm chứng) là luôn đặt tên lớp
gộp CHÍNH XÁC vào danh sách "targets" thay vì tên các phần chưa gộp.

**Target repo:** vllm-project/compressed-tensors (the `find_matched_target`/module-target-matching logic consumed by vLLM's `CompressedTensorsConfig`; also reachable via vLLM's `vllm/model_executor/layers/quantization/compressed_tensors/` integration layer, which is the vantage point this was diagnosed from)
**Severity:** High (silent misconfiguration — a `config_groups` entry intended for a fused layer is silently ignored in favor of a catch-all default, surfacing much later as an unrelated-looking merge-mismatch crash, not as a config-validation error at load time)
**Affected versions:** compressed-tensors as vendored/pinned by vllm 0.26 (exact standalone `compressed-tensors` PyPI version not independently pinned in this environment — we accessed it only via vLLM's dependency, not as a standalone install; please cross-check against the current `main` before triage)
**Local fix:** Not a code patch (this is a matching-order bug in a third-party dependency, not something we patch in-place) — workaround is entirely in how `config_groups[*].targets` is authored: use the FUSED vLLM parameter name (e.g. `"in_proj_qkvz"`, `"in_proj_ba"`), never the individual unfused HF projection names. See `scripts/graft_gguf_gdn.py`'s `FUSED_NAME_FOR_HF_SUFFIX` table and `build_new_config_group`.
**Duplicate check:** No existing issue found describing this exact matching-order interaction (substring-vs-class-name catch-all winning ahead of fused-component reconciliation). We were unable to independently browse the compressed-tensors issue tracker from this environment (no network access) to do a live duplicate check the way earlier drafts in this set did against vllm/vllm-gguf-plugin — **this draft's duplicate-check step is therefore incomplete and should be redone live before filing.**

## Summary

compressed-tensors' `find_matched_target` (consumed by vLLM when resolving which `config_groups` entry — i.e. which quantization scheme — applies to a given `nn.Module` during weight loading) resolves a match in three steps, in this order:

1. Exact/regex match of the fully-qualified layer name against each candidate group's `targets`.
2. A **substring** match of the module's Python class name (e.g. `"MergedColumnParallelLinear"`) against `targets` — i.e. `"Linear" in "MergedColumnParallelLinear"` is `True`.
3. Fused-vs-unfused-component reconciliation — the logic that would otherwise resolve a `config_groups` entry written against individual unfused projection names (e.g. `"in_proj_b"`, `"in_proj_a"`) against the actual fused layer instance vLLM builds (`in_proj_ba`).

Because step 2 runs before step 3, and step 2 matches **any** Linear-family module against a bare `"Linear"` target via substring containment, a checkpoint's default catch-all `config_group` (`targets: ["Linear"]` — a common, reasonable-looking way to express "apply this scheme to every linear layer") always wins step 2 before step 3 — the step that holds the fused-component logic — ever gets a chance to run for that layer. Any `config_groups` entry authored against the fused layer's *unfused component names* is therefore silently never applied to fused layers, regardless of how specific or intentional it was.

## Reproduction and exact failure signature

Confirmed empirically while grafting a second, independently-quantized scheme onto Qwen3.5's GDN input projections on top of an existing champion checkpoint (task name in our internal tracking: TASK M-exec): authoring the new `config_groups` entry with unfused per-projection `targets` (`"in_proj_b"`, `"in_proj_a"`) produced a checkpoint where vLLM's Marlin loader tried to merge an int8-packed shard into a parameter sized for the champion checkpoint's *unrelated, pre-existing* int4 group128 default scheme:

```
RuntimeError: cannot merge quantized shard 1 ... different num_bits/group_size
```

This is a confusing failure at merge time, with no indication that the actual root cause was a matching-order issue upstream in `find_matched_target`/`_match_fused_layer` — it looks like a data/shape bug, not a config-resolution bug. Root-caused by instrumenting `find_matched_target`/`_match_fused_layer` directly (stepping through which `config_groups` entry each call actually resolved to), not guessed from behavior alone.

## Why this matters beyond our specific case

Any workflow that authors a `config_groups` entry targeting a **fused** vLLM layer (any `MergedColumnParallelLinear`/`QKVParallelLinear` — common across most current dense and MoE architectures, not just Qwen3.5's GDN) by naming its **unfused, pre-merge** HF-side sub-projections, in the presence of a checkpoint that also has any catch-all-ish `targets` entry whose value is a substring of the fused module's class name (`"Linear"` being the obvious, extremely common case), silently loses that config entry. There is no error at config-parse time and no error at model-build time — only a much later, differently-shaped crash (or, worse, no crash at all if the mismatched scheme happens to be shape-compatible, in which case the wrong scheme is silently applied).

## The workaround (verified) and the proposed fix

**Workaround (what we now do):** always author `config_groups[*].targets` for a fused layer using the layer's actual FUSED vLLM parameter name (`"in_proj_qkvz"`, `"in_proj_ba"` for Qwen3.5's GDN; the equivalent fused name for any other architecture's `MergedColumnParallelLinear`/`QKVParallelLinear`), never the unfused component names — this reaches an exact match at step 1, before step 2's substring catch-all can intervene.

**Proposed fix (for compressed-tensors maintainers to evaluate):** either (a) run fused-component reconciliation (current step 3) before the class-name substring catch-all (current step 2), so a specific-but-unfused target is preferred over a generic catch-all when both could apply; or (b) restrict the class-name substring match to require the candidate target to be a *whole path segment* or a registered alias rather than an arbitrary substring, so `"Linear"` does not incidentally match every `*Linear` subclass; or, at minimum, (c) emit a warning when a `config_groups` entry's `targets` never produces an exact/regex match for any module actually present in the model, so an unreachable config entry (as opposed to a merely-overridden one) is at least visible.

## Caveat on this draft's completeness

Unlike the other reports in this set, this one was written from static reasoning over vLLM's own compressed-tensors integration code plus one directly-observed, root-caused failure — we did not have network access in this environment to browse the standalone `compressed-tensors` GitHub repository, confirm its exact file/line numbers on current `main`, or run a live duplicate-check against its issue tracker the way the other drafts in this collection do. Before filing, re-verify the exact file (`compressed_tensors/utils/match.py` or equivalent) and line numbers against current `main`, and run the same duplicate-check pass the other drafts received.
