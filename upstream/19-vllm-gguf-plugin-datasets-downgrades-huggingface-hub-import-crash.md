# [Bug] Installing `datasets` downgrades `huggingface_hub` below the version `vllm-gguf-plugin` needs — breaks serving of ANY model, not just GGUF, from import time

## Tóm tắt cho người không chuyên

Cài thêm một thư viện phụ thường dùng cho việc chuẩn bị dữ liệu huấn luyện
(`datasets`) vô tình kéo theo một phiên bản CŨ của một thư viện khác
(`huggingface_hub`) mà gói GGUF của vLLM cần bản MỚI hơn. Hậu quả nghiêm
trọng hơn vẻ ngoài: máy chủ bị lỗi ngay khi khởi động — không chỉ với mô
hình GGUF, mà với BẤT KỲ mô hình nào — vì lỗi xảy ra ngay lúc nạp gói phần
mềm (import), trước khi phần mềm kịp biết người dùng có định dùng GGUF hay
không. Cách khắc phục là cài lại phiên bản mới của thư viện bị hạ cấp sau
khi cài xong `datasets`.

**Target repo:** vllm-project/vllm-gguf-plugin
**Severity:** High (total server startup failure, for ANY model — not scoped to GGUF usage — triggered by an unrelated, common package install; the failure mode gives no indication that `datasets` is the cause)
**Affected versions:** vllm-gguf-plugin 0.0.4 (as installed alongside `vllm` 0.26); observed `huggingface_hub` downgraded to 1.23.0 by a plain `pip install datasets`
**Local fix:** `scripts/setup_env.sh` force-reinstalls `huggingface_hub` (`pip install -q -U huggingface_hub`) immediately after installing `datasets`, unconditionally, as a standing guard.
**Duplicate check:** No existing issue found in vllm-project/vllm-gguf-plugin about this specific dependency-resolution interaction with `datasets`.

## Summary

`vllm_gguf_plugin`'s loader (`loader.py`) does, at import time:

```python
from huggingface_hub import ResolvedRevision, ...
```

`ResolvedRevision` is a symbol that does not exist in older `huggingface_hub` releases — we observed it missing from `huggingface_hub==1.23.0`. Separately, `pip install datasets` (a common, unremarkable dependency for anyone also doing quantization/calibration work alongside serving — `llmcompressor`, evaluation scripts using `load_dataset`, etc., in the same environment) resolves and installs `huggingface_hub==1.23.0` as one of its own dependency constraints, silently downgrading whatever newer `huggingface_hub` version `vllm`/`vllm_gguf_plugin` had previously installed and required.

The result: any subsequent `vllm serve` invocation — for **any** model, not just a GGUF one — crashes at plugin-import time with an `ImportError` for `ResolvedRevision`, before vLLM has even reached the point of deciding whether the requested model needs GGUF-specific handling at all.

## Reproduction

```bash
pip install vllm vllm-gguf-plugin      # huggingface_hub resolves to a recent version here
pip install datasets                   # silently downgrades huggingface_hub to 1.23.0
vllm serve Qwen/Qwen3.5-2B             # NOT a GGUF model -- still crashes
```

```
ImportError: cannot import name 'ResolvedRevision' from 'huggingface_hub'
```

## Why this is worse than an ordinary dependency conflict

- It is not scoped to GGUF usage. A user who installs `datasets` for a completely unrelated reason (e.g. following an evaluation/benchmarking tutorial, or preparing calibration data for an unrelated quantization workflow) and then tries to serve any plain safetensors checkpoint hits this crash, with no obvious connection between "I installed `datasets` yesterday" and "my server won't start today."
- The failure is at Python **import** time inside `vllm_gguf_plugin`, which vLLM apparently loads eagerly as an installed plugin regardless of whether the served model is GGUF — so simply not using GGUF does not avoid the crash once the plugin package is installed at all.
- `pip`'s own dependency resolver does not flag this as a conflict at `datasets` install time (both `huggingface_hub==1.23.0` and whatever newer version `vllm-gguf-plugin` needs are presumably within some declared-compatible range from pip's point of view, or the ordering of installs simply lets the later, narrower `datasets` constraint win) — so there is no resolver warning pointing at the problem either.

## The fix we apply locally

`scripts/setup_env.sh` installs `datasets`/`llmcompressor` (needed for several of our own tooling scripts, not for serving itself) and then unconditionally force-reinstalls a current `huggingface_hub` immediately afterward:

```bash
pip install -q llmcompressor datasets || echo "..."
# datasets pins an older huggingface_hub ... silently overwrites the
# newer huggingface_hub vllm/vllm_gguf_plugin need
pip install -q -U huggingface_hub || echo "WARNING: huggingface_hub re-pin failed"
```

This is a workaround, not a fix — it depends on us remembering to always re-pin after installing `datasets`, in that order, every time.

## What we'd ask upstream to consider

Either (a) vllm-gguf-plugin declares an explicit, resolver-visible lower-bound version constraint on `huggingface_hub` (e.g. `huggingface_hub>=X.Y` in its own package metadata) tight enough that pip's resolver would refuse to let `datasets` silently downgrade it below the version `ResolvedRevision` requires, rather than the current apparent absence of such a constraint; or (b) the plugin's import of `ResolvedRevision` (and any other version-sensitive `huggingface_hub` symbol) is deferred/guarded so an incompatible `huggingface_hub` produces a clear, actionable error message ("vllm-gguf-plugin requires huggingface_hub>=X, found Y — this is commonly caused by installing `datasets` after `vllm-gguf-plugin`") rather than a bare `ImportError` with no attribution to the actual cause.
