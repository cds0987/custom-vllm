# [Bug] gguf-py's tensor_mapping.py has no template for Qwen3.5's linear_attn.dt_bias (only qwen3next's dt_proj is covered)

## Tóm tắt cho người không chuyên

Thư viện xử lý tên tensor của định dạng GGUF có một "từ điển" các tên tham số
đã biết cho từng kiến trúc mô hình, dùng để đối chiếu qua lại giữa tên HuggingFace
và tên GGUF. Kiến trúc Qwen3.5 dùng một cái tên tham số mà từ điển này chưa
từng được dạy, nên các công cụ dựa vào nó báo "không tìm thấy tham số" dù
tham số đó thực sự tồn tại trong file. Cách sửa là thêm đúng một dòng tên
mới vào từ điển.

**Target repo:** ggml-org/llama.cpp (gguf-py, the `gguf` PyPI package's source)
**Severity:** Medium (blocks GGUF weight-name mapping for one tensor family per layer; the symptom is a generic "unmapped parameter" error, not a crash inside gguf-py itself)
**Affected versions:** gguf 0.19.0
**Local fix:** `scripts/patch_gguf_tensor_mapping.py`
**Duplicate check:** No existing issue found in ggml-org/llama.cpp for this specific tensor-name template gap.

## Summary

The `gguf` package ships a fixed table of known HF tensor-name conventions per architecture in `tensor_mapping.py`, used by downstream consumers (llama.cpp itself, and Python-side tools like vllm-gguf-plugin) to translate between HF parameter names and GGUF tensor names. Other hybrid-attention architectures already in the table — qwen3next, mamba, jamba — name their SSM decay-gate bias tensor `...dt_proj` (a `Linear` layer, with separate `.weight` and `.bias` sub-tensors). Qwen3.5 instead names it `...linear_attn.dt_bias` directly — a standalone bias parameter, not a `Linear` layer's bias — and no template exists for this name shape at all.

## Why this surfaces as a mapping failure, not directly in gguf-py

`gguf-py` itself doesn't error — the gap only becomes visible once a downstream consumer's suffix-stripping normalizes `dt_bias` the same way it already normalizes other trailing-`_weight`/`_bias` names without a dot (see the companion vllm-gguf-plugin report, item 5, for the matching `_bias` stripping fix). Once the base name is normalized from `linear_attn.dt_bias` down to `linear_attn.dt`, that name still isn't recognized by any existing `tensor_mapping.py` template — so the consumer's own "did every HF parameter get mapped" check fails:

```
Failed to map GGUF parameters: ['model.layers.{N}.linear_attn.dt_bias', ...]  (for every layer)
```

## The fix

Add the missing template as a sibling entry to the existing qwen3next `dt_proj` template, in the same tensor-name-family block of `tensor_mapping.py`:

```python
MODEL_TENSOR.SSM_DT: (
    ...
    "model.layers.{bid}.linear_attn.dt_proj",   # qwen3next
    "model.layers.{bid}.linear_attn.dt",        # qwen3.5  <-- added
    ...
),
```

(The exact key/block name may differ slightly by `gguf-py` version; our patch targets the qwen3next `dt_proj` line as its anchor and inserts the qwen3.5 form immediately after it, in the same list.)

## Verification

After this template is added (together with the plugin-side `_bias` suffix-stripping fix it depends on), `linear_attn.dt_bias` resolves through the normal GGUF name-mapping path for every gated-delta-net layer, and no longer needs to be treated as an unmapped/missing parameter.
