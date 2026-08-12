# [Bug] Qwen3_5ForCausalLM / Qwen3_5MoeForCausalLM classes exist but are never registered — architecture silently rewritten to the multimodal class, crashing with an empty vision tower

## Tóm tắt cho người không chuyên

Phần mềm có sẵn hai "công thức" đúng để chạy phiên bản chỉ-văn-bản của
Qwen3.5, nhưng quên đăng ký chúng vào danh sách công thức được phép dùng.
Kết quả là hệ thống tự động chọn nhầm sang công thức dành cho phiên bản có
xử lý hình ảnh, rồi sập với thông báo lỗi gây hiểu lầm (nhắc tới "tháp thị
giác trống rỗng") khiến người dùng khó đoán ra nguyên nhân thật. Cách sửa là
thêm hai dòng đăng ký còn thiếu.

**Target repo:** vllm-project/vllm
**Severity:** Critical (crash; and the crash traceback is misleading about the actual cause)
**Affected versions:** vllm 0.26
**Local fix:** `scripts/patch_vllm_qwen35_registry.py`
**Duplicate check:** No existing issue found specifically about the text-only Qwen3.5 registry gap. Related but distinct: vllm-project/vllm#38122/PR #38140 and #36456 are both about loading Qwen3.5 GGUF at all (naming/config-path bugs upstream of model instantiation); this bug is downstream of all of those — it fires once loading otherwise succeeds and vllm tries to pick which model class to instantiate.

## Summary

`vllm/model_executor/models/qwen3_5.py` defines both text-only classes:

```python
class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):        # ~line 376
class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, ...):  # ~line 380
```

but `vllm/model_executor/models/registry.py` only lists the multimodal variants:

```python
"Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
"Qwen3_5MoeForConditionalGeneration": ("qwen3_5", "Qwen3_5MoeForConditionalGeneration"),
```

A checkpoint whose config declares `architectures=["Qwen3_5ForCausalLM"]` — which is exactly what a GGUF resolves to, since vllm-gguf-plugin's `GGUFConfigParser` sets `architectures` from `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` — therefore does not match any registered architecture.

## The silent part

Instead of failing loudly with "unknown architecture," `_ModelRegistry._normalize_arch()` falls through to `try_match_architecture_defaults()`, which rewrites the class-name *suffix*:

```
"Qwen3_5ForCausalLM"
  -> suffix "ForCausalLM" is not itself a registered architecture
  -> fuzzy-matches to "Qwen3_5ForConditionalGeneration"  (registered!)
```

Both suffixes share the same `("generate", "none")` runner/convert defaults, so vllm considers this normalization a legitimate match and silently instantiates the **multimodal** model class for what is architecturally a text-only checkpoint.

## Reproduction and exact error text

```bash
vllm serve unsloth/Qwen3.5-2B-GGUF:Q4_K_M \
  --hf-config-path Qwen/Qwen3.5-2B --tokenizer Qwen/Qwen3.5-2B
```

The GGUF (as-is, no separate `mmproj-*.gguf`) carries no vision weights, so the vision tower is instantiated but built with empty/zero-element quantized weights. The startup profile run then pushes a dummy image through it:

```
File "vllm/v1/worker/gpu_model_runner.py", in profile_run
    dummy_encoder_outputs = self.model.embed_multimodal(...)
File "vllm/model_executor/models/qwen3_vl.py", in _process_image_input
    image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
File "vllm_gguf_plugin/quantization/linear.py", in _fused_mul_mat_gguf
    return x @ qweight.T
RuntimeError: size mismatch, got input (65536), mat (65536x1024), vec (0)
```

The `vec (0)` is the tell — the vision projection weight matrix has zero elements, because there was never any vision data in the checkpoint to load into it. The traceback points at a matmul shape mismatch deep in the GGUF quantization kernel; the actual root cause (wrong model class selected three layers up, in the registry) is nowhere visible in the stack trace, which makes this bug unusually hard to diagnose from the error alone.

## The fix

Register the two existing text-only classes in `registry.py`, next to the existing `Qwen3NextForCausalLM` entry (Qwen3.5's decoder layer literally subclasses Qwen3Next's):

```python
"Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),
"Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),
```

Once registered, `"Qwen3_5ForCausalLM"` resolves directly and the suffix-rewrite path in `_normalize_arch()` is never reached. Text-only Qwen3.5 checkpoints (GGUF and otherwise) load the model class they actually are.

## Related fixes required in the same code path

Registering the class is necessary but not sufficient — once `Qwen3_5ForCausalLM` becomes reachable, it exposes two more gaps that were previously unreachable dead code (the multimodal class had already worked around them): missing `IsHybrid`/mamba-state support (report 08) and a missing `quant_config` on `embed_tokens` (report 09).
