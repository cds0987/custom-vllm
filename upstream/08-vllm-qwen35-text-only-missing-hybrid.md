# [Bug] Text-only Qwen3_5ForCausalLMBase is missing IsHybrid + mamba-state classmethods, present only on the multimodal class

## Tóm tắt cho người không chuyên

Qwen3.5 là mô hình "lai" giữa hai kiểu ghi nhớ ngữ cảnh khác nhau, và phần
mềm quản lý bộ nhớ đặc biệt đó chỉ được khai báo cho phiên bản có hình ảnh,
không có cho phiên bản chỉ-văn-bản — dù cả hai dùng chung cơ chế lai này.
Hậu quả là ngay khi lỗi ở báo cáo trước được sửa, phiên bản chỉ-văn-bản vẫn
sập vì thiếu khai báo. Cách sửa là chuyển khai báo đó lên lớp cha dùng
chung.

**Target repo:** vllm-project/vllm
**Severity:** High (crash; only reachable after fixing report 07's registration gap)
**Affected versions:** vllm 0.26
**Local fix:** `scripts/patch_vllm_qwen35_hybrid.py`
**Duplicate check:** No existing issue found. This bug was unreachable until `Qwen3_5ForCausalLM` became selectable at all (see report 07) — before that fix, this code path was simply never exercised, which is presumably why it went unnoticed.

## Summary

Qwen3.5 interleaves gated-delta-net (SSM/Mamba-style) layers with full-attention layers — `Qwen3_5DecoderLayer` literally subclasses `Qwen3NextDecoderLayer`, and `Qwen3NextForCausalLM` declares `IsHybrid` for exactly this reason (it needs vllm's hybrid KV-cache/mamba-state machinery). But `Qwen3_5ForCausalLMBase` — the shared base both the text-only and multimodal classes derive from — only lists:

```python
class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
```

`IsHybrid`, and the three mamba-state classmethods it requires (`get_mamba_state_dtype_from_config`, `get_mamba_state_shape_from_config`, `get_mamba_state_copy_func`), were only ever put on `Qwen3_5ForConditionalGeneration` (the multimodal subclass), not on the base or on `Qwen3_5ForCausalLM` (the text-only subclass).

## Why this matters

`is_hybrid(model)` is just `getattr(model, "is_hybrid", False)`. Without `IsHybrid`, vllm never derives `cache_config.mamba_block_size` for this model, and KV-cache spec collection trips its own internal assertion:

```
File "vllm/model_executor/layers/mamba/abstract.py", line 47, in get_kv_cache_spec
    assert mamba_block_size is not None
AssertionError
```

## Reproduction

Requires report 07's registration fix to be applied first (otherwise `Qwen3_5ForCausalLM` is never selected and this code path is unreachable):

```bash
vllm serve unsloth/Qwen3.5-2B-GGUF:Q4_K_M \
  --hf-config-path Qwen/Qwen3.5-2B --tokenizer Qwen/Qwen3.5-2B
```

## The fix

Add `IsHybrid` to `Qwen3_5ForCausalLMBase`'s bases, and share the multimodal class's already-correct mamba-state classmethods (they read nothing but `hf_text_config`, so they're valid verbatim for the text-only class too):

```python
class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
):
    ...

# Qwen3_5ForConditionalGeneration is defined after the base class in the same
# module, so the shared classmethods can only be attached after both exist:
for _name in (
    "get_mamba_state_dtype_from_config",
    "get_mamba_state_shape_from_config",
    "get_mamba_state_copy_func",
):
    setattr(
        Qwen3_5ForCausalLMBase,
        _name,
        classmethod(getattr(Qwen3_5ForConditionalGeneration, _name).__func__),
    )
```

The upstream-native fix would be to define `IsHybrid` and the three classmethods once on `Qwen3_5ForCausalLMBase` directly, and have `Qwen3_5ForConditionalGeneration` inherit them rather than the other way around — we implemented it as a post-hoc attachment only because that's what a source patch against the installed package can do without restructuring class definition order.

## Verification

After both fixes (report 07 + this one), `mamba_block_size` is derived correctly and KV-cache spec collection succeeds; the model proceeds to weight loading (where reports 01–04 become relevant).
