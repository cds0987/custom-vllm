# [Feature] `ParallelLMHead`/`VocabParallelEmbedding` have no compressed-tensors quantization path — the output head can never be compressed

## Tóm tắt cho người không chuyên

Lớp cuối cùng của mô hình (bảng "đoán từ tiếp theo", thường rất lớn vì có
một hàng cho mỗi từ trong từ điển) không thể được nén nhỏ lại như các lớp
khác, dù nhóm đã thử và tính toán ra số liệu nén hợp lệ về mặt toán học.
Nguyên nhân là lớp này thuộc một "họ" lớp mạng khác (Embedding) mà cơ chế
nén hiện tại của vLLM chỉ hỗ trợ cho họ "Linear". Đây không phải lỗi mà là
một tính năng còn thiếu — nhóm đề xuất bổ sung, và ước tính chi phí không hề
nhỏ: bảng này chiếm khoảng 2GB bộ nhớ không thể giảm được trên checkpoint đã
thử.

**Target repo:** vllm-project/vllm
**Type:** Feature request (not a bug — no existing code path is broken; a capability that exists for `nn.Linear`-family layers simply does not exist for `nn.Embedding`-family layers)
**Affected versions:** vllm 0.26
**Local fix:** None (this is precisely the gap — no local patch was attempted, since it requires adding a new quantization scheme entry point to vLLM itself, not a small anchor-based patch)
**Duplicate check:** No existing issue found specifically requesting compressed-tensors (or any general packed-int) quantization support for `ParallelLMHead`/`VocabParallelEmbedding`.

## Summary

We attempted to quantize `lm_head` (a `248320 x 4096` matrix on the checkpoint we tested — the single largest individual weight in the model after the embedding table, which it is tied to) to int8, group size 32, using the same RTN encoding scheme (`weight_packed`/`weight_scale`/`weight_shape`) already used successfully elsewhere in this project for quantizing GDN input projections (see the companion vLLM bug report on `MergedColumnParallelLinear` packed loading). The quantization arithmetic itself succeeds and is accurate (measured RMS error 0.0055 against the fp16 reference — well within acceptable range). But the resulting checkpoint fails to serve:

```
There is no module or parameter named 'lm_head.weight_packed'
```

## Root cause

`ParallelLMHead` inherits from `VocabParallelEmbedding` (the `Embedding` family — a flat `weight` parameter, sharded by vocabulary partition), not from any `Linear`-family base class. vLLM's compressed-tensors WNA16 scheme (`CompressedTensorsWNA16.create_weights`, and the broader machinery that recognizes `weight_packed`/`weight_scale`/`weight_shape` parameter names during checkpoint loading) is wired up exclusively against the `Linear`-family quantization method resolution path — there is no quantization method registration for `VocabParallelEmbedding`-family layers at all in the version we tested. This isn't a bug in the sense of broken code; the code path required to even attempt loading a quantized embedding/lm_head simply does not exist.

## Why this matters

`lm_head` (and the tied input embedding, when weights are tied — as they are on the checkpoint we tested) is often the single largest individual weight tensor in a small-to-mid-size model, because its size scales with vocabulary size rather than hidden size. On the 4B-parameter-class checkpoint we tested, the unquantized `lm_head`/embedding weight alone accounts for roughly 2 GiB — a fixed cost that cannot currently be reduced through quantization no matter what scheme is applied to the rest of the model, on this stack. For memory-constrained single-GPU deployments (our own motivating case: fitting a hybrid Mamba/attention model plus its KV cache on a single L4), this is a meaningfully large, currently-unavoidable chunk of VRAM.

## What we're asking for

A quantization method registration for `VocabParallelEmbedding`/`ParallelLMHead` analogous to what already exists for `Linear`-family layers under compressed-tensors WNA16 — i.e. recognizing `weight_packed`/`weight_scale`/`weight_shape` (or whatever scheme-appropriate parameter set) on an `Embedding`-family module, and dispatching to an appropriate dequantize/matmul (or embedding-lookup-then-dequantize) kernel at inference time. We do not have a proposed implementation for this — the quantize/dequantize arithmetic side is already proven (see our RMS-error-verified encoder, `scripts/graft_lm_head_int8.py`), but the vLLM-side loading/kernel-dispatch integration is a larger change than we're positioned to author blind, since it likely needs input from whoever designed the existing `Linear`-family scheme-registration mechanism to get the sharding/TP interaction right for a vocab-sharded embedding.

## What we verified is NOT the blocker

To rule out "we packed it wrong" before filing this as a vLLM-side gap rather than our own bug: the packed tensor shapes, parameter naming convention (`weight_packed`/`weight_scale`/`weight_shape`), and `config_groups` targeting all follow exactly the same convention that DOES work for `Linear`-family fused layers elsewhere in this same checkpoint (see the companion `MergedColumnParallelLinear` report) — the only difference is the base class of the target module. The error message (`no module or parameter named 'lm_head.weight_packed'`) is consistent with vLLM never having registered `weight_packed` as an expected parameter name on this module type in the first place, not with a shape/naming mismatch on our end.
