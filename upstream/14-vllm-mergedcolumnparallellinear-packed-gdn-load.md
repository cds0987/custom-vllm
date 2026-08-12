# [Bug] `MergedColumnParallelLinear.weight_loader_v2` cannot load compressed-tensors packed shards — blocks quantizing Qwen3.5's (and any fused-projection hybrid model's) GDN input projections; plus a silent `weight_shape` corruption once shards are packed

## Tóm tắt cho người không chuyên

Một số lớp mạng được "ghép" từ nhiều tệp trọng số nhỏ thành một khối lớn lúc
nạp. Khi những tệp nhỏ đó đã được nén (kiểu 4-bit/8-bit), bộ nạp trọng số của
vLLM không biết cách ghép chúng đúng cách và bị lỗi ngay (thông báo lỗi rất
khó hiểu). Ngay cả khi vượt qua lỗi này, có thêm một lỗi âm thầm khác: một
mẩu thông tin phụ ghi "kích thước thật" của khối ghép bị ghi đè nhầm thành
kích thước của MỘT mảnh nhỏ cuối cùng thay vì kích thước của cả khối — có
thể khiến bước nén-lại phần cứng sau đó tính sai mà không báo lỗi. Bản vá đã
sửa cả hai, cho phép lần đầu tiên nén được toàn bộ phần "trí nhớ" đặc biệt
của Qwen3.5.

**Target repo:** vllm-project/vllm
**Severity:** High (blocks a whole class of checkpoints — any quantized compressed-tensors scheme applied to a fused/merged linear layer with more than one packed sub-projection — with either an opaque `AssertionError` or, once naively worked around, a silent shape-corruption that can propagate into a wrong Marlin/Machete repack)
**Affected versions:** vllm 0.26 (`vllm/model_executor/layers/linear.py`, `MergedColumnParallelLinear.weight_loader_v2`; `vllm/model_executor/layers/parameter.py`)
**Local fix:** `scripts/patch_vllm_gdn_quant_load.py`
**Duplicate check:** No existing issue found in vllm-project/vllm specifically about `MergedColumnParallelLinear.weight_loader_v2` mishandling compressed-tensors `weight_packed`/`weight_shape` parameters for a fused layer whose sub-projections were quantized independently. Adjacent to compressed-tensors' own `find_matched_target` routing bug (companion report to compressed-tensors) but a distinct failure in a different codebase (vllm's own loader, not compressed-tensors' target-matching), reachable only once that routing bug is worked around.

## Summary

Qwen3.5's gated-delta-net (GDN) attention fuses four separate HF checkpoint projections into two vLLM `MergedColumnParallelLinear` parameters (`vllm/model_executor/models/qwen3_5.py`'s `hf_to_vllm_mapper`):

```
in_proj_qkv + in_proj_z -> in_proj_qkvz   (output_sizes=[key,key,val,val])
in_proj_b   + in_proj_a -> in_proj_ba     (output_sizes=[num_v_heads]*2)
```

When every one of the four source HF modules is plain fp16, `MergedColumnParallelLinear.weight_loader_v2` (`vllm/model_executor/layers/linear.py`) works correctly: each shard is a plain tensor, sliced and copied by output-dim offset. Quantizing any of the four with compressed-tensors (the AWQ/GPTQ W4A16 path) breaks this.

## Bug 1 — opaque `AssertionError` on packed shards

`CompressedTensorsWNA16.create_weights` declares `weight_packed = PackedvLLMParameter(input_dim=1, output_dim=0, packed_dim=1, ...)` — packing lives on dim 1 (K, the reduction axis), while the merged-column loader shards along dim 0 (N, the output axis). Since `packed_dim (1) != output_dim (0)`, concatenating independently-packed shards along N is mathematically valid provided every shard packs K identically (same `num_bits`, same `group_size`). But `weight_loader_v2`'s single-shard branch calls `param.load_merged_column_weight(...)`, which for a `PackedvLLMParameter` ends in:

```python
assert param_data.shape == loaded_weight.shape   # parameter.py:175
```

`shard_offset`/`shard_size` are computed purely from `self.output_sizes` (correct — packed_dim != output_dim means no packed_factor scaling applies), but `loaded_weight` is whatever `weight_packed` tensor that ONE HF sub-module's own quantization produced. If the two halves of a merge pair were quantized with different `num_bits`, their packed-K widths differ and the assert fires with no indication of why — an opaque crash deep in `parameter.py` for what is, at the checkpoint level, a perfectly legitimate quantization scheme.

## Bug 2 — `weight_shape` silently corrupted once packed shards are made to load

While fixing bug 1, we traced a second, non-crashing bug in the same method. `weight_shape` is a plain `BasevLLMParameter` (a 2-element int64 `[N, K]` metadata tensor compressed-tensors kernels use to know the logical, pre-packing shape for Marlin/Machete repacking) — it has no `output_dim`, so it is not a `_ColumnvLLMParameter` and has no per-shard narrowing logic. `weight_loader_v2` dispatches it through `BasevLLMParameter.load_merged_column_weight`, which ignores all `shard_offset`/`shard_size` kwargs and does a full `_assert_and_load` (`parameter.py:105-106`). Since every shard's own `weight_shape` is ALSO shape `(2,)`, the assert trivially passes — but each shard call *overwrites the entire parameter*, so after all shards load, `layer.weight_shape` silently holds whichever sub-module's `[N, K]` was loaded **last** (e.g. `in_proj_z`'s `[value_dim, hidden]` instead of the true merged `in_proj_qkvz` shape `[key_dim*2+value_dim*2, hidden]`). Whatever kernel consumes `weight_shape` during `process_weights_after_loading` (Marlin repack reads it to reshape `weight_packed`) would then either crash on an irreconcilable shape, or — worse — silently repack against the wrong logical shape.

## The fix

Both bugs are patched inside `MergedColumnParallelLinear.weight_loader_v2`:

- A `weight_shape`-shaped `BasevLLMParameter` (int64, `numel()==2`) is special-cased before the normal shard/tuple dispatch: instead of copying whatever shard tensor arrives, its data is always set to `[self.output_size_per_partition, self.input_size_per_partition]` — values the layer already knows from `create_weights`, independent of shard load order, so idempotent across repeated calls.
- The final single-shard `param.load_merged_column_weight(...)` call (the one that raises the `AssertionError` for packed params) is wrapped so that, specifically when the param is a packed compressed-tensors parameter whose `packed_dim` is not the `output_dim` being sharded, the stock assert is re-raised as a `RuntimeError` naming the layer, the shard, and the concrete likely cause (mismatched `num_bits`/`group_size` across the sub-projections folded into this fused param) — turning an undiagnosable crash into an actionable one. Any other `AssertionError` (unrelated params/paths) is re-raised unchanged.

Deliberately not attempted: repacking mismatched-scheme shards. The concat-only approach requires every sub-projection merged into one vLLM param to share `num_bits` and `group_size` — a constraint the quantize-side tooling must respect (each merge pair internally consistent; the two pairs may still differ from each other, since they land in different vLLM params).

## Verification

With this patch, a compressed-tensors-quantized Qwen3.5 checkpoint with GDN's `in_proj_qkvz`/`in_proj_ba` quantized (int4 and int8 variants both tested) loads through vLLM's stock `MergedColumnParallelLinear` path, logs `Using MarlinLinearKernel for CompressedTensorsWNA16` for the GDN shards, and serves without the `AssertionError`. This is, to our knowledge, the first Qwen3.5 checkpoint with its GDN mixer fully quantized (not just attention/MLP) to load and serve on vLLM.
