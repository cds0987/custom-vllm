# [Bug] Tuple shard-ids collapse per-shard GGUF weight types into one, mis-decoding some shards with the wrong quant type (Q5_K read as Q4_K)

## Tóm tắt cho người không chuyên

Một lớp mạng "gộp" của mô hình được ghép từ nhiều mảnh nhỏ, mỗi mảnh có thể
được nén theo một kiểu nén khác nhau. Phần mềm nạp file lại đôi khi gộp
nhầm nhãn "kiểu nén" của nhiều mảnh vào chung một ô nhớ, khiến một mảnh dữ
liệu được đọc bằng công thức giải nén SAI (ví dụ đọc dữ liệu nén 5-bit như
thể nó là dữ liệu nén 4-bit). Kết quả nhẹ thì báo lỗi ngay lúc chạy, nặng thì
suy ra sai số âm thầm nếu người dùng chỉ vá một nửa lỗi. Bản vá đảm bảo mỗi
mảnh luôn giữ đúng nhãn kiểu nén của riêng nó.

**Target repo:** vllm-project/vllm-gguf-plugin
**Severity:** High (crash on some fused-layer configs; silent wrong-dtype decode on others)
**Affected versions:** vllm-gguf-plugin 0.0.4, vllm 0.26
**Local fix:** `scripts/patch_gguf_weight_type_loader.py`
**Duplicate check:** No existing issue found in vllm-project/vllm-gguf-plugin covering this.

## Summary

`_gguf_weight_type_loader_v2()` in `vllm_gguf_plugin/quantization/params.py` only takes its fast "store directly" path when there is no shard id:

```python
def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):
    if loaded_shard_id is None and hasattr(param, "_store"):
        param._store(loaded_weight)
        return
    base_loader(param, loaded_weight, loaded_shard_id)
```

For any fused layer (`QKVParallelLinear`, `MergedColumnParallelLinear` — any model with fused `qkv_proj`/`gate_up_proj`, which includes Qwen3.5's `in_proj_qkvz`), the per-shard weight-*type* tag legitimately does carry a shard id, so this falls through to vllm's generic `weight_loader_v2` → `_load_fused_module_from_checkpoint`, which reads `param.output_dim`:

```
AttributeError: 'GGUFUninitializedWeightTypeParameter' object has no attribute 'output_dim'
```

A GGUF weight-type parameter is a scalar ggml-dtype tag, not a sharded weight tensor — it has no `input_dim`/`output_dim` by design. Routing it through the fused-weight path is itself the bug; `_store_gguf_weight_type()` already exists specifically to handle the sharded case (it writes into `param.shard_weight_type[shard_id]`).

## A second bug once the first is naively fixed

The naive fix — "always call `_store()`, shard id or not" — surfaces a second bug: vllm sometimes hands the loader a **tuple** of shard ids when one GGUF tensor feeds several shards of a fused layer at once. Storing under the tuple key (instead of iterating it) makes every later *per-shard* lookup miss its type, and `GGUFLinearMethod.apply()` silently falls back to `layer.qweight_type.weight_type` for those shards — i.e. it picks one dtype for a set of shards that don't actually share one.

## Reproduction and exact error text

Serving Qwen3.5's `in_proj_qkvz` (a fused module) reproduces both failure modes depending on which half of the fix is applied:

Without any fix — `AttributeError`:
```
AttributeError: 'GGUFUninitializedWeightTypeParameter' object has no attribute 'output_dim'
```

With only the naive "always `_store()`" half-fix — silent mis-decode, surfacing later as:
```
shard_weight_type = {3: 12, (0, 1, 2): 13}
```
i.e. the true per-shard types were `{q: 13, k: 13, v: 13, z: 12}` (Q5_K = 13 for q/k/v, Q4_K = 12 for z), but the tuple key collapses q/k/v into `(0, 1, 2): 13`... except once every shard subsequently reports the *same* recorded type via the fallback path, `apply()` takes its single-type fast path and runs one matmul over the zero-padded concatenation sized for the *widest* shard:

```
ValueError: Invalid row width 1760 for quant type 12: must be divisible by 144
```

(1760 bytes = 2560/256 superblocks × 176 bytes/superblock = Q5_K's row width; Q4_K would need 1440. The mismatch is the give-away that a Q5_K-shaped row is being decoded as Q4_K.)

## The fix

```python
def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):
    if hasattr(param, "_store"):
        if isinstance(loaded_shard_id, (tuple, list)):
            for _sid in loaded_shard_id:
                param._store(loaded_weight, shard_id=_sid)
        else:
            param._store(loaded_weight, shard_id=loaded_shard_id)
        return
    base_loader(param, loaded_weight, loaded_shard_id)
```

Two changes from upstream: (1) drop the `loaded_shard_id is None` guard so `_store()` is always used when available — it already accepts `shard_id=None` for the non-sharded case; (2) expand a tuple/list shard id into one `_store()` call per shard, so each shard keeps its own recorded ggml type instead of collapsing onto one dict entry.

## Verification

After the fix, `in_proj_qkvz`'s recorded shard types are distinct per shard (`{0: 13, 1: 13, 2: 13, 3: 12}` instead of `{3: 12, (0,1,2): 13}`), `apply()` takes its mixed-type decode path, and each shard is dequantized with its correct ggml type. Confirmed by loading and serving Qwen3.5-4B Q5_K_M without the `ValueError` and with numerically correct output (cross-checked against the HF reference per report 02).
