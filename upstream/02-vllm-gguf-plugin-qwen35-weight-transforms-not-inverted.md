# [Bug] Qwen3.5 GGUF checkpoints are not byte-faithful HF weights — llama.cpp's conversion-time transforms are never inverted on load

## Tóm tắt cho người không chuyên

Công cụ tạo file GGUF (llama.cpp) không chỉ nén số mà còn lặng lẽ "biến đổi"
một số con số trong lúc lưu file — ví dụ đảo dấu, cộng thêm 1, hay sắp xếp
lại thứ tự — vì bản thân nó cần lưu như vậy để chạy nhanh. Phần mềm nạp file
này vào vLLM lại không biết phải "giải mã ngược" các phép biến đổi đó trước
khi dùng, nên các con số bị sai lệch nghiêm trọng và mô hình trả lời hoàn
toàn vô nghĩa dù không hề báo lỗi. Cách sửa là thêm đúng 4 phép tính ngược
tương ứng lúc nạp file, và nhóm nghiên cứu đã đối chiếu số học để xác nhận
kết quả khớp với bản gốc.

**Target repo:** vllm-project/vllm-gguf-plugin
**Severity:** Critical (silent correctness failure — model loads and runs, output is garbage)
**Affected versions:** vllm-gguf-plugin 0.0.4, vllm 0.26, gguf 0.19.0
**Local fix:** `scripts/patch_gguf_qwen35_transforms.py`
**Duplicate check:** vllm-project/vllm-gguf-plugin#80 ("GGUF Qwen 3.5/3.6 MoE architecture not supported") is the tracking issue for Qwen3.5 support in general but has no fix merged upstream; one commenter links a third-party fork (`localweights/vllm-gguf-plugin`) that reportedly adds support, but that work has not been submitted upstream and we have not audited it for overlap with the specific transforms documented below. No other existing issue covers these transforms specifically.

## Summary

A Qwen3.5 GGUF file is **not** a byte-faithful copy of the HF checkpoint's tensor values or layout. llama.cpp's converter (`convert_hf_to_gguf.py`, `Qwen3NextModel.modify_tensors` and `_LinearAttentionVReorderBase.modify_tensors`) applies four transforms when writing the GGUF, folding runtime operations and llama.cpp-internal layout requirements into the stored bytes. `transformers`' own GGUF loader has no Qwen3.5/qwen3next entry at all (confirmed: `transformers/modeling_gguf_pytorch_utils.py`'s architecture table does not include `qwen35`/`qwen3next`), so there is no reference implementation anywhere in the Python ecosystem that already does this inversion — vllm-gguf-plugin is the only engine attempting to load these files, and it currently loads them with **zero** transform-inversion, i.e. it treats the GGUF tensor values as if they were the HF values verbatim.

## The four transforms and their inverses

### 1. `A_log` stored as `-exp(A_log)`

llama.cpp folds the runtime negate-and-exponentiate that vllm (and HF) apply at inference time directly into the stored weight. vllm expects to read the raw `A_log` and apply `-exp()` itself at runtime:

```python
g = -A_log.float().exp() * softplus(a + dt_bias)   # vllm runtime
```

If the GGUF value is taken as-is, vllm double-applies the transform: `-exp(-exp(A_log))` instead of `-exp(A_log)`. Inverse:

```python
weight = torch.log(weight.float().neg().clamp_min(1e-30)).to(weight.dtype)
```

### 2. Norm weights stored as `1 + w`

`Qwen3_5RMSNorm` computes `x * (1.0 + w)`, i.e. HF's `w` is zero-centered (a checkpoint with `w == 0` reproduces the un-normalized identity). llama.cpp folds the `+1` into the stored weight so its own runtime norm can be a plain multiply. Inverse: `weight = weight - 1.0`. This applies to every `*.norm.weight` tensor **except** `linear_attn.norm`, which is a plain (non-zero-centered) RMSNorm and must NOT have 1 subtracted.

### 3. `conv1d` weight squeezed from `(C, 1, K)` to `(C, K)`

Pure storage-layout difference (covered together with the general GGUF-vs-vllm `Conv1d` shape mismatch in a separate report, since it also applies to non-Qwen3.5 hybrid models); this report's transform code restores `(C, 1, K)` (`weight.unsqueeze(1)`) after undoing the V-head reorder below.

### 4. V-heads reordered grouped → tiled

Qwen3.5's gated-delta-net has more value heads than key heads (e.g. Qwen3.5-4B: 16 K heads, 32 V heads — 2 V heads share each K head). HF's grouped layout lays out V heads as `[k0_v0, k0_v1, k1_v0, k1_v1, ...]`. ggml's binary ops broadcast along a *tiled* layout instead — `[v_slot0_over_all_k, v_slot1_over_all_k, ...]` — so llama.cpp's converter permutes every V-head-shaped tensor into tiled order before writing it. This touches the V rows of `in_proj_qkv`, all of `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, the V-channel slice of `conv1d`, and the columns of `out_proj`.

The inverse is the same permutation with `num_k_heads` and `num_v_per_k` swapped (grouped ↔ tiled is a self-inverse-shaped operation with the axis roles exchanged):

```python
def _qwen35_untile_v_heads(t, dim, num_k_heads, num_v_per_k, head_dim):
    shape = list(t.shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim+1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim+1] = perm[dim+1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)
```

### The `out_proj` special case

`out_proj`'s V-head permutation runs along its **input** dimension, which for quantized tensors is packed inside ggml's block layout — it cannot be permuted on the raw bytes. `gguf-py` can *decode* every K-quant type but its `quantize_blocks` only implements F16/F32/Q8_0 encoding (K-quants raise `NotImplementedError`). We dequantize, permute, and re-encode as Q8_0 — whose error (~0.2%) sits comfortably under the K-quant error already present in the file (Q5_K: ~2.7%), at the cost of ~94 MB more across the 24 gated-delta-net layers of a 4B model. The ggml dtype tag (`.qweight_type`) is not recoverable from the packed bytes alone (Q4_0 and Q4_K are both 144 bytes per 256 weights; several IQ types collide too), so we cache it from the preceding `.qweight_type` tensor, which always arrives immediately before its module's `.qweight` in the GGUF's tensor order.

## Reproduction

Serve a Qwen3.5 GGUF (any quant) with none of these inversions applied (i.e. current vllm-gguf-plugin 0.0.4):

```bash
vllm serve unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  --hf-config-path Qwen/Qwen3.5-4B --tokenizer Qwen/Qwen3.5-4B
```

Model loads without error. Every response degenerates to `"!!!!!!!!!!!!!!!!"` — the same visible symptom as the trailing-dot bug (report 01), because both bugs corrupt the same gated-delta-net path, but this one persists even after the trailing-dot fix (report 01) is applied: `A_log` now *loads*, but its stored value is `-exp(A_log_true)`, not `A_log_true`, so downstream computation is still wrong (`-exp(-exp(A_log_true))` instead of `-exp(A_log_true)`), just via a different arithmetic error with the same visible collapse.

## The fix

`scripts/patch_gguf_qwen35_transforms.py` (in our repro repo) patches two files in vllm-gguf-plugin:

- `weights_adapter/default.py`: stashes the resolved text config (needed for head counts/dims) into a module-level variable readable by `base.py`'s `transform_weight`.
- `weights_adapter/base.py`: adds a `_undo_qwen35_gguf_transform()` dispatcher, called from `transform_weight()` whenever `hf_name` contains `.linear_attn.` or ends in `norm.weight`, implementing all four inversions above.

## Verification

Weights compared numerically against the original HF checkpoint after applying this patch together with report 01's fix: `A_log`, `dt_bias`, `conv1d`, and norm weights match the HF reference to full precision; quantized tensors (everything routed through actual GGUF dequantization) differ by 0.004–0.07, consistent with expected quantization noise, not a transform bug. Model output is coherent.

## Scope note

We are not aware of any other GGUF-consuming Python engine (transformers, llama-cpp-python bindings used as a library, etc.) that implements this inversion for Qwen3.5/qwen3next — `transformers`' native GGUF loader (`modeling_gguf_pytorch_utils.py`) does not have a `qwen35`/`qwen3next` entry in its supported-architecture table at all, so it cannot even attempt to load these files, correctly or not. vllm-gguf-plugin is, to our knowledge, the first and only Python-side consumer of Qwen3.5 GGUF files, so there was no prior art to copy from; this report is intended to serve as that reference.
