"""
Invert the weight transforms llama.cpp applies when converting Qwen3.5 to GGUF.

A Qwen3.5 GGUF is not a byte-faithful copy of the HF checkpoint. llama.cpp's
converter (conversion/qwen.py: Qwen3NextModel.modify_tensors and
_LinearAttentionVReorderBase.modify_tensors) rewrites both values and layout:

  1. A_log        stored as -exp(A_log)          (llama.cpp folds the runtime op)
  2. norm.weight  stored as 1 + weight           (HF norms are zero-centered:
                  Qwen3_5RMSNorm does x * (1.0 + w); linear_attn.norm is
                  excluded, it is a plain RMSNorm)
  3. conv1d       squeezed from (C,1,K) to (C,K)
  4. V heads      reordered grouped -> tiled, because ggml's binary ops
                  broadcast tiled. Qwen3.5-4B has 16 K heads and 32 V heads,
                  so this touches in_proj_qkv (V rows only), in_proj_z,
                  in_proj_a, in_proj_b, A_log, dt_bias, conv1d (V channels
                  only) and out_proj (columns).

vllm consumes HF semantics, so every one of these has to be undone on load.
The plugin only undid nothing at all, which is why the model loaded cleanly and
then emitted pure garbage ("!!!!!!") — the gated-delta-net path was fed
-exp(-exp(A_log)), norms were off by one, and every V head sat in the wrong
slot. The dequantisation kernels themselves are fine (verified against
gguf.quants.dequantize on the same tensors: rel. error ~5e-4, i.e. fp16 noise).

The inverse of the tiled reorder is the same permutation with num_k_heads and
num_v_per_k swapped, so one helper covers both directions.
"""

import glob
import sysconfig

# ---------------------------------------------------------------- default.py

CFG_MARKER = "# --- custom_vllm: hand the qwen3.5 head layout to transform_weight ---"
CFG_ANCHOR = '''        if model_type == "qwen3_5_moe":
            model_type = "qwen35moe"
'''
CFG_PATCH = (
    CFG_ANCHOR
    + f"        {CFG_MARKER}\n"
    "        from . import base as _gguf_base\n"
    "        _gguf_base._QWEN35_CFG = (\n"
    '            text_config if model_type in ("qwen35", "qwen35moe") else None\n'
    "        )\n"
)

# -------------------------------------------------------------------- base.py

TRANSFORM_MARKER = "# --- custom_vllm: undo llama.cpp's Qwen3.5 GGUF weight transforms ---"
CONV1D_MARKER = "# --- custom_vllm: GGUF stores conv1d weight 2-D, vllm expects 3-D ---"

TRANSFORM_ANCHOR = '        """Transform one loaded weight."""\n'
TRANSFORM_PATCH = f'''        """Transform one loaded weight."""
        {TRANSFORM_MARKER}
        if _QWEN35_CFG is not None and ".linear_attn." in hf_name:
            weight = _undo_qwen35_gguf_transform(hf_name, weight, _QWEN35_CFG)
        elif _QWEN35_CFG is not None and hf_name.endswith("norm.weight"):
            # llama.cpp writes 1 + w for every zero-centred RMSNorm.
            weight = weight - 1.0
'''

HELPERS = f'''

{TRANSFORM_MARKER}
_QWEN35_CFG = None


def _qwen35_untile_v_heads(t, dim, num_k_heads, num_v_per_k, head_dim):
    """Inverse of llama.cpp's grouped -> tiled V-head permutation."""
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    # tiled layout is [v0_k0..v0_k{{K-1}}, v1_k0, ...]; read it back as
    # (num_v_per_k, num_k_heads, head_dim) and swap to restore grouping.
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def _undo_qwen35_gguf_transform(hf_name, weight, cfg):
    import torch

    num_k = getattr(cfg, "linear_num_key_heads", 0)
    num_v = getattr(cfg, "linear_num_value_heads", 0)
    head_k = getattr(cfg, "linear_key_head_dim", 0)
    head_v = getattr(cfg, "linear_value_head_dim", 0)
    reorder = num_k > 0 and num_v > 0 and num_k != num_v
    num_v_per_k = (num_v // num_k) if num_k else 1

    def untile(t, dim, head_dim):
        if not reorder:
            return t
        return _qwen35_untile_v_heads(t, dim, num_k, num_v_per_k, head_dim)

    if hf_name.endswith("linear_attn.A_log"):
        # GGUF holds -exp(A_log); vllm re-applies -exp() at runtime.
        weight = torch.log(weight.float().neg().clamp_min(1e-30)).to(weight.dtype)
        return untile(weight, 0, 1)

    if hf_name.endswith("linear_attn.dt_bias"):
        return untile(weight, 0, 1)

    if hf_name.endswith("linear_attn.conv1d.weight"):
        if weight.ndim == 3:  # already (C, 1, K)
            weight = weight.squeeze(1)
        if reorder:
            qk = head_k * num_k * 2
            weight = torch.cat(
                [weight[:qk], untile(weight[qk:], 0, head_v)], dim=0
            )
        return weight.unsqueeze(1)

    if hf_name.endswith("linear_attn.in_proj_qkv.weight"):
        if reorder:
            qk = head_k * num_k * 2
            weight = torch.cat(
                [weight[:qk], untile(weight[qk:], 0, head_v)], dim=0
            )
        return weight

    if hf_name.endswith("linear_attn.in_proj_z.weight"):
        return untile(weight, 0, head_v)

    if hf_name.endswith(
        ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")
    ):
        return untile(weight, 0, 1)

    if hf_name.endswith("linear_attn.out_proj.weight"):
        return untile(weight, 1, head_v)

    return weight
'''


def patch(path, marker, anchor, replacement, *, append=None):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if marker in src:
        print(f"Already patched: {path}")
        return
    if anchor not in src:
        raise SystemExit(f"Anchor not found in {path}; plugin source may have changed")
    src = src.replace(anchor, replacement, 1)
    if append:
        src += append
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")


site_packages = sysconfig.get_paths()["purelib"]
default_py = glob.glob(f"{site_packages}/vllm_gguf_plugin/weights_adapter/default.py")
base_py = glob.glob(f"{site_packages}/vllm_gguf_plugin/weights_adapter/base.py")
if not default_py or not base_py:
    raise SystemExit(f"vllm_gguf_plugin weights_adapter not found under {site_packages}")

patch(default_py[0], CFG_MARKER, CFG_ANCHOR, CFG_PATCH)
patch(base_py[0], TRANSFORM_MARKER, TRANSFORM_ANCHOR, TRANSFORM_PATCH, append=HELPERS)
