#!/usr/bin/env python3
"""
TASK M -- graft GGUF-quantized GDN weights into a compressed-tensors champion.

BACKGROUND
----------
RedHatAI/Qwen3.5-9B-quantized.w4a16 (the "champion", compressed-tensors
W4A16, served through vLLM's Marlin kernel) keeps every GDN mixer's four
input projections (`linear_attn.in_proj_{qkv,z,b,a}`, ~75% of Qwen3.5-9B's
layers) fp16 -- they sit in the champion's `ignore` list. TASK G / G2a tried
quantizing those four projections with GPTQ (llm-compressor's GPTQModifier,
Hessian-calibrated) at int4 AND int8 and both cost ~15% ppl (5.96 vs the
champion's 5.1578, a WARN). The GGUF Q4_K_M release of the same model
(unsloth/Qwen3.5-9B-GGUF) quantizes those same tensors with llama.cpp's
calibration-free RTN K-quant scheme and clears the ppl gate -- so the
quality cost measured by G/G2a is plausibly specific to GPTQ's
Hessian/calibration step on these particular tensors (they feed GDN's
sigmoid/exp gating nonlinearities directly, not a plain attention score),
not to int4-vs-int8 bit width. This script tests that hypothesis mechanically:
lift the four in_proj_* tensors' *values* out of the GGUF (llama.cpp's own
RTN quantization, no calibration), re-encode them into a compressed-tensors
int8 group-quantized parameter (matching the champion's on-disk format
exactly, verified against local vLLM source below), and splice them into an
otherwise-untouched copy of the champion checkpoint. Every other tensor
(including the champion's own int4 W4A16 weights) is copied byte-for-byte.

REUSED, NOT REIMPLEMENTED
--------------------------
- scripts/gguf2marlin.py's `quantize_symmetric_group` (RTN symmetric
  affine-grid re-fit + small clip-ratio search) and `pack_rows`/
  `dequant_gptq_for_verification` (vLLM GPTQ on-disk bit-packing, verified
  against vllm/model_executor/layers/quantization/utils/quant_utils.py) do
  ALL of the actual dequant -> int8 grid math here; imported via
  importlib, not copy-pasted. Its `--k-quants-to int8` branch is the exact
  precedent for "GGUF K-quant tensor -> vLLM-servable int8 GPTQ code" this
  script leans on (see that module's DECISION 1b).
- scripts/patch_vllm_gdn_quant_load.py documents (and fixes) vLLM's
  merged-column loader for quantized GDN shards: within a fused pair
  (in_proj_qkv+in_proj_z -> vLLM's in_proj_qkvz, or in_proj_b+in_proj_a ->
  in_proj_ba) every shard MUST share the same on-disk scheme, because
  compressed-tensors' packing lives on the K axis (input dim), independent
  of the N-axis (output/channel) concatenation the merged loader performs.
  This script enforces that per-pair (see `validate_merge_pairs` below) and
  fails loudly, rather than silently degrading, on a mismatch.
- scripts/patch_gguf_tensor_mapping.py / scripts/patch_gguf_qwen35_transforms.py
  document the concrete llama.cpp -> HF value/layout transforms Qwen3.5
  GDN tensors go through in a GGUF; the V-head "tiled" reorder those
  patches undo for the *live-serving* GGUF path is reimplemented here in
  numpy (see `untile_v_heads` below) because this script reads the GGUF
  offline (no vllm_gguf_plugin / vLLM runtime available on this box).

DECISION 1 -- ON-DISK LAYOUT: compressed-tensors int8 WNA16, group-quantized
-----------------------------------------------------------------------------
Read directly from the local vLLM checkout
(D:\\Training\\AI_Module\\vllm\\vllm\\vllm), not assumed:

  vllm/model_executor/layers/quantization/compressed_tensors/schemes/
  compressed_tensors_wNa16.py, CompressedTensorsWNA16.create_weights
  (lines ~160-224):
      weight_packed = PackedvLLMParameter(input_dim=1, output_dim=0,
          packed_dim=1, packed_factor=32/num_bits,
          data=torch.empty(output_size_per_partition,
                            ceil(input_size_per_partition*num_bits/32),
                            dtype=torch.int32))
      weight_scale  = GroupQuantScaleParameter(output_dim=0, input_dim=1,
          data=torch.empty(output_size_per_partition,
                            input_size_per_partition // group_size,
                            dtype=params_dtype))
      weight_shape  = BasevLLMParameter(data=torch.empty(2, dtype=torch.int64))
  i.e. ON DISK the packed weight is (N, K_packed) -- **N-major**, transposed
  relative to a plain GPTQ checkpoint's (K_packed, N) -- and weight_scale is
  (N, num_groups), also N-major.

  vllm/model_executor/kernels/linear/mixed_precision/marlin.py,
  MarlinLinearKernel.process_weights_after_loading (lines 86-142):
      # note assumes that
      #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
      #  `weight_scale` is: {input_dim = 0, output_dim = 1}
      permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
  i.e. the FIRST thing the kernel does with the loaded weight_packed/
  weight_scale is transpose them into (K_packed, N) / (num_groups, N) --
  vllm/model_executor/parameter.py's `permute_param_layout_` (lines
  544-584) computes this via `torch.permute`, which for a 2-D tensor
  swapping dim 0<->1 IS a transpose. After that transpose, weight_packed's
  packed_dim is 0 (the K axis) -- textbook GPTQ pack_rows layout. CONCLUSION
  (not assumed, derived): the on-disk compressed-tensors int8 weight_packed
  is exactly `gguf2marlin.pack_rows(q_biased, 8, K, N).T` -- pack with
  gguf2marlin's own vetted int8 pack_rows (bit order: row i of each 4-row
  group in byte i of the int32, verified against vLLM's quant_utils.py by
  scripts/test_gguf2marlin.py's test_q4k_int8_branch), then transpose to
  (N, K_packed) before writing to safetensors. Same logic for weight_scale:
  gguf2marlin's `quantize_symmetric_group` scales output is (num_groups, N);
  transpose to (N, num_groups) for the on-disk layout.

  vllm/model_executor/layers/quantization/compressed_tensors/schemes/
  compressed_tensors_wNa16.py:38-48, WNA16_SUPPORTED_TYPES_MAP includes
  `8: scalar_types.uint8b128` -- 8-bit symmetric (zero=128=2**(8-1)) IS a
  first-class compressed-tensors WNA16 type, matching gguf2marlin's own
  `bits_zero_and_range(8) == (128, -128, 127)` exactly (same
  "2**(bits-1)" convention as vLLM's scalar_type.py, see gguf2marlin.py's
  DECISION 1b for the uint4b8/uint8b128 cross-check).

DECISION 2 -- GROUP SIZE: 32 (not 128, TASK G2a's choice)
------------------------------------------------------------
vllm/model_executor/kernels/linear/mixed_precision/marlin.py:53,
`MarlinLinearKernel.can_implement`: `if c.group_size not in
MARLIN_SUPPORTED_GROUP_SIZES: return False, ...` where
MARLIN_SUPPORTED_GROUP_SIZES = (-1, 32, 64, 128) (marlin_utils.py:35) is
checked with NO num_bits branch anywhere it's consumed -- the same list
serves every bit width Marlin supports (confirmed: gptq_marlin.py's own
SUPPORTED_GROUP_SIZES / SUPPORTED_GPTQ_QUANT_TYPES pairing in
quant_utils.py:705-706 is one shared list too). So group_size=32 is just as
valid for this int8 graft as it is for the champion's int4 groups or for
G2a's int8 g128 choice -- there is no "8-bit only supports coarser groups"
constraint in this codebase. This script defaults to 32 (not G2a's 128)
because: (a) it matches gguf2marlin's own default and Q4_0/Q4_K's native
32-wide sub-block granularity (regrouping into a coarser 128-wide group
would average over 4 of the *source* GGUF's own quantization boundaries for
no benefit -- the source data was never coherent at that granularity to
begin with), and (b) G2a's ~15% ppl cost was traced to GPTQ's
Hessian/calibration step, not to group_size or bit width (see BACKGROUND
above) -- so this is not "undoing" a group-size mistake, it is simply using
the finer, source-native grouping since group_size stopped being suspected
of anything. --group-size accepts any of vLLM's MARLIN_SUPPORTED_GROUP_SIZES.

DECISION 3 -- THE V-HEAD "TILE" TRANSFORM (must be undone before quantizing)
-------------------------------------------------------------------------------
scripts/patch_gguf_qwen35_transforms.py (module docstring + code) documents
that llama.cpp's GGUF converter stores Qwen3.5's V-head-carrying GDN tensors
(in_proj_qkv's V-rows, in_proj_z, in_proj_a, in_proj_b) in a "tiled" row
order rather than HF's "grouped" order, whenever
`linear_num_key_heads != linear_num_value_heads` (true whenever GDN uses
grouped-value heads, i.e. `num_v_per_k = linear_num_value_heads //
linear_num_key_heads > 1`). Skipping this step would graft
*mis-ordered* rows into the champion checkpoint -- silently wrong outputs,
not a crash, and not something the RMS-error report below would catch
(both sides of the RMS comparison would be equally tiled/untiled together,
since it re-derives its own reference from the same GGUF read). This
script ports `_qwen35_untile_v_heads` / `_undo_qwen35_gguf_transform` from
that patch to plain numpy (`untile_v_heads` / `untile_module` below) --
same math, since this script reads the GGUF offline rather than through
vllm_gguf_plugin's live loader. GDN's other transformed tensors (A_log,
dt_bias, conv1d, out_proj, norms) are OUT OF SCOPE here -- this script only
touches the four in_proj_* input projections.

DECISION 4 -- CONFIG.JSON SURGERY: additive, not destructive
-----------------------------------------------------------------
Read from vllm/model_executor/layers/quantization/compressed_tensors/
utils.py:
  - `_is_equal_or_regex_match` (lines 175-193): an `ignore` entry matches a
    layer name either by exact string equality, or -- if the entry starts
    with "re:" -- via `re.match(pattern, layer_name)` (start-anchored, NOT
    full-string; a trailing "$" in the pattern is what anchors the end).
  - `should_ignore_layer` / `check_equal_or_regex_match` (lines 50-110):
    this is the ONLY matching logic vLLM runs against `ignore`; no
    "contains" fallback is used for ignore lists (only `find_matched_target`
    uses that, for `config_groups[*].targets`, which this script also
    avoids relying on by writing exact literal target names instead of a
    regex -- see `_build_new_config_group` below).
  - `CompressedTensorsConfig.from_config` (compressed_tensors.py:227-263)
    reads `ignore` and `config_groups` straight from the top-level
    `quantization_config` dict with no extra transformation before
    `apply_vllm_mapper` renames layer-path targets -- so editing those two
    JSON fields directly, in place, with the exact same string convention
    the champion's own config.json already uses, is what vLLM will load.
Rather than deleting or rewriting the champion's `ignore` entries wholesale
(risking silently un-ignoring GDN submodules this script does NOT quantize
-- conv1d, norms, A_log, dt_bias, out_proj), this script only narrows
existing `ignore` entries that match a to-be-grafted module's exact name,
via `_is_equal_or_regex_match`-compatible negative-lookahead rewriting, and
RE-VERIFIES (against every grafted name and a set of "must stay ignored"
probe names -- other GDN submodules, non-grafted layers, lm_head) that the
rewritten entry still means exactly "everything the old one covered, minus
the modules actually grafted" before accepting it. A verification failure
raises `GraftConfigError` rather than writing a silently-wrong config.json.

USAGE
-----
    python scripts/graft_gguf_gdn.py --frame <champion_dir> \\
        --gguf <path/to/Qwen3.5-9B-Q4_K_M.gguf> --out <out_dir> \\
        [--group-size 32]

--frame accepts EITHER of two real module-naming conventions, detected
automatically (see `detect_module_prefix`): a text-only checkpoint already
run through fix_qwen35_hf_checkpoint.py ("model.layers.*"), or a
genuinely multimodal-capable checkpoint -- RedHatAI/Qwen3.5-9B-
quantized.w4a16 included -- used completely UNMODIFIED, straight from the
Hub ("model.language_model.layers.*"). Do NOT run
fix_qwen35_hf_checkpoint.py on the latter kind: that fixer's
"language_model." strip is only correct for text-only checkpoints loaded
through vLLM's Qwen3_5ForCausalLM class; a multimodal-capable checkpoint
(one that still carries a real vision tower, "visual.*" weights) loads
through Qwen3_5ForConditionalGeneration instead, which DOES have a
`language_model` submodule and needs the RAW, un-stripped key convention
for vLLM's own hf_to_vllm_mapper to find it -- confirmed empirically
during TASK M-exec: stripping RedHatAI's checkpoint before grafting
produced a checkpoint vLLM could not load at all
(`ValueError: There is no module or parameter named 'model' in
Qwen3_5ForConditionalGeneration`), and separately verified that
RedHatAI's own `ignore` list already uses the raw
"model.language_model.layers.*" convention (compressed-tensors config
matching operates on the checkpoint's OWN raw key names, not vLLM's
internal post-mapper module path) -- so grafting against the raw
convention keeps config.json internally consistent too, not just the
weights.

Runs entirely on CPU (numpy/torch/gguf/safetensors), no GPU and no vLLM
install required -- verification that vLLM actually loads and serves this
checkpoint is out of scope for this box; see the Colab runbook this script
prints at the end of a successful run.
"""

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import gguf
from gguf.quants import dequantize as gguf_dequantize
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gguf2marlin", SCRIPTS_DIR / "gguf2marlin.py")
g2m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g2m)

MARLIN_SUPPORTED_GROUP_SIZES = (32, 64, 128)
BITS = 8

# HF suffix -> ggml canonical tensor-name suffix, resolved from the
# INSTALLED gguf package's own MODEL_TENSORS[MODEL_ARCH.QWEN35] table (not
# guessed): in_proj_qkv -> MODEL_TENSOR.ATTN_QKV -> "blk.{bid}.attn_qkv",
# in_proj_z -> MODEL_TENSOR.ATTN_GATE -> "blk.{bid}.attn_gate",
# in_proj_b -> MODEL_TENSOR.SSM_BETA -> "blk.{bid}.ssm_beta",
# in_proj_a -> MODEL_TENSOR.SSM_ALPHA -> "blk.{bid}.ssm_alpha".
# gguf2marlin.py's generic architecture-name mapper deliberately can't be
# reused for these four (its own LIMITATIONS section documents that
# in_proj_qkv/in_proj_b collide ambiguously with a shorter-named generic
# convention from a different architecture) -- hardcoded here instead,
# same approach scripts/transcode_gguf_to_gptq.py already takes for
# Qwen3.5's other GDN tensors.
GGML_SUFFIX_FOR_HF_SUFFIX = {
    "in_proj_qkv": "attn_qkv",
    "in_proj_z": "attn_gate",
    "in_proj_b": "ssm_beta",
    "in_proj_a": "ssm_alpha",
}
GDN_SUFFIXES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
MERGE_PAIRS = (("in_proj_qkv", "in_proj_z"), ("in_proj_b", "in_proj_a"))
# vLLM's own MergedColumnParallelLinear fused parameter name for each HF
# suffix (see qwen3_5.py's hf_to_vllm_mapper.orig_to_new_stacked table,
# quoted in DECISION 4 below) -- config_groups' `targets` MUST use these
# FUSED names, not the four individual unfused ones, or vLLM's own
# compressed-tensors `find_matched_target` will never see them: it tries
# (1) exact/regex match of the fused layer_name against `targets`, then
# (2) a *substring* match of the module's class name (e.g.
# "MergedColumnParallelLinear") against `targets`, and ONLY THEN (3) the
# fused-vs-unfused-components reconciliation that would actually resolve
# individual "in_proj_b"/"in_proj_a" entries against the fused "in_proj_ba"
# layer. Step (2) matches ANY Linear-family module against a bare "Linear"
# target via substring containment, so a champion's default catch-all
# config_group (`targets: ["Linear"]`) always wins step (2) before step
# (3) -- which holds the fused-component logic -- ever runs. (Confirmed
# empirically during TASK M-exec: grafting with unfused per-projection
# targets produced a checkpoint where vLLM's Marlin loader tried to merge
# an int8-packed shard into a param sized for the champion's *int4* g128
# default scheme, RuntimeError "cannot merge quantized shard 1 ...
# different num_bits/group_size" -- root-caused by instrumenting
# find_matched_target/_match_fused_layer directly, not guessed.)
FUSED_NAME_FOR_HF_SUFFIX = {
    "in_proj_qkv": "in_proj_qkvz",
    "in_proj_z": "in_proj_qkvz",
    "in_proj_b": "in_proj_ba",
    "in_proj_a": "in_proj_ba",
}
GRAFT_GROUP_NAME = "graft_gdn_int8"


class GraftError(RuntimeError):
    pass


class GraftConfigError(GraftError):
    pass


# ==========================================================================
# 1. V-head "tile" transform (numpy port of patch_gguf_qwen35_transforms.py)
# ==========================================================================


# TASK K4: gguf2marlin.py now needs this exact same V-head untile math for
# its own generic GDN transcode path (A_log/dt_bias/conv1d/out_proj too,
# not just these four in_proj_* tensors), so the implementation moved
# there and is now the single shared copy -- this script already imports
# gguf2marlin.py as `g2m` (see above, for quantize_symmetric_group/
# pack_rows), so these are thin delegating wrappers rather than an
# independent copy of the same math. (The reverse import direction isn't
# possible: gguf2marlin.py can't import this file back, that would be a
# real circular import since this script loads gguf2marlin.py via
# importlib at module scope.) Kept under their original names/signatures
# here since test_graft_gguf_gdn.py calls them directly.
def untile_v_heads(t: np.ndarray, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> np.ndarray:
    """Inverse of llama.cpp's grouped -> tiled V-head permutation. See
    gguf2marlin.py's `qwen35_untile_v_heads` (this delegates to it)."""
    return g2m.qwen35_untile_v_heads(t, dim, num_k_heads, num_v_per_k, head_dim)


def untile_module(suffix: str, w: np.ndarray, *, reorder: bool, num_k: int, num_v_per_k: int,
                   head_k: int, head_v: int) -> np.ndarray:
    """Undo the tile transform for one of the four GDN in_proj_* tensors,
    given as a (out_features, in_features) HF-orientation array (see
    `dequant_ggml_tensor` for why that orientation is guaranteed). See
    gguf2marlin.py's `qwen35_untile_module` (this delegates to it); A_log/
    dt_bias/conv1d/out_proj remain out of scope for this script (it only
    grafts the four in_proj_* input projections, see the module docstring)."""
    return g2m.qwen35_untile_module(suffix, w, reorder=reorder, num_k=num_k, num_v_per_k=num_v_per_k,
                                     head_k=head_k, head_v=head_v)


# ==========================================================================
# 2. Merge-pair scheme validation (fail loudly, unlike gguf2marlin's
#    force-to-fp16 -- see scripts/patch_vllm_gdn_quant_load.py).
# ==========================================================================


def validate_merge_pairs(grafted: dict[tuple[int, str], bool]) -> None:
    """grafted: {(layer_idx, suffix): bool} for every (layer, GDN suffix)
    this script considered. Raises GraftConfigError if, for any layer, one
    member of a KNOWN merge pair (in_proj_qkv/in_proj_z or in_proj_b/
    in_proj_a) was grafted (-> int8) while its partner was not (-> stays
    fp16) -- vLLM's merged-column loader can only concatenate shards that
    share one on-disk scheme (patch_vllm_gdn_quant_load.py)."""
    layers = sorted({layer for layer, _ in grafted})
    for layer in layers:
        for a, b in MERGE_PAIRS:
            key_a, key_b = (layer, a), (layer, b)
            if key_a not in grafted or key_b not in grafted:
                continue
            if grafted[key_a] != grafted[key_b]:
                raise GraftConfigError(
                    f"layer {layer}: merge pair (in_proj_{a.split('_')[-1]}, "
                    f"in_proj_{b.split('_')[-1]}) has mismatched grafted status "
                    f"(in_proj_{a[len('in_proj_'):]}={grafted[key_a]!r} vs "
                    f"in_proj_{b[len('in_proj_'):]}={grafted[key_b]!r}). "
                    "vLLM's MergedColumnParallelLinear loader requires every "
                    "shard folded into one fused parameter to share the exact "
                    "same on-disk scheme (see scripts/patch_vllm_gdn_quant_load.py) "
                    "-- both members of this pair must be grafted together, or "
                    "neither."
                )


# ==========================================================================
# 3. int8 encode of one GDN tensor, reusing gguf2marlin's RTN + packing.
# ==========================================================================


def dequant_ggml_tensor(reader_tensor) -> np.ndarray:
    """GGUFReader tensor -> fp32 numpy array in HF Linear-weight orientation
    (out_features, in_features). GGUFReader.tensors' `.data` is already
    reshaped to this orientation (gguf_reader.py:359-367 reshapes the raw
    bytes with `np_dims = tuple(reversed(dims))`, i.e. row-major numpy shape
    reversed from ggml's own ne=[in_features, out_features] convention),
    matching gguf2marlin.py's own `out_features, in_features = t.shape[-1],
    t.shape[0]` convention -- verified by cross-reading gguf_reader.py
    directly, not assumed."""
    gtype = gguf.GGMLQuantizationType(reader_tensor.tensor_type)
    return np.ascontiguousarray(gguf_dequantize(reader_tensor.data, gtype), dtype=np.float32)


def encode_int8_module(w_out_in: np.ndarray, group_size: int):
    """(out_features, in_features) fp32 -> compressed-tensors on-disk int8
    tensors, via gguf2marlin's vetted RTN quantizer (see DECISION 1 above
    for the transpose derivation).

    Returns (weight_packed (N, K_packed) int32, weight_scale (N, num_groups)
    float32 -- cast to the frame's own dtype by the caller, see
    `resolve_frame_dtype` --, weight_shape (2,) int64 [N, K], rel_rms_error
    float).
    """
    out_features, in_features = w_out_in.shape
    w_kn = np.ascontiguousarray(w_out_in.T)  # (K, N) -- gguf2marlin's convention
    qweight, scales, w_ref = g2m.quantize_symmetric_group(w_kn, group_size, bits=BITS)

    weight_packed = np.ascontiguousarray(qweight.T)  # (N, K_packed)
    weight_scale = np.ascontiguousarray(scales.T)  # (N, num_groups)
    weight_shape = np.array([out_features, in_features], dtype=np.int64)

    denom = float(np.sqrt((w_kn.astype(np.float64) ** 2).mean())) or 1.0
    rel_rms = float(np.sqrt(((w_ref.astype(np.float64) - w_kn.astype(np.float64)) ** 2).mean())) / denom
    return weight_packed, weight_scale, weight_shape, rel_rms


# ==========================================================================
# 3b. Frame dtype resolution -- weight_scale must match the frame's OWN
#     on-disk dtype, not a hardcoded assumption.
# ==========================================================================

_DTYPE_STRING_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_frame_dtype(cfg: dict, text_cfg: dict) -> torch.dtype:
    """The grafted weight_scale tensor must match the frame's own on-disk
    dtype: CompressedTensorsWNA16.create_weights allocates
    GroupQuantScaleParameter as `torch.empty(..., dtype=params_dtype)`,
    where params_dtype is the model's configured dtype -- NOT a fixed fp16
    assumption. An earlier version of this script hardcoded
    `weight_scale.astype(np.float16)` here; checked directly against
    RedHatAI/Qwen3.5-9B-quantized.w4a16 (the actual --frame this script
    targets), that frame's config.json declares `"dtype": "bfloat16"`
    (root and text_config -- modern HF schema; older checkpoints may use
    `"torch_dtype"` instead) and its own weight_scale tensors are BF16 on
    disk (verified via safe_open against several mlp.*.weight_scale keys) --
    a hardcoded fp16 graft would have written a dtype-mismatched tensor into
    that checkpoint.

    Checks, in order: cfg['dtype'], cfg['torch_dtype'], text_cfg['dtype'],
    text_cfg['torch_dtype']. Falls back to float16 with a loud warning only
    if none of those keys are present at all (e.g. this repo's own
    synthetic test fixture in test_graft_gguf_gdn.py, which predates
    dtype-aware frames and only ever exercises fp16) -- this function never
    silently guesses for a frame that actually declares a dtype.
    """
    for source in (cfg, text_cfg):
        for key in ("dtype", "torch_dtype"):
            val = source.get(key)
            if val:
                dt = _DTYPE_STRING_MAP.get(str(val).lower())
                if dt is None:
                    raise GraftError(
                        f"--frame's config.json declares dtype={val!r} which this "
                        f"script doesn't recognize (known: {sorted(_DTYPE_STRING_MAP)}) "
                        "-- refusing to guess a weight_scale dtype."
                    )
                return dt
    print(
        "[graft_gguf_gdn] WARNING: --frame's config.json has no dtype/torch_dtype "
        "field (checked cfg root and cfg['text_config']) -- defaulting "
        "weight_scale to float16. This is only correct if the frame's OTHER "
        "weight_scale tensors are also fp16; verify against the frame's own "
        "safetensors (safe_open + get_slice(...).get_dtype()) before trusting "
        "this checkpoint.",
        file=sys.stderr,
    )
    return torch.float16


# ==========================================================================
# 4. config.json surgery -- narrow `ignore`, add one config_group.
# ==========================================================================


def _is_equal_or_regex_match(value: str, target: str) -> bool:
    """Verbatim semantics of vllm's compressed_tensors/utils.py
    `_is_equal_or_regex_match` (exact string, or `re.match` if the target
    starts with "re:") -- reimplemented here (not imported) since vLLM/
    compressed_tensors are not installed on this box; see DECISION 4."""
    if target.startswith("re:"):
        return re.match(target[3:], value) is not None
    return target == value


def narrow_ignore_list(ignore_list: list[str], grafted_names: list[str], probe_names: list[str]) -> list[str]:
    """Remove `grafted_names` from whichever `ignore_list` entries currently
    cover them, verifying (against `probe_names`, things that must STAY
    ignored) that nothing else is accidentally un-ignored. See DECISION 4."""
    if not grafted_names:
        return list(ignore_list)
    new_ignore = []
    for entry in ignore_list:
        matched = [n for n in grafted_names if _is_equal_or_regex_match(n, entry)]
        if not matched:
            new_ignore.append(entry)
            continue
        if not entry.startswith("re:"):
            if entry not in grafted_names:
                raise GraftConfigError(
                    f"ignore entry {entry!r} matched a grafted module via exact "
                    f"string equality but is not itself one of the grafted names "
                    f"{grafted_names!r} -- refusing to guess how to narrow it."
                )
            continue  # literal entry fully consumed by the graft -- drop it.

        pattern = entry[3:]
        excl = "|".join(re.escape(n) for n in matched)
        new_entry = f"re:(?!(?:{excl})$)(?:{pattern})"

        for n in matched:
            if _is_equal_or_regex_match(n, new_entry):
                raise GraftConfigError(
                    f"failed to narrow ignore pattern {entry!r}: grafted module "
                    f"{n!r} still matches the rewritten pattern {new_entry!r}."
                )
        for n in probe_names:
            was_ignored = _is_equal_or_regex_match(n, entry)
            now_ignored = _is_equal_or_regex_match(n, new_entry)
            if was_ignored and not now_ignored:
                raise GraftConfigError(
                    f"narrowing ignore pattern {entry!r} -> {new_entry!r} would "
                    f"also un-ignore unrelated module {n!r} -- refusing to write "
                    f"a config.json broader than intended."
                )
        new_ignore.append(new_entry)
    return new_ignore


def build_probe_names(num_hidden_layers: int, grafted_layers: set[int], module_prefix: str) -> list[str]:
    """Module names that must remain covered by whatever ignore rule
    previously covered the GDN block, used to catch an overly-broad
    narrowing (see `narrow_ignore_list`): the GDN submodules this script
    does NOT touch (conv1d/norm/out_proj/A_log/dt_bias) on every layer that
    has them, in_proj_* on layers this run did NOT graft, and a couple of
    definitely-unrelated modules (lm_head, a self_attn projection)."""
    probes = ["lm_head", f"{module_prefix}layers.0.self_attn.q_proj"]
    other_gdn_submodules = ("conv1d", "norm", "out_proj", "A_log", "dt_bias")
    for layer in range(num_hidden_layers):
        for sub in other_gdn_submodules:
            probes.append(f"{module_prefix}layers.{layer}.linear_attn.{sub}")
        if layer not in grafted_layers:
            for suffix in GDN_SUFFIXES:
                probes.append(f"{module_prefix}layers.{layer}.linear_attn.{suffix}")
    return probes


def build_new_config_group(quant_cfg: dict, group_size: int) -> dict:
    """Clone an existing config_group's `weights` QuantizationArgs template
    from the frame's own config.json (rather than hand-writing the full
    compressed-tensors QuantizationArgs pydantic schema -- extra fields
    like `type`/`observer`/`dynamic`/`actorder` are easy to get subtly
    wrong by hand and this way inherits the champion's own conventions),
    then overrides only num_bits/group_size/strategy/symmetric."""
    config_groups = quant_cfg.get("config_groups") or {}
    if not config_groups:
        raise GraftConfigError(
            "--frame's quantization_config has no config_groups to use as a "
            "template -- is this really a compressed-tensors checkpoint?"
        )
    template_group = next(iter(config_groups.values()))
    template_weights = template_group.get("weights")
    if not template_weights:
        raise GraftConfigError(
            f"config_group {next(iter(config_groups))!r} has no 'weights' "
            "QuantizationArgs to use as a template."
        )
    weights = dict(template_weights)
    weights["num_bits"] = BITS
    weights["group_size"] = group_size
    weights["strategy"] = "group"
    weights["symmetric"] = True
    weights["type"] = weights.get("type", "int")
    return {"targets": [], "weights": weights}


# ==========================================================================
# 5. Frame (champion checkpoint) I/O helpers.
# ==========================================================================


def load_frame_weight_map(frame_dir: Path) -> dict[str, str]:
    """tensor key -> shard filename, for either a sharded (*.safetensors +
    model.safetensors.index.json) or single-file (model.safetensors) frame."""
    index_path = frame_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return dict(index["weight_map"])
    single = frame_dir / "model.safetensors"
    if not single.exists():
        raise GraftError(f"neither {index_path} nor {single} exists under --frame")
    with safe_open(single, framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


_STRIPPED_QKV_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$")
_RAW_QKV_PATTERN = re.compile(r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$")


def detect_module_prefix(weight_map: dict[str, str]) -> str:
    """Detect whether this frame's GDN modules live under "model.layers.*"
    or "model.language_model.layers.*", and return the corresponding
    prefix (always ending in a literal "." so callers can just do
    f"{prefix}layers.{{N}}...").

    Two real, DIFFERENT conventions exist among this project's own champion
    candidates, and picking the wrong one produces a checkpoint that
    silently fails to serve:

    - "model.layers.*" -- this project's own quantize_*.py outputs (saved
      via plain AutoModelForCausalLM, no vision tower) after
      fix_qwen35_hf_checkpoint.py strips the "language_model." segment so
      vLLM's TEXT-ONLY Qwen3_5ForCausalLM class (which has no
      `language_model` submodule) can load them.
    - "model.language_model.layers.*" -- a genuinely multimodal-capable
      checkpoint (RedHatAI/Qwen3.5-9B-quantized.w4a16 included: it ships a
      real vision tower, "visual.*" weights) that vLLM loads through
      Qwen3_5ForConditionalGeneration, which DOES have a `language_model`
      submodule and expects the RAW, un-stripped key convention -- its own
      hf_to_vllm_mapper reorders "model.language_model.*" (checkpoint) into
      "language_model.model.*" (vLLM's internal module path) itself.
      Confirmed empirically during TASK M-exec: RedHatAI serves
      successfully completely unmodified straight from the Hub (no fixer
      ever applied, this whole campaign); pre-stripping it before grafting
      produced `ValueError: There is no module or parameter named 'model'
      in Qwen3_5ForConditionalGeneration` at serve time -- the mapper's own
      pattern match needs the RAW prefix to fire.

    Never guesses when the frame doesn't unambiguously match exactly one of
    the two -- raises GraftError instead.
    """
    has_stripped = any(_STRIPPED_QKV_PATTERN.match(k) for k in weight_map)
    has_raw = any(_RAW_QKV_PATTERN.match(k) for k in weight_map)
    if has_stripped and has_raw:
        raise GraftError(
            "--frame's weight_map matches BOTH the stripped ('model.layers.*') "
            "and raw ('model.language_model.layers.*') GDN naming conventions "
            "-- ambiguous, refusing to guess which one this frame actually "
            "serves through."
        )
    if has_stripped:
        return "model."
    if has_raw:
        return "model.language_model."
    raise GraftError(
        "--frame has no GDN in_proj_qkv weights matching either "
        "'model.layers.*' or 'model.language_model.layers.*' -- cannot "
        "determine this frame's module-name convention."
    )


def find_gdn_layers(weight_map: dict[str, str], module_prefix: str) -> set[int]:
    """Layer indices that have an (fp16, ignore-listed) in_proj_qkv weight
    in the frame -- i.e. GDN mixer layers, as opposed to full-attention
    layers (Qwen3.5-9B is a hybrid architecture, ~75% GDN / 25% full
    attention per this project's own measurements)."""
    pat = re.compile(rf"^{re.escape(module_prefix)}layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$")
    layers = set()
    for key in weight_map:
        m = pat.match(key)
        if m:
            layers.add(int(m.group(1)))
    return layers


# ==========================================================================
# 6. Main pipeline.
# ==========================================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", required=True, help=(
        "dir of the champion checkpoint to graft into -- either convention "
        "works (auto-detected, see detect_module_prefix): a text-only "
        "checkpoint already run through fix_qwen35_hf_checkpoint.py "
        "('model.layers.*'), or a genuinely multimodal-capable checkpoint "
        "like RedHatAI/Qwen3.5-9B-quantized.w4a16 used completely "
        "unmodified, straight from the Hub ('model.language_model.layers.*'"
        ") -- do NOT run fix_qwen35_hf_checkpoint.py on the latter kind, it "
        "breaks vLLM's ConditionalGeneration loading path for them"
    ))
    ap.add_argument("--gguf", required=True, help="path to the source .gguf file (e.g. Q4_K_M)")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--group-size", type=int, default=32, choices=MARLIN_SUPPORTED_GROUP_SIZES,
                     help="int8 group size for the grafted GDN modules (default 32 -- see DECISION 2)")
    args = ap.parse_args(argv)

    frame_dir = Path(args.frame)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_size = args.group_size

    cfg_path = frame_dir / "config.json"
    if not cfg_path.exists():
        raise GraftError(f"{cfg_path} not found -- is --frame a valid checkpoint dir?")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    quant_cfg = cfg.get("quantization_config")
    if not quant_cfg or "config_groups" not in quant_cfg:
        raise GraftConfigError(
            "--frame's config.json has no quantization_config.config_groups -- "
            "this script grafts INTO an already-quantized compressed-tensors "
            "checkpoint (the champion), it does not create one from scratch."
        )
    ignore_list = list(quant_cfg.get("ignore", []))

    text_cfg = cfg.get("text_config", cfg)
    head_fields = ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
    missing = [f for f in head_fields if text_cfg.get(f) is None]
    if missing:
        raise GraftError(
            f"--frame's config.json is missing GDN head-shape field(s) {missing} "
            f"(checked cfg root and cfg['text_config']) -- cannot determine "
            f"whether the GGUF's V-head 'tile' transform needs undoing "
            f"(see DECISION 3); refusing to guess."
        )
    num_k = int(text_cfg["linear_num_key_heads"])
    num_v = int(text_cfg["linear_num_value_heads"])
    head_k = int(text_cfg["linear_key_head_dim"])
    head_v = int(text_cfg["linear_value_head_dim"])
    reorder = num_k > 0 and num_v > 0 and num_k != num_v
    num_v_per_k = (num_v // num_k) if num_k else 1
    num_hidden_layers = int(text_cfg.get("num_hidden_layers", 0))
    print(f"[graft_gguf_gdn] GDN head shape: num_k={num_k} num_v={num_v} head_k={head_k} "
          f"head_v={head_v} reorder={reorder}", file=sys.stderr)

    frame_dtype = resolve_frame_dtype(cfg, text_cfg)
    print(f"[graft_gguf_gdn] frame weight_scale dtype: {frame_dtype}", file=sys.stderr)

    weight_map = load_frame_weight_map(frame_dir)
    module_prefix = detect_module_prefix(weight_map)
    print(f"[graft_gguf_gdn] frame module-name convention: {module_prefix!r}"
          + (" (raw, un-stripped -- multimodal-capable frame)" if module_prefix != "model." else " (stripped)"),
          file=sys.stderr)
    gdn_layers = find_gdn_layers(weight_map, module_prefix)
    if not gdn_layers:
        raise GraftError(
            f"no {module_prefix}layers.N.linear_attn.in_proj_qkv.weight keys found in --frame; nothing to graft"
        )
    print(f"[graft_gguf_gdn] frame has {len(gdn_layers)} GDN layers: {sorted(gdn_layers)}", file=sys.stderr)

    print(f"[graft_gguf_gdn] reading {args.gguf}", file=sys.stderr)
    reader = gguf.GGUFReader(args.gguf)
    ggml_tensors = {t.name: t for t in reader.tensors}

    # ---- pass 1: decide, per (layer, suffix), whether it can be grafted ----
    grafted: dict[tuple[int, str], bool] = {}
    hf_module_name = {}  # (layer, suffix) -> f"{module_prefix}layers.N.linear_attn.in_proj_X"
    ggml_name = {}
    for layer in sorted(gdn_layers):
        for suffix in GDN_SUFFIXES:
            module = f"{module_prefix}layers.{layer}.linear_attn.{suffix}"
            hf_module_name[(layer, suffix)] = module
            gname = f"blk.{layer}.{GGML_SUFFIX_FOR_HF_SUFFIX[suffix]}.weight"
            ggml_name[(layer, suffix)] = gname
            in_frame = (module + ".weight") in weight_map
            in_gguf = gname in ggml_tensors
            if in_frame and not in_gguf:
                print(f"[graft_gguf_gdn] WARNING: {module} present in frame but "
                      f"{gname!r} not found in GGUF -- leaving fp16", file=sys.stderr)
            grafted[(layer, suffix)] = in_frame and in_gguf

    validate_merge_pairs(grafted)

    grafted_names = [hf_module_name[k] for k, v in grafted.items() if v]
    grafted_layers = {layer for (layer, _), v in grafted.items() if v}
    kept_names = [hf_module_name[k] for k, v in grafted.items() if not v]

    # ---- pass 2: quantize each grafted tensor -------------------------------
    grafted_tensors: dict[str, torch.Tensor] = {}
    error_report = []  # (module, rel_rms)
    for (layer, suffix), do_graft in grafted.items():
        if not do_graft:
            continue
        module = hf_module_name[(layer, suffix)]
        w = dequant_ggml_tensor(ggml_tensors[ggml_name[(layer, suffix)]])
        w = untile_module(suffix, w, reorder=reorder, num_k=num_k, num_v_per_k=num_v_per_k,
                           head_k=head_k, head_v=head_v)

        with safe_open(frame_dir / weight_map[module + ".weight"], framework="pt") as f:
            frame_shape = tuple(f.get_slice(module + ".weight").get_shape())
        if tuple(w.shape) != frame_shape:
            raise GraftError(
                f"{module}: GGUF-derived shape {w.shape} (after untile) does not "
                f"match the frame's own weight shape {frame_shape} -- head-shape "
                f"config or the V-head untile logic is wrong; refusing to graft."
            )

        weight_packed, weight_scale, weight_shape, rel_rms = encode_int8_module(w, group_size)
        grafted_tensors[module + ".weight_packed"] = torch.from_numpy(weight_packed)
        grafted_tensors[module + ".weight_scale"] = torch.from_numpy(
            weight_scale.astype(np.float32)
        ).to(frame_dtype)
        grafted_tensors[module + ".weight_shape"] = torch.from_numpy(weight_shape)
        error_report.append((module, rel_rms))

    # ---- pass 3: config.json surgery -----------------------------------------
    probe_names = build_probe_names(num_hidden_layers or (max(gdn_layers) + 1), grafted_layers, module_prefix)
    # `ignore` narrowing keeps using the four UNFUSED per-projection names --
    # should_ignore_layer's own fused-component reconciliation (see that
    # function's source) already handles resolving them against the fused
    # layer correctly, unconditionally. `config_groups[*].targets` is
    # different: it MUST use the FUSED parameter names (see
    # FUSED_NAME_FOR_HF_SUFFIX's comment for why) or vLLM's
    # find_matched_target will never route these layers to this scheme.
    new_ignore = narrow_ignore_list(ignore_list, grafted_names, probe_names)
    new_group = build_new_config_group(quant_cfg, group_size)
    fused_target_names = sorted({
        f"{module_prefix}layers.{layer}.linear_attn.{FUSED_NAME_FOR_HF_SUFFIX[suffix]}"
        for (layer, suffix), do_graft in grafted.items()
        if do_graft
    })
    new_group["targets"] = fused_target_names

    new_quant_cfg = dict(quant_cfg)
    new_quant_cfg["ignore"] = new_ignore
    new_config_groups = dict(quant_cfg["config_groups"])
    if grafted_names:
        new_config_groups[GRAFT_GROUP_NAME] = new_group
    new_quant_cfg["config_groups"] = new_config_groups

    new_cfg = dict(cfg)
    new_cfg["quantization_config"] = new_quant_cfg
    (out_dir / "config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- pass 4: stream-copy every other tensor, shard by shard -------------
    grafted_original_keys = {hf_module_name[k] + ".weight" for k, v in grafted.items() if v}
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)

    index_path = frame_dir / "model.safetensors.index.json"
    is_sharded = index_path.exists()
    new_weight_map: dict[str, str] = {}
    for shard_name, keys in shards.items():
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(frame_dir / shard_name, framework="pt") as f:
            for key in keys:
                if key in grafted_original_keys:
                    continue  # replaced by weight_packed/weight_scale/weight_shape below
                out_tensors[key] = f.get_tensor(key)
        for module in {hf_module_name[k] for k, v in grafted.items() if v and weight_map[hf_module_name[k] + ".weight"] == shard_name}:
            for suffix2 in ("weight_packed", "weight_scale", "weight_shape"):
                out_tensors[f"{module}.{suffix2}"] = grafted_tensors[f"{module}.{suffix2}"]
        save_file(out_tensors, out_dir / shard_name, metadata={"format": "pt"})
        for key in out_tensors:
            new_weight_map[key] = shard_name

    if is_sharded:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"] = new_weight_map
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ---- pass 5: copy every other file verbatim (tokenizer, chat template) --
    skip_names = {"config.json", "model.safetensors.index.json"} | set(shards.keys())
    for path in frame_dir.iterdir():
        if path.name in skip_names or path.is_dir():
            continue
        shutil.copy2(path, out_dir / path.name)

    # ---- report ----------------------------------------------------------
    manifest = {
        "source_gguf": Path(args.gguf).name,
        "group_size": group_size,
        "bits": BITS,
        "grafted_modules": sorted(grafted_names),
        "kept_fp16_modules": sorted(kept_names),
        "tensors": [{"module": m, "rel_rms_error": e} for m, e in error_report],
    }
    (out_dir / "graft_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_report(out_dir, grafted_names, kept_names, error_report)
    return 0


def _print_report(out_dir, grafted_names, kept_names, error_report):
    print("\n" + "=" * 72)
    print("GRAFT REPORT")
    print("=" * 72)
    print(f"grafted to int8 (compressed-tensors, GGUF-sourced): {len(grafted_names)}")
    for name in sorted(grafted_names):
        print(f"    {name}")
    print(f"kept fp16 (unchanged from frame): {len(kept_names)}")
    for name in sorted(kept_names):
        print(f"    {name}")
    if error_report:
        errs = np.array([e for _, e in error_report])
        print(f"\nper-tensor relative RMS error, dequant(GGUF original) vs "
              f"dequant(grafted int8): n={len(errs)} mean={errs.mean():.6f} "
              f"max={errs.max():.6f} min={errs.min():.6f} (target ~0.005)")
        for name, e in sorted(error_report, key=lambda x: -x[1])[:8]:
            print(f"    {e:.6f}  {name}")
    print(f"\nwrote {out_dir}")
    print("\nSuggested Colab acceptance run (GPU, not run from this box):")
    print("    python scripts/fix_qwen35_hf_checkpoint.py <out_dir>  # if not already fixed")
    print("    vllm serve <out_dir> --max-model-len 8192")
    print("    # fast-bench conc1 + conc32, eval ppl on the 99-prompt SWE-bench set,")
    print("    # compare against the champion's baseline ppl=5.1578 (target ratio < 1.10)")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
