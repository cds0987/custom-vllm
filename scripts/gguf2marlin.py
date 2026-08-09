#!/usr/bin/env python3
"""
Offline GGUF -> GPTQ-Marlin safetensors transcoder (architecture-generic).

GOAL
----
Take a GGUF checkpoint quantized with llama.cpp's Q4_0 / Q4_1 / Q4_K(_M) and
rewrite it as a plain HF-style safetensors checkpoint that vLLM 0.26 loads
through its *AutoGPTQ* checkpoint format and serves with the Marlin kernel --
without leaving ggml's block layout, and without a calibration pass. This is
RTN (round-to-nearest) re-packing of int4 values that are, in the Q4_0 case,
bit-identical to what a real GPTQ run would have produced; see "ERROR BUDGET"
below.

DECISION 1 -- OUTPUT FORMAT: AutoGPTQ ("gptq"/"gptq_marlin"), not
compressed-tensors
------------------------------------------------------------------
Read directly from the local vLLM checkout at
D:\\Training\\AI_Module\\vllm\\vllm\\vllm (0.26-era source; the class names
below do NOT match the upstream "gptq_marlin.py" some vLLM versions ship --
this checkout refactored that file away):

  vllm/model_executor/layers/quantization/auto_gptq.py
      AutoGPTQConfig / AutoGPTQLinearMethod -- config parsing, per-layer
      Parameter registration (create_weights, ~line 381).
  vllm/model_executor/layers/quantization/utils/quant_utils.py
      quantize_weights() / pack_rows() / pack_cols() / unpack_cols() -- the
      *exact* reference bit-packing math a GPTQ checkpoint must match on
      disk. This module has zero vLLM-internal dependencies (only torch),
      so its packing convention is reproduced verbatim below in numpy.
  vllm/model_executor/kernels/linear/mixed_precision/marlin.py
      MarlinLinearKernel.process_weights_after_loading -- calls
      ops.gptq_marlin_repack() on the *standard GPTQ* on-disk tensors to
      build Marlin's private tile-interleaved layout at load time. We do
      NOT need to reproduce Marlin's internal layout -- only a correct
      GPTQ checkpoint, which vLLM repacks for us.
  vllm/model_executor/layers/quantization/utils/marlin_utils.py
      MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128] (line ~35) -- so
      group_size=32 (this script's default, per the task) is valid.

Per-Linear tensor contract (auto_gptq.py:381-445, quant_utils.py:610-845),
pack_factor = 32 // bits = 8 for 4-bit:

    qweight  int32  (in_features // pack_factor, out_features)
    qzeros   int32  (in_features // group_size, out_features // pack_factor)
    scales   fp16   (in_features // group_size, out_features)
    g_idx    int32  (in_features,)

qweight packing (pack_rows, quant_utils.py:758-779) packs along dim 0 (K):
    for i in range(pack_factor):
        packed |= q[i::pack_factor, :] << (bits * i)
i.e. row i (0..7) of each 8-row group occupies the i-th nibble (bits
[4i:4i+4)) of the int32 -- row 0 -> lowest nibble. Stored values are the
*biased* uint4b8 code: q_biased = round(w/scale) + 8, clamped to [0, 15]
(scalar_types.uint4b8, bias=8 -- quant_utils.py quantize_weights()).

qzeros: AutoGPTQLinearMethod.create_weights passes zero_points=False
unconditionally (auto_gptq.py:350), and for zero_points=False,
MarlinLinearKernel.process_weights_after_loading *discards the loaded
qzeros tensor entirely* and substitutes an empty one (marlin.py:208-209) --
its content is never read at inference time for a symmetric checkpoint.
Only its shape must be right so the weight loader doesn't choke. We still
fill it with the standard symmetric encoding (every nibble = 8, i.e. every
int32 word = 0x88888888) so the checkpoint is well-formed for any other
GPTQ-compatible tool that DOES read qzeros.

g_idx: always registered and loaded (auto_gptq.py:395-402); with
desc_act=False it is discarded at repack time (marlin.py:172-180) but must
still be present on disk. We write the trivial (unpermuted) g_idx = i //
group_size, matching "RTN has no activation order".

config.json quantization_config (auto_gptq.py:194-238, TYPE_MAP at
line ~101): required flat fields quant_method="gptq", bits, group_size,
sym, desc_act. TYPE_MAP only supports (bits=4, sym=True) and (bits=8,
sym=True) -- asymmetric GPTQ raises ValueError, so this script only emits
sym=True. Per-module selection is an ALLOWLIST, "modules_in_block_to_
quantize" (default None -> auto-detected from which safetensors keys carry
a .qweight suffix, auto_gptq.py:278-303's maybe_update_config) -- we rely
on that auto-detection rather than writing the list ourselves, matching
the convention already used by scripts/transcode_gguf_to_gptq.py.

Why not compressed-tensors (weight_packed/weight_scale/weight_zero_point)?
Its bit-packing is defined by the external `compressed_tensors` pip
package, not by anything in this vLLM checkout, so its on-disk contract
can't be pinned down by reading local source alone. AutoGPTQ's contract is
fully specified inside this repo (quant_utils.py) and has ~half the config
surface (4 flat scalars vs compressed-tensors' config_groups/QuantizationArgs
schema). Given the goal is "generate a correct checkpoint offline with no
calibration and no vLLM install on this box", AutoGPTQ is the safer target.

DECISION 2 -- GGUF BLOCK LAYOUTS (from the installed `gguf==0.19.0`
package; gguf/quants.py + gguf/constants.py; empirically cross-checked
below against gguf.quants.quantize()/dequantize() as an independent
reference, not just read from source)
-----------------------------------------------------------------------
Q4_0 (block=32 weights, 18 bytes: 2-byte fp16 `d` + 16 packed bytes):
    byte b (0..15) holds element b in its LOW nibble and element b+16 in
    its HIGH nibble (halves, not interleaved pairs).
    dequant: value = d * (nibble - 8)          -- symmetric, zero=8 fixed.
This is *exactly* GPTQ's uint4b8 symmetric encoding with the same zero
point. When --group-size is 32 (Q4_0's native block size), the on-disk
nibble IS the GPTQ-biased code and `d` IS the GPTQ per-group scale --
q_biased = nibble, scale = d, transcoded with ZERO float rounding. This
script exploits that: Q4_0 tensors take a byte-exact fast path
(`unpack_q4_0_exact`) instead of dequant+requant, which is how "Q4_0 rel.
RMS error must be exactly 0" is *guaranteed*, not just empirically likely.
Verified against gguf.quants.quantize()/dequantize() (independent
reference implementation) for both single- and multi-row synthetic
tensors before relying on it here -- max abs diff was 0.0 in every trial.

Q4_1 (block=32, 20 bytes: fp16 `d` + fp16 `m` + 16 packed bytes, same
nibble halves as Q4_0): dequant: value = d * nibble + m -- asymmetric (a
per-block *min*, not a symmetric zero=8). There is no group_size=32
symmetric-zero=8 encoding that reproduces an arbitrary affine (d, m) grid
exactly, so Q4_1 goes through the generic path: dequantize with gguf's own
dequantize() (independent, vetted implementation), then re-quantize
per-group(=group_size) into symmetric zero=8 GPTQ codes.

Q4_K (super-block=256 = 8 sub-blocks of 32, 144 bytes: fp16 `d`, fp16
`dmin`, 12-byte packed 6-bit sub-scale/sub-min pairs, 128 bytes qs):
per-element dequant value = (d*sc[j])*nibble - (dmin*m[j]), j = (index
within super-block) // 32. The implied zero-point (dmin*m[j])/(d*sc[j]) is
not integral in general, so Q4_K cannot be bit-exact under a fixed-zero=8
GPTQ code -- it goes through the same generic dequant + per-group-32
symmetric requantize path as Q4_1.

CORRECTION vs. the original task assumption (measured, not guessed --
see scripts/test_gguf2marlin.py's Q4_K/Q4_1 checks): re-fitting an
*independently affine-quantized* Q4_K/Q4_1 block into a *fixed-zero=8
symmetric* GPTQ code is NOT generically near-lossless, even though both
sides use 16 levels over the same group. Q4_K's per-block zero (encoded
via dmin*m[j]) is a free real number chosen independently of GPTQ's fixed
zero=8 code position; unless that offset happens to land within ~half a
quantization step of "code 8 == value 0" (a coincidence, not something a
generic block satisfies), no scale choice recovers the source's 16 exact
values -- the two grids are only "aligned" in level *count*, not in
level *position*. Empirically (scripts/test_gguf2marlin.py, Gaussian
weights, group_size=32, incl. a variant with exactly-symmetric per-block
min/max): relative RMS error between GGUF's own Q4_K dequant and this
script's GPTQ-dequant lands around 5-9%, essentially the same order as
quantizing fresh fp32 data straight to int4 group32 (~8-9% here; a finer
grid search does not meaningfully change this -- it is int4 RTN's
intrinsic noise floor, not a search-quality artifact, and matches the
magnitude scripts/quantize_gptq_9b.py's docstring independently reports
for group128 RTN). "< 1e-3" is achievable only for Q4_0 (see above), by
construction (same code, same scale, zero float rounding involved) --
not for Q4_1/Q4_K's generic re-fit path. This script still prints the
measured per-tensor error so a real run's numbers are never hidden or
assumed; treat a Q4_1/Q4_K tensor whose error is *far* outside the
5-15% band (e.g. >30%, or NaN) as an actual bug (wrong axis, wrong
group boundary), not "expected noise".

Any other block type present in a Q4_K_M mix (Q5_K, Q6_K, Q8_0, Q3_K,
Q2_K, ...) or plain F16/F32 tensors are dequantized to fp16 and kept
UNQUANTIZED ("mixed-scheme" checkpoint) -- see "WHICH TENSORS STAY FP16"
below. gguf-py's own dequantize() is reused for all of these (and for
Q4_1/Q4_K's float path); we do not reimplement K-quant sub-scale bit
unpacking ourselves; it is intricate (12-byte 6-bit-packed sub-scales) and
already correctly implemented and unit-covered upstream.

DECISION 3 -- ARCHITECTURE-GENERIC NAME MAPPING
------------------------------------------------
Rather than hand-enumerating one architecture's HF parameter names (as
scripts/transcode_gguf_to_gptq.py does for Qwen3.5), this script goes the
other direction: it reads `general.architecture` from the GGUF, resolves
the `gguf.MODEL_ARCH` enum, and builds `gguf.get_tensor_name_map(arch,
n_blocks)`. That object's `.mapping` dict already contains, for every
ggml tensor purpose relevant to that architecture, EVERY known upstream
HF-style name variant mapped to the ggml canonical name (used at GGUF
*write* time to recognize a source HF checkpoint's layer under any
naming convention it might use). We invert it: for each ggml canonical
name found in the actual GGUF file, we pick the *shortest* candidate name
that looks like the modern generic HF decoder convention
("model.layers.{bid}....", "lm_head", or a bare "model." global tensor) --
this is the convention shared by Llama/Qwen/Qwen2/Qwen3/Mistral/Gemma/Phi
and most current dense decoder architectures, and is what vLLM's stock HF
weight loader expects for a plain (non-GGUF) safetensors checkpoint. If no
such candidate exists for some tensor (this happens for exotic/hybrid
architectures whose upstream HF naming never matched the generic
convention -- Qwen3.5's GDN mixer is one such case, see LIMITATIONS), the
tensor is kept under a synthetic name and flagged loudly rather than
silently mismapped; the checkpoint will need architecture-specific
follow-up (Qwen3.5 already has one: scripts/transcode_gguf_to_gptq.py,
which also inverts llama.cpp's Qwen3.5-specific value transforms -- this
script deliberately does NOT attempt that, see LIMITATIONS).

WHICH TENSORS STAY FP16
------------------------
- Embeddings (token_embd -> *.embed_tokens) and the output head
  (output.weight -> lm_head, when untied) are ALWAYS kept fp16,
  regardless of their source GGUF block type (Q4_K_M commonly stores
  these as Q6_K or F16) -- matches the task's explicit policy and typical
  practice (quantizing the vocab-sized embedding/head has an outsized
  accuracy cost for a small VRAM win).
- Any tensor whose GGUF block type is not one of {Q4_0, Q4_1, Q4_K} is
  dequantized and kept fp16 (norms are always F32 in ggml and fall in
  here too, along with any Q5_K/Q6_K/Q8_0 tensor a Q4_K_M mix uses for a
  handful of "more sensitive" layers).
- Non-2D tensors (biases, 1-D norm/gate parameters) are never quantized
  (GPTQ's Linear-weight contract is inherently 2-D); kept fp16.
- vLLM's AutoGPTQ path only quantizes plain nn.Linear-shaped weights
  anyway (auto_gptq.py's Parameter registration assumes a 2-D weight);
  writing a 1-D tensor as qweight/qzeros/scales/g_idx would not be loaded
  by any LinearMethod regardless.

MERGED-PROJECTION SCHEME CONSISTENCY (the GDN qkvz/ba issue)
---------------------------------------------------------------
scripts/patch_vllm_gdn_quant_load.py documents a real vLLM loader
constraint: some architectures (Qwen3.5's gated-delta-net mixer) fuse
several independent HF checkpoint tensors into ONE vLLM
MergedColumnParallelLinear parameter at load time. Concatenating
independently-packed int4 shards along the output (N) axis is only valid
when every shard was packed with the IDENTICAL quantization scheme (same
bits, same group_size) -- packing lives along the K axis, so mismatched
per-shard K-packing corrupts (or crashes) the merge. Since this script
uses a single global (bits=4, group_size=<arg>) scheme for every tensor it
DOES quantize, a mismatch can only arise as "one half of a merge pair
ended up fp16 (because its source GGUF block type was not Q4_*) while the
other half got quantized" -- exactly the scenario the task calls out
("mot Q4 mot Q6"). `KNOWN_MERGE_PAIRS` below lists such pairs by HF-name
suffix (currently just Qwen3.5's two GDN pairs, extend as needed for other
hybrid architectures); if a pair is present and their would-be-quantized
status disagrees, BOTH tensors of the pair are forced to fp16 and a
warning is printed, before any quantization work is done on either.

ERROR BUDGET / WHAT THIS SCRIPT VERIFIES
-------------------------------------------
This is RTN, not calibrated GPTQ -- see scripts/transcode_gguf_to_gptq.py's
docstring for the standard caveats (no Hessian, no activation statistics,
each group independently rounded). Every quantized tensor's relative RMS
error, vs. the tensor's own GGUF-dequantized reference, is measured and
printed in a summary table at the end of the run:
    Q4_0            -> must be exactly 0.0 (byte-exact fast path, no
                       float rounding involved at all -- see DECISION 2)
    Q4_1 / Q4_K     -> empirically ~5-15% on Gaussian-ish weight data
                       (see the CORRECTION note under DECISION 2 -- this
                       is int4 RTN's intrinsic noise floor when re-fitting
                       an independently-chosen affine grid into a fixed
                       symmetric zero=8 code, not something a smarter
                       scale search removes). NOT the < 1e-3 originally
                       hoped for; that target was checked against ground
                       truth and found unreachable in general, and this
                       script says so rather than silently claiming it.
A run that shows Q4_0 with nonzero error, or Q4_1/Q4_K error far outside
that band (e.g. > 30%, or NaN), indicates an actual bug (mis-derived
group boundaries, wrong bit order, wrong axis) -- treat THAT as a
failure, not the expected 5-15% RTN noise itself.

LIMITATIONS (be honest)
--------------------------
- The "shortest modern-HF candidate" tie-break (DECISION 3) can pick a
  syntactically valid but architecturally WRONG name when one ggml tensor
  role is legitimately shared by several upstream naming conventions.
  Verified concretely for Qwen3.5: MODEL_TENSOR.ATTN_QKV's candidate list
  for arch QWEN35 contains BOTH "model.layers.{bid}.linear_attn.
  in_proj_qkv" (the real Qwen3.5 GDN name) AND "model.layers.{bid}.
  self_attn.qkv_proj" (a different, shorter, generic fused-QKV
  convention some other architecture uses for the same ggml role) --
  the shortest-wins heuristic picks the wrong one here. Same issue hits
  in_proj_b (collides with a shorter "self_attn.b_proj" alias). This is
  NOT a hypothetical -- scripts/test_gguf2marlin.py's merge-pair test
  unit-tests apply_merge_pair_fixups() directly against synthetic
  records rather than depending on a real GGUF resolving these two
  particular ambiguous names correctly, for exactly this reason. In
  general, disambiguating "which of several architectures' naming
  conventions applies" requires more than the ggml tensor's role alone
  (e.g. the target HF config's actual layer_types) -- out of scope for a
  fully generic script. Qwen3.5's GDN tensors specifically should still
  go through scripts/transcode_gguf_to_gptq.py, which hardcodes the real
  names instead of guessing.
- Only Q4_0 / Q4_1 / Q4_K source tensors are converted to GPTQ int4.
  Q2_K/Q3_K/Q5_K/Q6_K/Q8_0/IQ*-only checkpoints have nothing to quantize
  (every tensor falls into the fp16-passthrough path) -- this script will
  still run and produce a valid (fully unquantized) HF checkpoint, but
  that defeats the purpose; it is meant for Q4_0/Q4_1/Q4_K_M sources.
- The output config.json only reconstructs a handful of
  architecture-agnostic fields (vocab_size, hidden_size if derivable,
  num_hidden_layers, tie_word_embeddings) plus quantization_config.
  Pass --hf-config <path/to/config.json> to overlay the *real* upstream
  config (attention heads, rope, activation, hybrid-layer metadata, ...)
  -- without it, the emitted config.json is NOT sufficient to serve any
  non-trivial architecture and this script says so loudly at the end.
- No llama.cpp-side value/layout transforms are undone for hybrid/exotic
  architectures (e.g. Qwen3.5's GDN A_log/dt_bias/-exp()/tiled-V-head
  inversions -- see scripts/patch_gguf_qwen35_transforms.py and
  scripts/transcode_gguf_to_gptq.py, which DOES implement those). Tensors
  this script's generic name-mapper cannot resolve to a "model."-style HF
  name are written under a synthetic `unmapped.<ggml_name>` key and
  listed in the final report -- the checkpoint will not serve correctly
  until those are handled by an architecture-specific follow-up. For
  Qwen3.5 specifically, use scripts/transcode_gguf_to_gptq.py instead;
  this script's Qwen3.5 test run only exercises the generic-mapping path
  (the vanilla q/k/v/o/gate/up/down/norm/embed tensors resolve correctly;
  the four GDN in_proj_* tensors and A_log/dt_bias are expected to land in
  `unmapped.*` here, by design -- see scripts/test_gguf2marlin.py).
- No forward pass / activation calibration of any kind (RTN only, see
  ERROR BUDGET above).
- Streaming: this script processes one GGUF tensor at a time (bounded
  peak memory during compute -- never materializes more than one tensor's
  fp32 dequant buffer, and GGUFReader mmaps the source file rather than
  loading it whole), but the *packed* (already ~4x-8x smaller than fp16)
  output tensors accumulate in a dict until the final single
  safetensors.torch.save_file() call, since the public safetensors API
  has no incremental/streaming writer. For very large models this could
  be extended to shard across multiple *.safetensors files with an index;
  out of scope here since 4-bit output size is the whole point.

USAGE
-----
    python scripts/gguf2marlin.py model.gguf out_dir/ [--group-size 32]
                                   [--hf-config path/to/config.json]

Runs entirely on CPU with numpy/torch/gguf/safetensors -- no vLLM install
and no GPU needed. Verification (that vLLM 0.26 actually loads and serves
this checkpoint via gptq_marlin on the target GPU) is out of scope for
this box; see the Colab runbook printed at the end of a successful run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import gguf
from gguf.quants import dequantize as gguf_dequantize

# --------------------------------------------------------------------------
# GPTQ tensor-layout constants -- see DECISION 1 above.
# --------------------------------------------------------------------------
BITS = 4
PACK_FACTOR = 32 // BITS  # 8 int4 values per int32
QUANT_MIN, QUANT_MAX = -8, 7  # uint4b8 signed range
ZERO_POINT = 8  # uint4b8's fixed bias
MARLIN_SUPPORTED_GROUP_SIZES = (32, 64, 128)  # from marlin_utils.py (excl. -1 == no grouping)

QUANTIZABLE_GGUF_TYPES = {
    gguf.GGMLQuantizationType.Q4_0,
    gguf.GGMLQuantizationType.Q4_1,
    gguf.GGMLQuantizationType.Q4_K,
}

# HF-name suffix pairs that vLLM fuses into ONE MergedColumnParallelLinear
# parameter at load time (see scripts/patch_vllm_gdn_quant_load.py). Extend
# this list for other hybrid architectures with fused checkpoint tensors.
KNOWN_MERGE_PAIRS = [
    ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
]

ALWAYS_FP16_HF_SUFFIXES = ("embed_tokens", "lm_head")


# ==========================================================================
# 1. GPTQ bit-packing primitives -- verbatim port of vllm's
#    quant_utils.pack_rows / unpack_rows (numpy instead of torch so this
#    runs without a torch-with-CUDA install; math is identical).
# ==========================================================================


def pack_rows(q: np.ndarray, bits: int, k: int, n: int) -> np.ndarray:
    """int(0..2**bits-1) (K,N) -> int32 packed (K//pack_factor, N).

    Row i (0..pack_factor-1) within each pack_factor-sized group of K goes
    into bits [bits*i : bits*i+bits) of the packed int32 word -- matches
    vllm.model_executor.layers.quantization.utils.quant_utils.pack_rows
    exactly (the layout ops.gptq_marlin_repack expects on the vLLM side).
    """
    pack_factor = 32 // bits
    assert k % pack_factor == 0, f"k={k} not divisible by pack_factor={pack_factor}"
    q32 = q.astype(np.uint32)
    out = np.zeros((k // pack_factor, n), dtype=np.uint32)
    for i in range(pack_factor):
        out |= (q32[i::pack_factor, :] & ((1 << bits) - 1)) << (bits * i)
    return out.astype(np.int32)


def unpack_rows(packed: np.ndarray, bits: int, k: int, n: int) -> np.ndarray:
    """Inverse of pack_rows -- used only for this script's own verification."""
    pack_factor = 32 // bits
    arr = packed.astype(np.uint32)
    out = np.zeros((k, n), dtype=np.uint32)
    mask = (1 << bits) - 1
    for i in range(pack_factor):
        out[i::pack_factor, :] = (arr >> (bits * i)) & mask
    return out.astype(np.int32)


def make_symmetric_qzeros(num_groups: int, n: int, bits: int) -> np.ndarray:
    """All-zero-point-8 qzeros. Inert at inference (see DECISION 1 -- vLLM's
    Marlin kernel discards qzeros content when zero_points=False), but
    shape-correct and byte-correct for any tool that does read it."""
    pack_factor = 32 // bits
    assert n % pack_factor == 0
    packed_val = 0
    for i in range(pack_factor):
        packed_val |= ZERO_POINT << (bits * i)
    return np.full((num_groups, n // pack_factor), packed_val, dtype=np.uint32).astype(np.int32)


# ==========================================================================
# 2. Q4_0 byte-exact fast path (see DECISION 2 -- verified against
#    gguf.quants.quantize()/dequantize() as an independent reference).
# ==========================================================================


def unpack_q4_0_exact(raw_bytes: np.ndarray, out_features: int, in_features: int):
    """Bit-exact extraction of a Q4_0 tensor's GPTQ-biased int4 codes and
    fp16 per-group(32) scales, with NO float dequant/requant step.

    `raw_bytes` is the tensor's raw ggml byte buffer (gguf.ReaderTensor.data
    for a Q4_0 tensor), any shape that flattens to (n_blocks, 18) in
    row-major block order -- true for GGUFReader's own tensor.data layout.
    Only mathematically valid when in_features % 32 == 0 (Q4_0's native
    block size) and the target group_size is also 32 -- caller must check.

    Returns (q_biased (in_features, out_features) uint8 in [0,15],
             scales (in_features // 32, out_features) fp16 numpy array).
    """
    assert in_features % 32 == 0, f"in_features={in_features} not divisible by 32"
    n_groups = in_features // 32
    raw = raw_bytes.reshape(-1, 18)
    assert raw.shape[0] == out_features * n_groups, (
        f"unexpected Q4_0 block count {raw.shape[0]}, expected "
        f"{out_features * n_groups} for shape ({out_features},{in_features})"
    )
    d = raw[:, 0:2].copy().view(np.float16).reshape(-1)  # (out_features*n_groups,)
    qs = raw[:, 2:18]  # (out_features*n_groups, 16)
    low = (qs & 0x0F).astype(np.uint8)
    high = ((qs >> 4) & 0x0F).astype(np.uint8)
    elems = np.concatenate([low, high], axis=1)  # (blocks, 32), already GPTQ-biased [0,15]

    q_biased_NK = elems.reshape(out_features, in_features)  # (N, K)
    scales_N_groups = d.reshape(out_features, n_groups)  # (N, num_groups)

    q_biased_KN = np.ascontiguousarray(q_biased_NK.T)  # (K, N)
    scales_groupsN = np.ascontiguousarray(scales_N_groups.T)  # (num_groups, N)
    return q_biased_KN, scales_groupsN


# ==========================================================================
# 3. Generic RTN symmetric-group requantizer (Q4_1 / Q4_K / any dequantized
#    fp32 source -- see DECISION 2).
# ==========================================================================


def quantize_symmetric_group(w_kn: np.ndarray, group_size: int, n_grid: int = 9):
    """fp32 (K,N) -> GPTQ int4 symmetric (zero=8), group_size along K.

    Plain min/max-derived scale, refined by a small clip-ratio grid search
    (shrinking the min/max envelope trades the rare outlier's error for
    better resolution on the bulk of the group -- standard RTN+clip
    baseline, uses only this tensor's own values, still calibration-free).
    Returns (qweight int32 packed (K//8,N), scales fp16 (K//group_size,N),
    w_ref fp32 (K,N) dequantized reconstruction, for error reporting).
    """
    k, n = w_kn.shape
    assert k % group_size == 0, f"k={k} not divisible by group_size={group_size}"
    num_groups = k // group_size
    wg = w_kn.reshape(num_groups, group_size, n)

    max_val = wg.max(axis=1, keepdims=True)
    min_val = wg.min(axis=1, keepdims=True)
    base_scale = np.maximum(np.abs(max_val / QUANT_MAX), np.abs(min_val / QUANT_MIN))
    base_scale = np.clip(base_scale, 1e-10, None)

    best_scale, best_mse = base_scale, None
    for ratio in np.linspace(1.0, 0.6, n_grid):
        s = np.clip(base_scale * ratio, 1e-10, None)
        q_try = np.clip(np.round(wg / s), QUANT_MIN, QUANT_MAX)
        mse = ((q_try * s - wg) ** 2).mean(axis=1, keepdims=True)
        if best_mse is None:
            best_mse, best_scale = mse, s
        else:
            better = mse < best_mse
            best_scale = np.where(better, s, best_scale)
            best_mse = np.where(better, mse, best_mse)
    scale = best_scale

    q = np.clip(np.round(wg / scale), QUANT_MIN, QUANT_MAX)
    w_ref = (q * scale).reshape(k, n).astype(np.float32)
    scales_out = scale.reshape(num_groups, n).astype(np.float16)
    q_biased = (q + ZERO_POINT).reshape(k, n).astype(np.int32)
    qweight = pack_rows(q_biased, BITS, k, n)
    return qweight, scales_out, w_ref


def dequant_gptq_for_verification(qweight: np.ndarray, scales: np.ndarray, k: int, n: int, group_size: int) -> np.ndarray:
    """Unpack + dequantize our own on-disk GPTQ tensors back to (K,N) fp32,
    independent of the encoder's internals -- the "read our own output
    back" half of the per-tensor error report."""
    q = unpack_rows(qweight, BITS, k, n).astype(np.float32) - ZERO_POINT
    num_groups = k // group_size
    q = q.reshape(num_groups, group_size, n)
    s = scales.astype(np.float32).reshape(num_groups, 1, n)
    return (q * s).reshape(k, n)


# ==========================================================================
# 4. Architecture-generic ggml-name -> HF-name mapping (see DECISION 3).
# ==========================================================================


def resolve_arch(reader: "gguf.GGUFReader"):
    field = reader.fields.get("general.architecture")
    if field is None:
        return None, None
    arch_str = field.contents()
    name_to_enum = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}
    return arch_str, name_to_enum.get(arch_str)


# EXACT tail-suffixes of the single most common modern HF decoder naming
# convention (Llama/Qwen/Qwen2/Qwen3/Mistral/Gemma/Phi/StableLM/...).
# Picking "shortest candidate" alone is NOT a reliable proxy for "the
# modern convention" -- verified concretely (build_ggml_to_hf_map's
# docstring): the much older llama-pth convention ("attention.wq",
# "feed_forward.w1", ...) is frequently *shorter* than the modern one, and
# even a loose "contains self_attn." substring match is too coarse (e.g.
# "self_attn.dense" -- a Falcon/GPT-NeoX-style alias for the SAME ggml
# attn_output role -- is shorter than the correct "self_attn.o_proj" and
# would still win). Matching the exact canonical suffix avoids both
# failure modes. This is a property of "the ecosystem's dominant
# convention", not of any one specific model family, so using it does not
# reintroduce the per-architecture hardcoding this script otherwise
# avoids (see DECISION 3, and its LIMITATIONS entry for the one place
# this still isn't enough -- hybrid/SSM tensor roles, e.g. Qwen3.5's GDN
# in_proj_qkv/in_proj_b, that have no exact-suffix match here at all and
# fall through to the less reliable shortest-overall tie-break).
_MODERN_HF_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "input_layernorm", "post_attention_layernorm",
    "embed_tokens", "lm_head", "model.norm",
)


def build_ggml_to_hf_map(arch_enum, n_blocks: int) -> dict[str, str]:
    """ggml canonical tensor name -> a "modern generic HF decoder" name.

    gguf.get_tensor_name_map(arch, n_blocks).mapping maps EVERY known
    upstream naming-convention variant (across all architectures gguf-py
    has ever needed to write) to (MODEL_TENSOR, ggml_canonical_name). We
    invert it and, for each ggml canonical name, pick a candidate that (1)
    starts with "model." or "lm_head" (excludes the ggml name itself and
    non-"model."-rooted conventions), preferring (2) one that contains a
    _MODERN_HF_SUFFIXES exact suffix, breaking remaining ties by shortest name.
    Falls back to the shortest candidate overall (with a caller-visible
    gap) when no "model."/"lm_head" candidate exists at all.
    """
    tmap = gguf.get_tensor_name_map(arch_enum, n_blocks)
    candidates_by_ggml: dict[str, list[str]] = {}
    for key, (_tensor, ggml_name) in tmap.mapping.items():
        candidates_by_ggml.setdefault(ggml_name, []).append(key)

    ggml_to_hf = {}
    for ggml_name, candidates in candidates_by_ggml.items():
        modern = [c for c in candidates if c.startswith("model.") or c.startswith("lm_head")]
        pool = modern if modern else candidates
        hinted = [c for c in pool if any(c.endswith(h) for h in _MODERN_HF_SUFFIXES)]
        ggml_to_hf[ggml_name] = min(hinted if hinted else pool, key=len)
    return ggml_to_hf


def split_ggml_name(name: str):
    if name.endswith(".weight"):
        return name[: -len(".weight")], ".weight"
    if name.endswith(".bias"):
        return name[: -len(".bias")], ".bias"
    return name, ""


def apply_merge_pair_fixups(records: list[dict]) -> set[str]:
    """See KNOWN_MERGE_PAIRS / module docstring "MERGED-PROJECTION SCHEME
    CONSISTENCY". Given the pass-1 classification records (each a dict with
    at least "hf_base" and "would_quantize"), returns the set of hf_base
    names that must be forced to fp16 because their merge-pair partner
    disagrees on quantized-vs-not. Factored out from main() so it's
    directly unit-testable without a real GGUF file or subprocess --
    resolving an architecture's ggml-name -> HF-name mapping (a separate,
    much harder problem for hybrid architectures, see LIMITATIONS) is not
    a precondition for exercising this logic correctly.
    """
    forced_fp16 = set()
    for suffix_a, suffix_b in KNOWN_MERGE_PAIRS:
        pairs = [
            (r_a, r_b)
            for r_a in records if r_a["hf_base"].endswith(suffix_a)
            for r_b in records if r_b["hf_base"] == r_a["hf_base"][: -len(suffix_a)] + suffix_b
        ]
        for r_a, r_b in pairs:
            if r_a["would_quantize"] != r_b["would_quantize"]:
                print(f"[gguf2marlin] WARNING: merge pair ({r_a['hf_base']}, {r_b['hf_base']}) "
                      f"has mismatched schemes (one quantizable, one not) -- forcing BOTH to "
                      f"fp16 so vLLM's merged-column loader sees a consistent shard scheme "
                      f"(see scripts/patch_vllm_gdn_quant_load.py)", file=sys.stderr)
                forced_fp16.add(r_a["hf_base"])
                forced_fp16.add(r_b["hf_base"])
    return forced_fp16


# ==========================================================================
# 5. Main transcode pipeline.
# ==========================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Transcode a Q4_0/Q4_1/Q4_K(_M) GGUF checkpoint into a GPTQ "
        "safetensors checkpoint vLLM 0.26 serves via gptq_marlin. See this file's "
        "module docstring for the full format decision writeup."
    )
    ap.add_argument("gguf_path", help="path to the input .gguf file")
    ap.add_argument("out_dir", help="output directory for the GPTQ checkpoint")
    ap.add_argument("--group-size", type=int, default=32, choices=MARLIN_SUPPORTED_GROUP_SIZES,
                     help="GPTQ group size (default 32 -- matches Q4_0's native block size, "
                          "enabling the byte-exact fast path; must be one of vLLM's "
                          "MARLIN_SUPPORTED_GROUP_SIZES)")
    ap.add_argument("--hf-config", default=None,
                     help="optional path to a real upstream config.json to overlay "
                          "(architecture-specific fields this script cannot reconstruct "
                          "generically -- see LIMITATIONS in the module docstring)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_size = args.group_size

    print(f"[gguf2marlin] reading {args.gguf_path}", file=sys.stderr)
    reader = gguf.GGUFReader(args.gguf_path)

    arch_str, arch_enum = resolve_arch(reader)
    n_blocks_field = reader.fields.get(f"{arch_str}.block_count") if arch_str else None
    n_blocks = int(n_blocks_field.contents()) if n_blocks_field is not None else 128
    if arch_enum is None:
        print(f"[gguf2marlin] WARNING: unknown architecture {arch_str!r}; "
              f"falling back to LLAMA's tensor-name table (covers standard "
              f"attn/mlp/norm/embed naming for most dense decoders)", file=sys.stderr)
        arch_enum = gguf.MODEL_ARCH.LLAMA
    ggml_to_hf = build_ggml_to_hf_map(arch_enum, n_blocks)
    print(f"[gguf2marlin] architecture={arch_str!r} n_blocks={n_blocks} "
          f"({len(ggml_to_hf)} known tensor purposes)", file=sys.stderr)

    # ---- pass 1: classify every tensor (cheap -- no dequant yet) ----------
    records = []  # dict(hf_base, suffix, ggml_name, tensor, would_quantize, mapped)
    for t in reader.tensors:
        base, suffix = split_ggml_name(t.name)
        hf_base = ggml_to_hf.get(base)
        mapped = hf_base is not None
        if not mapped:
            hf_base = f"unmapped.{base}"
        is_always_fp16 = any(hf_base.endswith(s) for s in ALWAYS_FP16_HF_SUFFIXES)
        is_2d = len(t.shape) == 2
        would_quantize = (
            mapped and is_2d and not is_always_fp16
            and gguf.GGMLQuantizationType(t.tensor_type) in QUANTIZABLE_GGUF_TYPES
        )
        records.append({
            "hf_base": hf_base, "suffix": suffix, "ggml_name": base,
            "tensor": t, "would_quantize": would_quantize, "mapped": mapped,
        })

    # ---- pass 2: merge-pair scheme-consistency fixups ----------------------
    forced_fp16 = apply_merge_pair_fixups(records)

    # ---- pass 3: do the actual work ----------------------------------------
    out_tensors: dict[str, torch.Tensor] = {}
    quantized_modules, fp16_kept, unmapped_names = [], [], []
    error_report = []  # (hf_base, gguf_type_name, rel_rms)

    for r in records:
        t = r["tensor"]
        hf_base, suffix = r["hf_base"], r["suffix"]
        hf_name = hf_base + suffix
        gtype = gguf.GGMLQuantizationType(t.tensor_type)
        quantize_this = r["would_quantize"] and hf_base not in forced_fp16

        if not r["mapped"]:
            unmapped_names.append(hf_base)

        if quantize_this:
            out_features, in_features = int(t.shape[-1]), int(t.shape[0])
            if gtype == gguf.GGMLQuantizationType.Q4_0 and group_size == 32:
                q_biased, scales = unpack_q4_0_exact(t.data, out_features, in_features)
                qweight = pack_rows(q_biased.astype(np.int32), BITS, in_features, out_features)
                w_ref = dequant_gptq_for_verification(qweight, scales, in_features, out_features, group_size)
                w_pre = np.ascontiguousarray(gguf_dequantize(t.data, gtype), dtype=np.float32).T
            else:
                w_pre_out_in = np.ascontiguousarray(gguf_dequantize(t.data, gtype), dtype=np.float32)
                w_pre = w_pre_out_in.T  # (K, N)
                qweight, scales, w_ref = quantize_symmetric_group(w_pre, group_size)

            k_dim, n_dim = in_features, out_features
            qzeros = make_symmetric_qzeros(k_dim // group_size, n_dim, BITS)
            g_idx = (np.arange(k_dim, dtype=np.int32) // group_size)

            out_tensors[hf_name.replace(".weight", ".qweight")] = torch.from_numpy(np.ascontiguousarray(qweight))
            out_tensors[hf_name.replace(".weight", ".qzeros")] = torch.from_numpy(np.ascontiguousarray(qzeros))
            out_tensors[hf_name.replace(".weight", ".scales")] = torch.from_numpy(np.ascontiguousarray(scales))
            out_tensors[hf_name.replace(".weight", ".g_idx")] = torch.from_numpy(np.ascontiguousarray(g_idx))
            quantized_modules.append(hf_base)

            denom = float(np.sqrt((w_pre.astype(np.float64) ** 2).mean())) or 1.0
            rel_rms = float(np.sqrt(((w_ref.astype(np.float64) - w_pre.astype(np.float64)) ** 2).mean())) / denom
            error_report.append((hf_base, gtype.name, rel_rms))
        else:
            if gtype in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                arr = np.ascontiguousarray(t.data, dtype=np.float32)
            else:
                arr = np.ascontiguousarray(gguf_dequantize(t.data, gtype), dtype=np.float32)
            out_tensors[hf_name] = torch.from_numpy(arr.astype(np.float16))
            fp16_kept.append(hf_base)

    # lm_head: drop if tied (mirrors scripts/transcode_gguf_to_gptq.py).
    has_output = any(t.name in ("output.weight",) for t in reader.tensors)
    tie_word_embeddings = not has_output
    if tie_word_embeddings:
        out_tensors.pop("lm_head.weight", None)

    from safetensors.torch import save_file
    save_file(out_tensors, out_dir / "model.safetensors", metadata={"format": "pt"})

    # ---- config.json --------------------------------------------------------
    cfg = {}
    if args.hf_config:
        cfg = json.loads(Path(args.hf_config).read_text(encoding="utf-8"))
        cfg = cfg.get("text_config", cfg)
    else:
        token_embd = next((t for t in reader.tensors if t.name == "token_embd.weight"), None)
        if token_embd is not None:
            cfg["vocab_size"] = int(token_embd.shape[-1])
            cfg["hidden_size"] = int(token_embd.shape[0])
        cfg["num_hidden_layers"] = n_blocks
        cfg["tie_word_embeddings"] = tie_word_embeddings
        cfg["model_type"] = arch_str

    cfg["quantization_config"] = {
        "quant_method": "gptq",
        "bits": BITS,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[gguf2marlin] wrote {out_dir} -- {len(quantized_modules)} modules quantized to "
          f"GPTQ int4 g{group_size}, {len(fp16_kept)} kept fp16, {len(unmapped_names)} unmapped",
          file=sys.stderr)

    _print_report(args, quantized_modules, fp16_kept, unmapped_names, error_report, args.hf_config is None)
    return 0


def _print_report(args, quantized_modules, fp16_kept, unmapped_names, error_report, config_is_best_effort):
    print("\n" + "=" * 72)
    print("VERIFICATION REPORT")
    print("=" * 72)
    print(f"quantized (GPTQ int4 g{args.group_size}): {len(quantized_modules)}")
    print(f"kept fp16: {len(fp16_kept)}")
    print(f"unmapped (no generic HF name found, written under 'unmapped.*'): "
          f"{len(unmapped_names)}")
    if unmapped_names:
        print(f"    {unmapped_names[:12]}{' ...' if len(unmapped_names) > 12 else ''}")
        print("    NOTE: these tensors will NOT be visible to vLLM's stock HF loader "
              "under any real parameter name -- see LIMITATIONS in the module docstring.")

    if error_report:
        print(f"\nper-tensor relative RMS error (GPTQ-dequant vs GGUF-dequant), by source type:")
        by_type: dict[str, list[float]] = {}
        for _name, gtype_name, rel_rms in error_report:
            by_type.setdefault(gtype_name, []).append(rel_rms)
        for gtype_name, errs in sorted(by_type.items()):
            errs_np = np.array(errs)
            print(f"    {gtype_name:8s}  n={len(errs):4d}  mean={errs_np.mean():.6f}  "
                  f"max={errs_np.max():.6f}  min={errs_np.min():.6f}")
        worst = sorted(error_report, key=lambda x: -x[2])[:5]
        print("  worst 5 tensors:")
        for name, gtype_name, rel_rms in worst:
            print(f"    {rel_rms:.6f}  {gtype_name:6s}  {name}")

    if config_is_best_effort:
        print("\nNOTE: no --hf-config given -- config.json only has generic "
              "vocab_size/hidden_size/num_hidden_layers/quantization_config fields. "
              "Pass --hf-config <real config.json> before serving anything but a toy model.")

    print("\nSuggested Colab acceptance run (GPU, not run from this box):")
    print("    vllm serve <out_dir> --quantization gptq --max-model-len 8192")
    print("    curl <url>/v1/completions -d '{\"prompt\": \"...\", \"max_tokens\": 32}'")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
