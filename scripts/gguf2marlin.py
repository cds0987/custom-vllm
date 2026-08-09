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

DECISION 1b -- OPT-IN int8 BRANCH FOR K-QUANTS (--k-quants-to int8)
--------------------------------------------------------------------
Default behaviour (--k-quants-to int4, unchanged from before this flag
existed) still routes Q4_1/Q4_K through the ~5-15%-RMS symmetric-zero=8
int4 re-fit described in DECISION 2/CORRECTION below, and leaves Q5_K/Q6_K
untouched fp16. Passing --k-quants-to int8 instead promotes Q4_1, Q4_K,
Q5_K, AND Q6_K to a GPTQ **8-bit** encoding (group_size stays whatever
--group-size says, default 32): dequant each block with gguf-py's own
dequantize() (same vetted reference as the int4 generic path), then
requantize per-group(32) into a *symmetric* int8 GPTQ code -- 256 levels
instead of 16 cuts the RTN quantization-noise floor roughly in proportion
to level count, which is why the task's ~0.5% (5e-3) target is plausible
for int8 where < 1e-2 was not for int4 (measured, see
scripts/test_gguf2marlin.py's Q4_K-int8 test, not merely asserted here).

Verified against the LOCAL vLLM checkout (D:\\Training\\AI_Module\\vllm\\vllm\\vllm)
that GPTQ 8-bit + group_size=32 is a real Marlin-served on-disk format --
not assumed:
  - vllm/model_executor/layers/quantization/auto_gptq.py:101-104
    AutoGPTQConfig.TYPE_MAP = {(4, True): scalar_types.uint4b8,
    (8, True): scalar_types.uint8b128} -- 8-bit symmetric IS a first-class
    AutoGPTQ/Marlin type, not something this format has to be coerced into.
  - vllm/scalar_type.py:354 `uint8b128 = ScalarType.uint(8, 128)` -- bias
    (zero point) is 128 = 2**(8-1), the same "2**(bits-1)" pattern as
    uint4b8's bias=8=2**(4-1) (line 350). bits_zero_and_range() below
    generalizes this rather than hardcoding two magic numbers.
  - vllm/model_executor/layers/quantization/utils/quant_utils.py:705-706
    SUPPORTED_GPTQ_QUANT_TYPES = [uint4b8, uint8b128];
    SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128] -- ONE group-size list shared
    by both bit widths. vllm/model_executor/layers/quantization/utils/
    marlin_utils.py:35 MARLIN_SUPPORTED_GROUP_SIZES is the same list and is
    not indexed by bit width anywhere it's consumed
    (_check_marlin_supported, marlin_utils.py:117-149, takes quant_type and
    group_size as independent arguments). CONCLUSION: the task's suggested
    fallback ("if 8-bit only supports group 128/-1, follow that") does not
    apply to this codebase -- group_size=32 is valid for 8-bit exactly as
    for 4-bit, confirmed by reading the actual assertion/check code, not
    inferred from a resource-usage argument.
  - pack_rows/pack_cols (quant_utils.py:758-805) are already parameterized
    by num_bits with pack_factor = 32 // num_bits computed generically (4
    for 8-bit vs 8 for 4-bit) -- "int8 packs fewer elements per int32" is
    literally this formula, not a separate contract to reverse-engineer.
    pack_rows() in this script (Section 1 below) already took a `bits`
    argument for exactly this reason; only the *call sites* needed to stop
    hardcoding BITS=4.
  - Per-module MIXED bit widths in one checkpoint (Q4_0 tensors stay
    int4 while promoted K-quant tensors go int8) are not a hack layered on
    top of a single-scheme format: vllm/model_executor/layers/
    quantization/utils/gptq_utils.py's override_config()/
    get_dynamic_override() implement AutoGPTQConfig's documented `dynamic`
    regex-keyed per-module override (auto_gptq.py:136-145's own docstring
    example literally shows overriding `bits` for a layer subset), invoked
    per-layer from get_linear_quant_method() (gptq_utils.py:117-147) via
    `deepcopy(config)` + `override_config(cloned_config, prefix=prefix)`
    before the LinearMethod is constructed -- i.e. every Linear layer gets
    its OWN AutoGPTQConfig with bits/group_size/sym resolved from whichever
    dynamic rule (if any) matches its prefix, entirely independent of every
    other layer's resolution. This script writes one such rule per
    promoted-to-int8 module: `{"+:" + re.escape(hf_base) + "$": {"bits": 8}}"`
    in config.json's quantization_config["dynamic"] (base config stays
    bits=4 for the Q4_0/Q4_1/Q4_K-int4 majority).

Symmetric zero point qzeros/g_idx/pack_rows all now take a `bits` parameter
(threaded through from the target scheme instead of the module-global BITS
constant) so the SAME code paths serve both branches -- see Section 1/3
below; nothing about Q4_0's byte-exact fast path changes.

Not attempted here: promoting Q2_K/Q3_K/Q8_0/IQ*-family tensors to int8 --
out of the task's explicit scope ("Q4_K/Q4_1/Q5_K/Q6_K"); they remain fp16
passthrough in both --k-quants-to modes.

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
  the shortest-wins heuristic picks the wrong one here. In general,
  disambiguating "which of several architectures' naming conventions
  applies" requires more than the ggml tensor's role alone (e.g. the
  target HF config's actual layer_types) -- out of scope for a fully
  generic script.
  TASK K3 UPDATE: this WAS silently producing wrong-but-plausible-looking
  names for every one of Qwen3.5's GDN tensors (attn_qkv, attn_gate,
  ssm_beta, ssm_a, ssm_dt, ssm_norm, ssm_out -- confirmed on a real
  Qwen3.5-2B GGUF: vLLM raised "There is no module or parameter named
  'layers.0.A_log' in Qwen3_5Model"), not landing them in `unmapped.*` as
  this section previously (incorrectly) claimed. `qwen35_gdn_override_name`
  (see its own docstring above, next to `resolve_arch`) now hardcodes the
  correct "model.layers.{bid}.linear_attn.*" name for all nine GDN tensor
  roles when `arch_enum` is Qwen3.5/Qwen3.5-MoE, overriding the generic
  mapper for exactly those tensors -- same hardcode-the-known-exception
  approach scripts/graft_gguf_gdn.py's GGML_SUFFIX_FOR_HF_SUFFIX already
  used for the four in_proj_* tensors, extended here to the remaining
  five (A_log, dt_bias, conv1d, norm, out_proj) gguf2marlin.py also names.
  This fixes NAMING only -- see the very next bullet for what's still
  outstanding.
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
- STILL OUTSTANDING (not part of TASK K3's scope -- naming only): no
  llama.cpp-side VALUE/layout transforms are undone for Qwen3.5's GDN
  tensors -- see scripts/patch_gguf_qwen35_transforms.py's docstring:
  A_log is stored as -exp(A_log), linear_attn.norm's siblings are stored
  as 1+weight (linear_attn.norm itself is excluded -- plain RMSNorm),
  conv1d is squeezed (C,1,K)->(C,K), and V-head-carrying tensors
  (in_proj_qkv's V-rows, in_proj_z, in_proj_a, in_proj_b, and per that
  same docstring also A_log/dt_bias/conv1d's V-channels and out_proj's
  columns) are stored in llama.cpp's "tiled" head order rather than HF's
  "grouped" order whenever num_k_heads != num_v_heads. gguf2marlin.py
  copies these tensors' raw dequantized VALUES through unchanged -- only
  their NAMES are now correct (see the LIMITATIONS entry above). A
  checkpoint built by this script for a real Qwen3.5 GGUF (any size with
  grouped-value GDN heads, i.e. num_v_per_k > 1) will therefore now LOAD
  without a naming crash but almost certainly still generate garbage,
  exactly as scripts/patch_gguf_qwen35_transforms.py's own docstring
  describes happening to the live vllm_gguf_plugin path before that patch
  existed. scripts/transcode_gguf_to_gptq.py DOES implement all of these
  inversions (see its `_undo_qwen35_gguf_transform`) and remains the
  correct tool for a from-scratch Qwen3.5 GDN checkpoint; porting the same
  inversions into gguf2marlin.py's generic pipeline is unresolved --
  flag this explicitly before trusting a gguf2marlin.py-produced Qwen3.5
  checkpoint's *generation quality* (as opposed to "does it load") on a
  GGUF with grouped-value GDN heads.
- Tensors this script's generic name-mapper cannot resolve to a
  "model."-style HF name (any architecture other than Qwen3.5's own GDN
  tensors, which are now hardcoded -- see above) are written under a
  synthetic `unmapped.<ggml_name>` key and listed in the final report --
  the checkpoint will not serve correctly until those are handled by an
  architecture-specific follow-up.
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
import re
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

# GGUF-parser-internal architecture-string aliases that are NOT real
# transformers `model_type` values -- see scripts/patch_gguf_plugin.py's
# "qwen3.5 model_type normalization" (it maps the real HF model_type
# "qwen3_5"/"qwen3_5_moe" to these gguf-only spellings purely so
# vllm-gguf-plugin's own gguf.MODEL_ARCH_NAMES exact-match lookup
# succeeds). Used below as a --hf-config sanity-check heuristic (Bug C
# in this repo's auto-marlin hook): if a caller-supplied "real" config's
# model_type lands on one of these, it is very likely actually this
# script's own generic fallback (or a copy of it), not the true upstream
# config -- see scripts/patch_gguf_auto_marlin.py's module docstring.
_GGUF_INTERNAL_ONLY_MODEL_TYPES = {
    "qwen35": "qwen3_5",
    "qwen35moe": "qwen3_5_moe",
}

# --------------------------------------------------------------------------
# BUG (fallback architectures) FIX -- see module docstring.
#
# Reproduced on a real Qwen/Qwen3.5-2B config.json: its `text_config` sub-
# dict has no 'architectures' field of its own (only the top-level config
# does, per transformers' nested-config convention for a model that ALSO
# ships a multimodal wrapper class). The old code unconditionally did
# `cfg = cfg.get("text_config", cfg)`, discarding the top-level
# 'architectures' entirely and leaving the emitted config.json without one
# -- which sent vLLM down its OWN fallback (vllm/transformers_utils/
# config.py, "Architecture mapping for models without explicit
# architectures field": `MODEL_MAPPING_NAMES[config.model_type]`, i.e.
# transformers' plain AutoModel registry, NOT AutoModelForCausalLM) and
# produced "Qwen3_5TextModel" -- a bare backbone class with no LM head /
# generate() support, not something vLLM can serve as a CausalLM.
#
# `resolve_architectures` below fixes the DISCARDING problem generically
# (checks the top-level config too, not just whichever sub-dict `cfg` end
# up aliasing) and, only if genuinely nothing is available anywhere, falls
# back to inferring the correct *ForCausalLM* class from `model_type` --
# using this small table of real, verified mappings first (not a single
# hardcoded model: scripts/transcode_gguf_to_gptq.py's own
# `cfg["architectures"] = ["Qwen3_5ForCausalLM"]` and vLLM's own model
# registry, D:\Training\AI_Module\vllm\vllm\vllm\model_executor\models\
# registry.py, are the source for each entry here), THEN a best-effort
# generic naming-convention guess (loudly flagged as unverified) only for
# a model_type this table doesn't recognize.
_MODEL_TYPE_TO_CAUSAL_LM_CLASS = {
    # Qwen3.5 text_config.model_type is "qwen3_5_text" (see D:\Training\
    # AI_Module\vllm\vllm\vllm\transformers_utils\configs\qwen3_5.py:23);
    # top-level model_type is "qwen3_5". Both text-only checkpoints this
    # script produces load through the plain Qwen3_5ForCausalLM class
    # (registry.py has no separate "*_text"-suffixed entry -- the
    # ConditionalGeneration/multimodal wrapper is a different top-level
    # class this script never targets, see LIMITATIONS).
    "qwen3_5_text": "Qwen3_5ForCausalLM",
    "qwen3_5": "Qwen3_5ForCausalLM",
    "qwen3_5_moe_text": "Qwen3_5MoeForCausalLM",
    "qwen3_5_moe": "Qwen3_5MoeForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "qwen3_next": "Qwen3NextForCausalLM",
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
    "phi3": "Phi3ForCausalLM",
}


def _guess_causal_lm_class_name(model_type: str) -> str:
    """Best-effort transformers-style class name for a model_type this
    script doesn't have a verified mapping for: CamelCase each
    '_'-separated segment and append 'ForCausalLM' (e.g. "my_model" ->
    "MyModelForCausalLM"). This is a GUESS, not a verified lookup -- unlike
    transformers' actual model_type resolution (its own MODEL_FOR_CAUSAL_LM_
    MAPPING_NAMES registry), there is no general rule that reliably
    reconstructs an arbitrary architecture's exact class spelling from its
    model_type string alone (case/digit-boundary conventions genuinely
    differ per family -- e.g. the real "qwen3_5" keeps its underscore
    before the digit in "Qwen3_5ForCausalLM", while "qwen2" has none at all
    in "Qwen2ForCausalLM" -- this generic rule gets the latter right and
    would get the former wrong, which is exactly why it's the FALLBACK,
    checked only after `_MODEL_TYPE_TO_CAUSAL_LM_CLASS` above). Callers
    MUST treat this as unverified and warn loudly (see
    `resolve_architectures`)."""
    return "".join(seg.capitalize() for seg in model_type.split("_") if seg) + "ForCausalLM"


def resolve_architectures(raw_cfg: dict, text_cfg: dict) -> tuple[list[str] | None, bool]:
    """Decide the output config.json's 'architectures' list for a
    --hf-config overlay. Returns (architectures_or_None, is_unverified_guess).

    Order (see the BUG (fallback architectures) FIX note above):
      (a) raw_cfg's own top-level 'architectures', if present -- the
          top-level config is what a real checkpoint repo's config.json
          carries this in even when `text_config` doesn't have its own
          copy (confirmed: Qwen/Qwen3.5-2B's real config.json).
      (b) text_cfg's own 'architectures', if it happens to carry one
          itself (covers a flat, non-nested config where `text_cfg IS
          raw_cfg` and already has it -- a no-op restating (a), and a
          nested config that unusually duplicates it on the sub-dict).
      (c) infer from text_cfg's 'model_type' via
          _MODEL_TYPE_TO_CAUSAL_LM_CLASS (verified mappings).
      (d) best-effort generic guess (see _guess_causal_lm_class_name),
          is_unverified_guess=True so the caller prints a loud warning.
      (e) (None, False) if there's no model_type anywhere to guess from
          either -- the caller already warns about that missing-model_type
          case separately.
    """
    if raw_cfg.get("architectures"):
        return list(raw_cfg["architectures"]), False
    if text_cfg.get("architectures"):
        return list(text_cfg["architectures"]), False
    model_type = text_cfg.get("model_type")
    if not model_type:
        return None, False
    known = _MODEL_TYPE_TO_CAUSAL_LM_CLASS.get(model_type)
    if known is not None:
        return [known], False
    return [_guess_causal_lm_class_name(model_type)], True


# --- int8 branch (--k-quants-to int8) -- see DECISION 1b below for the
# vLLM-source evidence that this is a real, Marlin-served on-disk format
# (not a guess): bias/range come straight from vllm/scalar_type.py's
# ScalarType.uint(bits, 2**(bits-1)) definition, verified for both 4- and
# 8-bit below rather than assumed to generalize.
def bits_zero_and_range(bits: int) -> tuple[int, int, int]:
    """(zero_point, quant_min, quant_max) for a symmetric uintNb(2**(N-1))
    GPTQ code -- zero_point == 8 for bits=4 (scalar_types.uint4b8) and
    == 128 for bits=8 (scalar_types.uint8b128), matching vllm/scalar_type.py:
    uint4b8 = ScalarType.uint(4, 8), uint8b128 = ScalarType.uint(8, 128)."""
    zp = 1 << (bits - 1)
    return zp, -zp, zp - 1

QUANTIZABLE_GGUF_TYPES = {
    gguf.GGMLQuantizationType.Q4_0,
    gguf.GGMLQuantizationType.Q4_1,
    gguf.GGMLQuantizationType.Q4_K,
}

# Additionally promoted to GPTQ int8 (not left fp16) when --k-quants-to int8
# is passed -- see DECISION 1b. Q4_0 is deliberately NOT in this set: it
# keeps its bit-exact int4 fast path regardless of --k-quants-to (task
# requirement: "giu nguyen, dung pha").
K_QUANT_INT8_ELIGIBLE_TYPES = {
    gguf.GGMLQuantizationType.Q4_1,
    gguf.GGMLQuantizationType.Q4_K,
    gguf.GGMLQuantizationType.Q5_K,
    gguf.GGMLQuantizationType.Q6_K,
}


def classify_target_bits(gtype, k_quants_to: str):
    """GGUF block type + --k-quants-to mode -> GPTQ bit width to encode this
    tensor as, or None to keep it fp16. Q4_0 is always bits=4 (bit-exact
    fast path). In "int4" mode (default, matches the script's original
    behaviour byte-for-byte): Q4_1/Q4_K -> bits=4 (the ~5-15% RMS lossy
    generic path documented in the module docstring); Q5_K/Q6_K -> fp16
    (unsupported, unchanged). In "int8" mode: Q4_1/Q4_K/Q5_K/Q6_K all -> 8.
    """
    if gtype == gguf.GGMLQuantizationType.Q4_0:
        return 4
    if k_quants_to == "int8":
        return 8 if gtype in K_QUANT_INT8_ELIGIBLE_TYPES else None
    return 4 if gtype in (gguf.GGMLQuantizationType.Q4_1, gguf.GGMLQuantizationType.Q4_K) else None

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
    """All-zero-point qzeros (every nibble/byte == bits_zero_and_range(bits)[0],
    i.e. 8 for bits=4, 128 for bits=8). Inert at inference (see DECISION 1 --
    vLLM's Marlin kernel discards qzeros content when zero_points=False), but
    shape-correct and byte-correct for any tool that does read it."""
    pack_factor = 32 // bits
    assert n % pack_factor == 0
    zero_point, _, _ = bits_zero_and_range(bits)
    packed_val = 0
    for i in range(pack_factor):
        packed_val |= zero_point << (bits * i)
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


def quantize_symmetric_group(w_kn: np.ndarray, group_size: int, bits: int = BITS, n_grid: int = 9):
    """fp32 (K,N) -> GPTQ int<bits> symmetric (zero=2**(bits-1)), group_size
    along K.

    Plain min/max-derived scale, refined by a small clip-ratio grid search
    (shrinking the min/max envelope trades the rare outlier's error for
    better resolution on the bulk of the group -- standard RTN+clip
    baseline, uses only this tensor's own values, still calibration-free).
    Returns (qweight int32 packed (K//pack_factor,N), scales fp16
    (K//group_size,N), w_ref fp32 (K,N) dequantized reconstruction, for
    error reporting). `bits` defaults to the module's int4 constant for
    backward compatibility with existing call sites; pass bits=8 for the
    --k-quants-to int8 branch (see DECISION 1b).
    """
    zero_point, quant_min, quant_max = bits_zero_and_range(bits)
    k, n = w_kn.shape
    assert k % group_size == 0, f"k={k} not divisible by group_size={group_size}"
    num_groups = k // group_size
    wg = w_kn.reshape(num_groups, group_size, n)

    max_val = wg.max(axis=1, keepdims=True)
    min_val = wg.min(axis=1, keepdims=True)
    base_scale = np.maximum(np.abs(max_val / quant_max), np.abs(min_val / quant_min))
    base_scale = np.clip(base_scale, 1e-10, None)

    best_scale, best_mse = base_scale, None
    for ratio in np.linspace(1.0, 0.6, n_grid):
        s = np.clip(base_scale * ratio, 1e-10, None)
        q_try = np.clip(np.round(wg / s), quant_min, quant_max)
        mse = ((q_try * s - wg) ** 2).mean(axis=1, keepdims=True)
        if best_mse is None:
            best_mse, best_scale = mse, s
        else:
            better = mse < best_mse
            best_scale = np.where(better, s, best_scale)
            best_mse = np.where(better, mse, best_mse)
    scale = best_scale

    q = np.clip(np.round(wg / scale), quant_min, quant_max)
    w_ref = (q * scale).reshape(k, n).astype(np.float32)
    scales_out = scale.reshape(num_groups, n).astype(np.float16)
    q_biased = (q + zero_point).reshape(k, n).astype(np.int32)
    qweight = pack_rows(q_biased, bits, k, n)
    return qweight, scales_out, w_ref


def dequant_gptq_for_verification(qweight: np.ndarray, scales: np.ndarray, k: int, n: int, group_size: int, bits: int = BITS) -> np.ndarray:
    """Unpack + dequantize our own on-disk GPTQ tensors back to (K,N) fp32,
    independent of the encoder's internals -- the "read our own output
    back" half of the per-tensor error report. `bits` defaults to int4 for
    backward compatibility; pass bits=8 for an int8-packed tensor."""
    zero_point, _, _ = bits_zero_and_range(bits)
    q = unpack_rows(qweight, bits, k, n).astype(np.float32) - zero_point
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


# --------------------------------------------------------------------------
# BUG D FIX -- Qwen3.5's GDN (gated-delta-net) mixer tensors, hardcoded.
#
# This was the "unresolved gap" flagged in TASK I's LIMITATIONS (see the
# module docstring's DECISION 3 LIMITATIONS entry) and confirmed on a real
# Qwen3.5-2B GGUF checkpoint during Colab acceptance: vLLM raised
# "There is no module or parameter named 'layers.0.A_log' in Qwen3_5Model"
# (note: no "linear_attn." segment at all).
#
# Root cause, verified empirically (build_ggml_to_hf_map(gguf.MODEL_ARCH.
# QWEN35, ...) run directly against the installed gguf==0.19.0 package):
# the generic "shortest exact-suffix-hinted, else shortest overall"
# tie-break picks the WRONG candidate for every one of these ggml roles,
# not merely an ambiguous one that falls through to "unmapped.*" as the
# module docstring previously (incorrectly) assumed:
#
#     blk.0.ssm_a       -> model.layers.0.A_log                (WRONG; want ...linear_attn.A_log)
#     blk.0.ssm_dt.bias  -> model.layers.0.dt_proj.bias          (WRONG; want ...linear_attn.dt_bias)
#     blk.0.ssm_conv1d  -> model.layers.0.conv1d               (WRONG; want ...linear_attn.conv1d)
#     blk.0.ssm_norm    -> model.layers.0.mamba.norm            (WRONG; want ...linear_attn.norm)
#     blk.0.ssm_out     -> model.layers.0.out_proj              (WRONG; want ...linear_attn.out_proj)
#     blk.0.ssm_beta    -> model.layers.0.self_attn.b_proj      (WRONG; want ...linear_attn.in_proj_b)
#     blk.0.attn_qkv    -> model.layers.0.self_attn.qkv_proj    (WRONG; want ...linear_attn.in_proj_qkv)
#     blk.0.attn_gate   -> model.layers.0.self_attn.g_proj      (WRONG; want ...linear_attn.in_proj_z)
#
# (blk.0.ssm_alpha is the sole exception -- gguf's table has exactly one
# "model."-style candidate for SSM_ALPHA, "...linear_attn.in_proj_a", so it
# already resolves correctly; kept in the override table below anyway for
# completeness/uniformity, it's a no-op there.)
#
# The reason a purely "generic, architecture-agnostic" tie-break can never
# get these right: every one of gguf's own alias lists for these ggml
# roles (tensor_mapping.py) mixes Qwen3.5's real name in with SHORTER
# aliases from older/differently-shaped architectures (mamba-hf's bare
# "model.layers.{bid}.A_log", chatglm/persimmon's "...query_key_value",
# afmoe's "...self_attn.gate_proj", ...) that happen to share the same
# ggml tensor purpose -- "shortest wins" and "ends with a known modern
# suffix" both lose here, exactly as scripts/graft_gguf_gdn.py's own
# GGML_SUFFIX_FOR_HF_SUFFIX table already had to hardcode for the four
# in_proj_* tensors. This extends that same hardcode to the REMAINING GDN
# tensors gguf2marlin.py (unlike graft_gguf_gdn.py) also has to name:
# A_log, dt_bias, conv1d, norm, out_proj.
#
# Real HF names verified against vllm/model_executor/models/qwen3_5.py +
# .../mamba/gdn/qwen_gdn_linear_attn.py (same source scripts/
# transcode_gguf_to_gptq.py's PER_LAYER_TENSORS_LINEAR_ATTN table already
# hardcodes -- cross-checked against it here, not re-derived from scratch).
# Keyed by the RAW ggml tensor name's "blk.{bid}." tail (i.e. after
# stripping the block prefix) rather than by the post-split "base" name,
# because dt_bias needs special handling: llama.cpp stores it as a bare
# bias parameter "blk.{bid}.ssm_dt.bias" (see scripts/
# patch_gguf_tensor_mapping.py's docstring), but the real HF parameter name
# is "linear_attn.dt_bias" -- a single leaf name that already ends in
# "_bias", NOT "linear_attn.dt" + ".bias" (gguf2marlin's normal
# base+suffix concatenation, correct for genuine Linear .weight/.bias
# pairs, would produce the wrong "...dt.bias" here). Overriding on the
# full raw name sidesteps that split/concatenate step entirely for this
# tensor. A_log is similar in spirit (a bare parameter with NO suffix at
# all, "blk.{bid}.ssm_a" -- see scripts/patch_gguf_empty_suffix.py) though
# it doesn't hit the same concatenation bug; kept in this same table for
# uniformity.
# --------------------------------------------------------------------------
_QWEN35_ARCHES = frozenset(
    getattr(gguf.MODEL_ARCH, _name)
    for _name in ("QWEN35", "QWEN35MOE")
    if hasattr(gguf.MODEL_ARCH, _name)
)

_QWEN35_GDN_RAW_TAIL_TO_HF_SUFFIX = {
    "ssm_a": "linear_attn.A_log",  # bare, no ggml suffix at all
    "ssm_dt.bias": "linear_attn.dt_bias",  # bare bias, ggml suffix ".bias"
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",  # already correct generically; kept for uniformity
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}
_QWEN35_GDN_RAW_NAME_PATTERN = re.compile(
    r"^blk\.(\d+)\.(" + "|".join(re.escape(k) for k in _QWEN35_GDN_RAW_TAIL_TO_HF_SUFFIX) + r")$"
)


def qwen35_gdn_override_name(raw_ggml_name: str) -> str | None:
    """Full HF parameter name for a Qwen3.5 GDN tensor's raw ggml name (e.g.
    "blk.3.ssm_a" -> "model.layers.3.linear_attn.A_log"), or None if
    `raw_ggml_name` isn't one of the hardcoded GDN tensors above. Caller is
    responsible for only invoking this when `arch_enum` is Qwen3.5/Qwen3.5-MoE
    (see `_QWEN35_ARCHES`) -- these ggml roles are shared with other, unrelated
    architectures where this override would be wrong."""
    m = _QWEN35_GDN_RAW_NAME_PATTERN.match(raw_ggml_name)
    if m is None:
        return None
    bid, tail = m.group(1), m.group(2)
    return f"model.layers.{bid}." + _QWEN35_GDN_RAW_TAIL_TO_HF_SUFFIX[tail]


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


def _scheme_key(r: dict):
    """A record's quantization scheme for merge-pair comparison: the target
    bit width (4 or 8), or None/False for "kept fp16". Prefers the "scheme"
    key (bits or None) when present -- added for the --k-quants-to int8
    branch, where two merge-pair members can BOTH be "would_quantize=True"
    yet still be incompatible if one is int4 and the other int8 (different
    pack_factor => different K-packed width, same failure mode as one side
    being fp16, see module docstring). Falls back to the plain boolean
    "would_quantize" for callers (and the existing unit test) that don't
    supply "scheme" -- behaviourally identical to the original bool-only
    check in that case.
    """
    if "scheme" in r:
        return r["scheme"]
    return r["would_quantize"]


def apply_merge_pair_fixups(records: list[dict]) -> set[str]:
    """See KNOWN_MERGE_PAIRS / module docstring "MERGED-PROJECTION SCHEME
    CONSISTENCY". Given the pass-1 classification records (each a dict with
    at least "hf_base" and "would_quantize", optionally "scheme" -- see
    _scheme_key), returns the set of hf_base names that must be forced to
    fp16 because their merge-pair partner uses a different scheme (one
    quantized/other not, OR one int4/other int8). Factored out from main()
    so it's directly unit-testable without a real GGUF file or subprocess --
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
            if _scheme_key(r_a) != _scheme_key(r_b):
                print(f"[gguf2marlin] WARNING: merge pair ({r_a['hf_base']}, {r_b['hf_base']}) "
                      f"has mismatched schemes (scheme={_scheme_key(r_a)!r} vs "
                      f"{_scheme_key(r_b)!r}) -- forcing BOTH to fp16 so vLLM's "
                      f"merged-column loader sees a consistent shard scheme "
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
    ap.add_argument("--k-quants-to", choices=("int4", "int8"), default="int4",
                     help="default 'int4' -- Q4_1/Q4_K use the original ~5-15%%-RMS "
                          "symmetric int4 re-fit (see CORRECTION under DECISION 2), "
                          "Q5_K/Q6_K stay fp16, unchanged from before this flag existed. "
                          "'int8' -- see DECISION 1b: promotes Q4_1/Q4_K/Q5_K/Q6_K to a "
                          "GPTQ 8-bit (uint8b128) encoding instead (~0.5%% RMS target); "
                          "Q4_0 is unaffected either way (always its int4 bit-exact fast "
                          "path). Produces a MIXED-bits checkpoint using AutoGPTQConfig's "
                          "'dynamic' per-module override (base bits=4, promoted modules "
                          "get a 'dynamic' regex entry with bits=8).")
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
    records = []  # dict(hf_base, suffix, ggml_name, tensor, would_quantize, scheme, mapped)
    for t in reader.tensors:
        # BUG D FIX: Qwen3.5's GDN mixer tensors take priority over the
        # generic mapper -- see qwen35_gdn_override_name's docstring above
        # for why the generic tie-break gets every one of these wrong.
        override_name = (
            qwen35_gdn_override_name(t.name) if arch_enum in _QWEN35_ARCHES else None
        )
        if override_name is not None:
            hf_base, suffix = split_ggml_name(override_name)
            mapped = True
        else:
            base, suffix = split_ggml_name(t.name)
            hf_base = ggml_to_hf.get(base)
            mapped = hf_base is not None
            if not mapped:
                hf_base = f"unmapped.{base}"
        is_always_fp16 = any(hf_base.endswith(s) for s in ALWAYS_FP16_HF_SUFFIXES)
        is_2d = len(t.shape) == 2
        gtype_ = gguf.GGMLQuantizationType(t.tensor_type)
        scheme = (
            classify_target_bits(gtype_, args.k_quants_to)
            if (mapped and is_2d and not is_always_fp16)
            else None
        )
        would_quantize = scheme is not None
        records.append({
            "hf_base": hf_base, "suffix": suffix, "ggml_name": base,
            "tensor": t, "would_quantize": would_quantize, "scheme": scheme, "mapped": mapped,
        })

    # ---- pass 2: merge-pair scheme-consistency fixups ----------------------
    forced_fp16 = apply_merge_pair_fixups(records)

    # ---- pass 3: do the actual work ----------------------------------------
    out_tensors: dict[str, torch.Tensor] = {}
    quantized_modules, int8_modules, fp16_kept, unmapped_names = [], [], [], []
    error_report = []  # (hf_base, gguf_type_name, bits, rel_rms)

    for r in records:
        t = r["tensor"]
        hf_base, suffix = r["hf_base"], r["suffix"]
        hf_name = hf_base + suffix
        gtype = gguf.GGMLQuantizationType(t.tensor_type)
        quantize_this = r["would_quantize"] and hf_base not in forced_fp16
        bits = r["scheme"] if quantize_this else None

        if not r["mapped"]:
            unmapped_names.append(hf_base)

        if quantize_this:
            out_features, in_features = int(t.shape[-1]), int(t.shape[0])
            if gtype == gguf.GGMLQuantizationType.Q4_0 and group_size == 32 and bits == 4:
                q_biased, scales = unpack_q4_0_exact(t.data, out_features, in_features)
                qweight = pack_rows(q_biased.astype(np.int32), 4, in_features, out_features)
                w_ref = dequant_gptq_for_verification(qweight, scales, in_features, out_features, group_size, bits=4)
                w_pre = np.ascontiguousarray(gguf_dequantize(t.data, gtype), dtype=np.float32).T
            else:
                w_pre_out_in = np.ascontiguousarray(gguf_dequantize(t.data, gtype), dtype=np.float32)
                w_pre = w_pre_out_in.T  # (K, N)
                qweight, scales, w_ref = quantize_symmetric_group(w_pre, group_size, bits=bits)

            k_dim, n_dim = in_features, out_features
            qzeros = make_symmetric_qzeros(k_dim // group_size, n_dim, bits)
            g_idx = (np.arange(k_dim, dtype=np.int32) // group_size)

            out_tensors[hf_name.replace(".weight", ".qweight")] = torch.from_numpy(np.ascontiguousarray(qweight))
            out_tensors[hf_name.replace(".weight", ".qzeros")] = torch.from_numpy(np.ascontiguousarray(qzeros))
            out_tensors[hf_name.replace(".weight", ".scales")] = torch.from_numpy(np.ascontiguousarray(scales))
            out_tensors[hf_name.replace(".weight", ".g_idx")] = torch.from_numpy(np.ascontiguousarray(g_idx))
            quantized_modules.append(hf_base)
            if bits == 8:
                int8_modules.append(hf_base)

            denom = float(np.sqrt((w_pre.astype(np.float64) ** 2).mean())) or 1.0
            rel_rms = float(np.sqrt(((w_ref.astype(np.float64) - w_pre.astype(np.float64)) ** 2).mean())) / denom
            error_report.append((hf_base, gtype.name, bits, rel_rms))
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
        raw_cfg = json.loads(Path(args.hf_config).read_text(encoding="utf-8"))
        cfg = raw_cfg.get("text_config", raw_cfg)
        # Minimal sanity check (Bug C in this repo's auto-marlin hook, see
        # scripts/patch_gguf_auto_marlin.py's module docstring): a caller
        # passing --hf-config is asserting "this is the REAL upstream
        # config", so the emitted model_type should be a real transformers
        # id, not a GGUF-parser-internal alias like this script's own
        # generic fallback below writes. Warn (do not block -- a real
        # config genuinely named this way is possible in principle and
        # this script has no authoritative registry to check against).
        _hf_model_type = cfg.get("model_type")
        if not _hf_model_type:
            print(
                "[gguf2marlin] WARNING: --hf-config "
                f"{args.hf_config!r} has no 'model_type' field -- vLLM's "
                "stock (non-gguf) config loader will not be able to select "
                "an architecture from this config.json.", file=sys.stderr,
            )
        elif _hf_model_type in _GGUF_INTERNAL_ONLY_MODEL_TYPES:
            print(
                f"[gguf2marlin] WARNING: --hf-config's model_type="
                f"{_hf_model_type!r} looks like a GGUF-parser-internal "
                "architecture alias (see vllm_gguf_plugin's own model_type "
                "normalization), not a real transformers model_type -- "
                "expected something like "
                f"{_GGUF_INTERNAL_ONLY_MODEL_TYPES[_hf_model_type]!r} for "
                "this architecture. Double-check this is really the "
                "upstream HF config.json and not a copy of a GGUF-derived "
                "one.", file=sys.stderr,
            )

        # BUG (fallback architectures) FIX -- see resolve_architectures's
        # docstring / the module-level comment above
        # _MODEL_TYPE_TO_CAUSAL_LM_CLASS. Previously `cfg = cfg.get(
        # "text_config", cfg)` above silently dropped a top-level
        # 'architectures' field whenever text_config existed but had none
        # of its own (Qwen/Qwen3.5-2B's real config.json does exactly
        # this), leaving the emitted config.json without one at all and
        # sending vLLM down its own "Qwen3_5TextModel" AutoModel fallback
        # instead of the CausalLM class. Always resolve and (re-)write
        # 'architectures' explicitly instead of relying on it having
        # survived the text_config swap above.
        architectures, is_guessed_arch = resolve_architectures(raw_cfg, cfg)
        if architectures is not None:
            had_it_already = bool(cfg.get("architectures"))
            cfg["architectures"] = architectures
            if is_guessed_arch:
                print(
                    "[gguf2marlin] WARNING: --hf-config has no 'architectures' "
                    f"field (checked both the top level and text_config) and "
                    f"model_type={_hf_model_type!r} is not in this script's "
                    "known model_type -> class table (_MODEL_TYPE_TO_CAUSAL_LM_"
                    f"CLASS) -- GUESSING architectures={architectures!r} from "
                    "model_type by naming convention alone (CamelCase each "
                    "'_'-segment + 'ForCausalLM'). This is UNVERIFIED and may "
                    "be wrong (see _guess_causal_lm_class_name's docstring for "
                    "why this can't be done reliably in general) -- if vLLM "
                    "can't find this class, edit the output config.json's "
                    "'architectures' field by hand with the real class name.",
                    file=sys.stderr,
                )
            elif not had_it_already:
                print(
                    f"[gguf2marlin] NOTE: --hf-config had no 'architectures' on "
                    f"text_config itself; inferred architectures={architectures!r} "
                    f"from {'the top-level config' if raw_cfg.get('architectures') else f'model_type={_hf_model_type!r}'} "
                    "instead of silently omitting the field (previous behaviour "
                    "-- see the BUG (fallback architectures) note in the module "
                    "docstring).", file=sys.stderr,
                )
        elif not _hf_model_type:
            pass  # already warned above (no model_type field at all either).
    else:
        token_embd = next((t for t in reader.tensors if t.name == "token_embd.weight"), None)
        if token_embd is not None:
            cfg["vocab_size"] = int(token_embd.shape[-1])
            cfg["hidden_size"] = int(token_embd.shape[0])
        cfg["num_hidden_layers"] = n_blocks
        cfg["tie_word_embeddings"] = tie_word_embeddings
        cfg["model_type"] = arch_str

    quant_cfg = {
        "quant_method": "gptq",
        "bits": BITS,  # base/majority scheme -- int4. Per-module overrides below.
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
    }
    if int8_modules:
        # See DECISION 1b: AutoGPTQConfig's "dynamic" regex-keyed per-module
        # override (vllm/model_executor/layers/quantization/utils/
        # gptq_utils.py override_config()/get_dynamic_override(), invoked
        # per-layer from get_linear_quant_method()) is what makes a
        # per-module bits=8 override on an otherwise-bits=4 checkpoint a
        # real, vLLM-served mixed-precision GPTQ config -- not a
        # convention this script invented.
        quant_cfg["dynamic"] = {
            f"+:{re.escape(hf_base)}$": {"bits": 8} for hf_base in int8_modules
        }
    cfg["quantization_config"] = quant_cfg
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- manifest.json -- per-tensor scheme + RMS error, machine-readable ---
    # (consumed directly by scripts/patch_gguf_auto_marlin.py's cache-hit
    # check/logging -- see that script's module docstring).
    manifest = {
        "source_gguf": Path(args.gguf_path).name,
        "group_size": group_size,
        "k_quants_to": args.k_quants_to,
        "quantized_modules": quantized_modules,
        "int8_modules": int8_modules,
        "fp16_kept": fp16_kept,
        "unmapped": unmapped_names,
        "tensors": [
            {"module": name, "gguf_type": gtype_name, "bits": bits, "rel_rms_error": rel_rms}
            for name, gtype_name, bits, rel_rms in error_report
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[gguf2marlin] wrote {out_dir} -- {len(quantized_modules)} modules quantized "
          f"({len(int8_modules)} of them int8 g{group_size}, rest int4 g{group_size}), "
          f"{len(fp16_kept)} kept fp16, {len(unmapped_names)} unmapped", file=sys.stderr)

    _print_report(args, quantized_modules, int8_modules, fp16_kept, unmapped_names, error_report, args.hf_config is None)
    return 0


def _print_report(args, quantized_modules, int8_modules, fp16_kept, unmapped_names, error_report, config_is_best_effort):
    print("\n" + "=" * 72)
    print("VERIFICATION REPORT")
    print("=" * 72)
    print(f"quantized (GPTQ, group_size={args.group_size}): {len(quantized_modules)} "
          f"({len(int8_modules)} int8, {len(quantized_modules) - len(int8_modules)} int4)")
    print(f"kept fp16: {len(fp16_kept)}")
    print(f"unmapped (no generic HF name found, written under 'unmapped.*'): "
          f"{len(unmapped_names)}")
    if unmapped_names:
        print(f"    {unmapped_names[:12]}{' ...' if len(unmapped_names) > 12 else ''}")
        print("    NOTE: these tensors will NOT be visible to vLLM's stock HF loader "
              "under any real parameter name -- see LIMITATIONS in the module docstring.")

    if error_report:
        print(f"\nper-tensor relative RMS error (GPTQ-dequant vs GGUF-dequant), by source type + bits:")
        by_type: dict[tuple[str, int], list[float]] = {}
        for _name, gtype_name, bits, rel_rms in error_report:
            by_type.setdefault((gtype_name, bits), []).append(rel_rms)
        for (gtype_name, bits), errs in sorted(by_type.items()):
            errs_np = np.array(errs)
            print(f"    {gtype_name:8s} int{bits}  n={len(errs):4d}  mean={errs_np.mean():.6f}  "
                  f"max={errs_np.max():.6f}  min={errs_np.min():.6f}")
        worst = sorted(error_report, key=lambda x: -x[3])[:5]
        print("  worst 5 tensors:")
        for name, gtype_name, bits, rel_rms in worst:
            print(f"    {rel_rms:.6f}  {gtype_name:6s} int{bits}  {name}")

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
