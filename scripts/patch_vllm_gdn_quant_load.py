"""
Enables quantized GDN (gated-delta-net) input projections for Qwen3.5 by
fixing two bugs in vLLM's merged-column-parallel weight loader that only
show up when the shards being merged are compressed-tensors PACKED
parameters, not plain fp16 tensors.

--- The crash this fixes ---

Qwen3.5's GDN attention (vllm/model_executor/layers/mamba/gdn/
qwen_gdn_linear_attn.py, QwenGatedDeltaNetAttention.__init__) fuses four
separate HF checkpoint projections into two vLLM MergedColumnParallelLinear
params:

    in_proj_qkv + in_proj_z -> in_proj_qkvz   (output_sizes=[key,key,val,val])
    in_proj_b   + in_proj_a -> in_proj_ba     (output_sizes=[num_v_heads]*2)

via hf_to_vllm_mapper (vllm/model_executor/models/qwen3_5.py):
    ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
    ".in_proj_z":   (".in_proj_qkvz", 3),
    ".in_proj_b":   (".in_proj_ba", 0),
    ".in_proj_a":   (".in_proj_ba", 1),

When every one of those four HF modules is fp16, the stock loader
(MergedColumnParallelLinear.weight_loader_v2 in
vllm/model_executor/layers/linear.py) works fine: each shard is a plain
tensor, sliced and copied by output-dim offset.

Quantizing any of the four with compressed-tensors (the AWQ/GPTQ W4A16
path) breaks this, confirmed by this project's own quantize_gptq_9b.py
(and its AWQ v2 predecessor), whose docstring records the exact failure:

    "Quantizing ANY of GDN's four input projections, at ANY bit width or
    algorithm, breaks vLLM's merged-parameter load path for this
    architecture (v2's AssertionError in load_merged_column_weight)"

Root cause, traced statically against this fork's installed source (no
GPU on this dev machine -- see STATUS.md -- so this was verified by
reading the loader/parameter/scheme code end-to-end, not by reproducing
the traceback):

1. CompressedTensorsWNA16.create_weights (vllm/model_executor/layers/
   quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py)
   declares:

       weight_packed = PackedvLLMParameter(input_dim=1, output_dim=0,
                                            packed_dim=1, ...)

   i.e. packing lives on dim 1 (K, the input/hidden dim -- multiple int4
   values packed per int32 word along the REDUCTION axis), while the
   merged-column loader shards along dim 0 (N, the output/channel axis).
   packed_dim (1) != output_dim (0), so packing is ALONG K, NOT N.
   Concatenating independently-quantized shards along N is therefore
   mathematically valid PROVIDED every shard packs K identically, i.e.
   shares the same num_bits (which sets packed_K = ceil(K * num_bits/32),
   independent of N) and the same group_size (which sets weight_scale's
   per-group column count, K/group_size, also independent of N). See the
   accompanying unit simulation for a numerical proof of the concat math.

2. The actual crash: MergedColumnParallelLinear.weight_loader_v2's
   single-shard branch calls
   ``param.load_merged_column_weight(loaded_weight=..., shard_offset=...,
   shard_size=...)``, which for a PackedvLLMParameter resolves to
   ``_ColumnvLLMParameter.load_merged_column_weight``
   (vllm/model_executor/parameter.py:156), ending in:

       assert param_data.shape == loaded_weight.shape   # parameter.py:175

   shard_offset/shard_size here are computed purely from
   ``self.output_sizes`` (raw N units -- correct, since packed_dim !=
   output_dim means no packed_factor scaling applies). But
   ``loaded_weight`` is whatever weight_packed tensor the checkpoint
   actually stored for that ONE HF sub-module (e.g. in_proj_z's own
   weight_packed), whose K-packed width was computed independently by
   whatever quantized *that* sub-module. If in_proj_qkv/in_proj_z (or
   in_proj_b/in_proj_a) were quantized with DIFFERENT num_bits, their
   packed_K values differ and the narrowed shapes mismatch -> the
   AssertionError quantize_gptq_9b.py's docstring describes. This is
   fixable (same scheme => valid concat) but was previously an opaque,
   undiagnosable AssertionError deep in parameter.py with no indication
   of *why*. This patch turns it into an actionable RuntimeError.

3. A second, silent (non-crashing) bug found while tracing the same
   method: ``weight_shape`` (a plain ``BasevLLMParameter`` -- the 2-elem
   int64 [N, K] metadata tensor compressed-tensors kernels use to know
   the logical, pre-packing shape for Marlin/Machete repacking) is NOT a
   ``_ColumnvLLMParameter``, so it has no ``output_dim`` and no per-shard
   narrowing logic. weight_loader_v2 dispatches it through
   ``BasevLLMParameter.load_merged_column_weight``, which ignores all
   shard_offset/shard_size kwargs and does a full ``_assert_and_load``
   (parameter.py:105-106). Since every shard's own weight_shape is ALSO
   shape (2,), the assert trivially passes -- but each shard call
   *overwrites the whole param*, so after all shards load,
   ``layer.weight_shape`` silently holds whichever sub-module's [N, K]
   was loaded LAST (e.g. in_proj_z's [value_dim, hidden] instead of the
   true merged in_proj_qkvz shape [key_dim*2+value_dim*2, hidden]).
   Whatever kernel consumes weight_shape during
   ``process_weights_after_loading`` (Marlin repack reads it to reshape
   weight_packed) would then either crash on a shape it can't reconcile
   with weight_packed's real size, or -- worse -- silently repack against
   the wrong logical shape. The fix: weight_shape is fully determined by
   the layer itself (``output_size_per_partition`` /
   ``input_size_per_partition``, both set once by ``create_weights``
   before any weight loads), so write it directly from the layer instead
   of trusting per-shard checkpoint values at all.

--- The fix ---

Both bugs are patched inside
``MergedColumnParallelLinear.weight_loader_v2``:

* A ``weight_shape``-shaped BasevLLMParameter (int64, numel()==2) is
  special-cased before the normal shard/tuple dispatch: instead of
  copying whatever shard tensor arrives, its data is always set to
  ``[self.output_size_per_partition, self.input_size_per_partition]``.
  Idempotent across repeated shard calls (every call writes the same
  correct value), so ordering across in_proj_qkv/in_proj_z (or
  in_proj_b/in_proj_a) shards no longer matters.

* The final single-shard ``param.load_merged_column_weight(...)`` call
  (the one that raises the AssertionError from parameter.py:175 for
  packed params) is wrapped in a try/except. When the param is a packed
  compressed-tensors parameter whose packed dim is NOT the output dim
  being sharded here (i.e. exactly the weight_packed case above) and the
  stock assert fires, it is re-raised as a RuntimeError naming the layer,
  the shard, and the concrete likely cause (mismatched num_bits/
  group_size across the sub-projections folded into this fused param),
  instead of the opaque ``assert param_data.shape == loaded_weight.shape``.
  Any other AssertionError (unrelated params/paths) is re-raised
  unchanged -- this patch does not touch the fp16 path or any other
  parameter type.

Deliberately NOT attempted: repacking mismatched-scheme shards. The
concat-only approach requires every sub-projection merged into one vLLM
param to share num_bits and group_size; see
scripts/quantize_gptq_9b.py's --quantize-gdn flag for the quantize-side
half of this constraint (in_proj_qkv+in_proj_z share one scheme,
in_proj_b+in_proj_a share a second scheme -- each pair internally
consistent, the two pairs may still differ from each other since they
land in different vLLM params).
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: merged-column loader for quantized GDN packed shards ---"

ANCHOR = '''    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if isinstance(param, PerTensorScaleParameter):
                if isinstance(loaded_shard_id, tuple):
                    for idx in loaded_shard_id:
                        param.load_merged_column_weight(
                            loaded_weight=loaded_weight, shard_id=idx
                        )
                else:
                    # When weights are already fused on disk (e.g. Phi-3's
                    # gate_up_proj), there is only a single scale for the
                    # entire fused matrix. Fill all slots with this scale
                    # to ensure that any subsequent reduction (like .max())
                    # works correctly while preserving the parameter shape.
                    for idx in range(param.data.shape[0]):
                        param.load_merged_column_weight(
                            loaded_weight=loaded_weight, shard_id=idx
                        )
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                output_sizes = [
                    adjust_block_scale_shard(weight_block_size, size, 0)[0]
                    for size in (output_sizes or self.output_sizes)
                ]
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return

        assert loaded_shard_id < len(self.output_sizes)

        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        shard_offset //= self.tp_size
        shard_size //= self.tp_size

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )
'''

PATCH = '''    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        ''' + PATCH_MARKER + '''
        # compressed-tensors' "weight_shape" is a 2-elem [N, K] metadata
        # tensor (a plain BasevLLMParameter with no output_dim), not a
        # sharded weight. The stock path (BasevLLMParameter
        # .load_merged_column_weight -> _assert_and_load) blindly
        # overwrites the whole param with whichever shard loads last,
        # silently corrupting it to that ONE sub-projection's own
        # (unmerged) shape instead of the true fused [sum(output_sizes),
        # input_size]. The correct merged shape is always known from the
        # layer itself once create_weights has run, so set it directly
        # instead of trusting per-shard checkpoint values.
        if (
            type(param) is BasevLLMParameter
            and param.data.dtype == torch.int64
            and param.data.numel() == 2
        ):
            merged_shape = torch.tensor(
                [self.output_size_per_partition, self.input_size_per_partition],
                dtype=param.data.dtype,
                device=param.data.device,
            )
            param.data.copy_(merged_shape)
            return

        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if isinstance(param, PerTensorScaleParameter):
                if isinstance(loaded_shard_id, tuple):
                    for idx in loaded_shard_id:
                        param.load_merged_column_weight(
                            loaded_weight=loaded_weight, shard_id=idx
                        )
                else:
                    # When weights are already fused on disk (e.g. Phi-3's
                    # gate_up_proj), there is only a single scale for the
                    # entire fused matrix. Fill all slots with this scale
                    # to ensure that any subsequent reduction (like .max())
                    # works correctly while preserving the parameter shape.
                    for idx in range(param.data.shape[0]):
                        param.load_merged_column_weight(
                            loaded_weight=loaded_weight, shard_id=idx
                        )
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                output_sizes = [
                    adjust_block_scale_shard(weight_block_size, size, 0)[0]
                    for size in (output_sizes or self.output_sizes)
                ]
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return

        assert loaded_shard_id < len(self.output_sizes)

        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        shard_offset //= self.tp_size
        shard_size //= self.tp_size

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        try:
            param.load_merged_column_weight(
                loaded_weight=loaded_weight,
                shard_id=loaded_shard_id,
                shard_offset=shard_offset,
                shard_size=shard_size,
                tp_rank=self.tp_rank,
            )
        except AssertionError as e:
            packed_dim = getattr(param, "packed_dim", None)
            output_dim = getattr(param, "output_dim", None)
            if packed_dim is not None and packed_dim != output_dim:
                # Packed compressed-tensors weight (e.g. weight_packed)
                # whose packing lives on the INPUT dim (packed_dim=1)
                # rather than the output dim (0) being sharded here.
                # Output-channel concatenation of such shards is only
                # valid when every sub-projection folded into this fused
                # param (Qwen3.5's in_proj_qkv/in_proj_z -> in_proj_qkvz,
                # or in_proj_b/in_proj_a -> in_proj_ba) was quantized with
                # the IDENTICAL scheme: same num_bits (-> same packed K,
                # since packed_K = ceil(K * num_bits / 32) does not depend
                # on N) and same group_size (-> same weight_scale
                # per-group column count). A shape mismatch here means the
                # checkpoint mixed schemes across one merged pair, which
                # this loader can only concatenate, not repack.
                raise RuntimeError(
                    f"{getattr(self, 'prefix', '<unknown layer>')}: cannot "
                    f"merge quantized shard {loaded_shard_id} into this "
                    "fused column-parallel layer -- its packed weight "
                    "shape does not match the other shard(s) already "
                    "loaded. This almost always means the sub-projections "
                    "folded into this parameter (e.g. Qwen3.5's "
                    "in_proj_qkv/in_proj_z or in_proj_b/in_proj_a) were "
                    "quantized with DIFFERENT num_bits and/or group_size. "
                    "The merged-column loader can only concatenate packed "
                    "compressed-tensors shards along the output (N) "
                    "dimension when every shard shares the exact same "
                    "quantization scheme; it cannot repack mismatched "
                    "input-dim (K) packing. Re-quantize so every "
                    "sub-projection merged into a single vLLM parameter "
                    "uses the same scheme/group_size (see "
                    "scripts/quantize_gptq_9b.py --quantize-gdn)."
                ) from e
            raise
'''

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm/model_executor/layers/linear.py")
if not matches:
    raise SystemExit(f"vllm/model_executor/layers/linear.py not found under {site_packages}")
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
elif ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {path}; vllm source may have changed")
else:
    src = src.replace(ANCHOR, PATCH, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
