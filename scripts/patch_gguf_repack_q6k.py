"""
Opt-in: re-encode Q6_K (and optionally Q5_K) tensors to Q8_0 at load time.

Kernel profiling a GGUF Qwen3.5 decode step on L4 (sm89) shows quantised
matmul kernels dominate CUDA time end to end (86.6%), and inside that,
q6_k_gemm_kernel ALONE accounts for 34.3% -- despite the checkpoint being
labelled "Q4_K_M". K-quant "*_M"/"*_L" mixes are not single-format: llama.cpp's
quantizer keeps a handful of shape/sensitivity-selected tensors at Q6_K (and
some at Q5_K) even inside a nominally-Q4_K_M file, and Q6_K's 210-byte block
layout is the most expensive of the family to decode on this Triton fused
path -- STATUS.md's whole-model format sweep (L4, decode, conc 32) measured
Q6_K at 704 tok/s vs Q8_0 at 839 vs Q4_K_M at 872, i.e. Q8_0 beats Q6_K despite
reading ~30% more bytes off VRAM, because Q8_0's flat (scale, 32x int8) layout
decodes far more cheaply than Q6_K's nested-scale block. q5_k_gemm is the next
largest slice, at 20.2%.

So: re-encode any Q6_K/Q5_K tensor to Q8_0 once, at load time, and pay a small
one-off VRAM cost on just those tensors in exchange for routing every future
matmul on them through the fast kernel -- targeting more than a third of
measured decode CUDA time. This is the same dequant-then-gguf.quants.quantize
round trip patch_gguf_qwen35_transforms.py already performs for
linear_attn.out_proj (gguf-py's quantize_blocks only implements F16/F32/Q8_0;
it cannot encode any K-quant, which is exactly why Q8_0 is the re-encode
target and not, say, Q6_K itself).

Mechanics (see patch_gguf_qwen35_transforms.py's HELPERS for the established
pattern this reuses): vllm_gguf_plugin's GGUF loader feeds each quantised
module through transform_weight(hf_name, weight) as two separate rows, tag
before data:

    "<module>.qweight_type"   scalar tensor holding the ggml type as an int
    "<module>.qweight"        packed uint8 bytes

Unquantised params arrive as "<module>.weight" and are never touched here.
On a ".qweight_type" row whose value is in the configured repack set, this
records module-base -> original ggml type in a local dict and returns the tag
rewritten to Q8_0 (torch.full_like(weight, 8)) so vllm's loader allocates and
labels the parameter as Q8_0 from the start. On the following ".qweight" row
for a module recorded that way, it dequantises with gguf.quants.dequantize
using the *original* type and re-encodes with gguf.quants.quantize(...,
Q8_0), then returns the repacked bytes.

Ordering / composition with patch_gguf_qwen35_transforms.py: that patch's
transform_weight code is anchored on the docstring and runs FIRST; its
out_proj branch may itself already dequantise+re-encode a K-quant tensor to
Q8_0 (for the V-head untile permutation) before this hook ever sees it. This
hook is anchored on and inserted directly after the tail of that patch's own
inserted code (the qwen3.5 norm-weight branch), so by construction it always
runs after it, operating on the possibly-already-transformed weight. It stays
correct without needing to special-case out_proj: when qwen35's branch has
already rewritten a tag to Q8_0, this hook sees value 8 on the ".qweight_type"
row, which is never in the repack set, so it does not record that module and
therefore does nothing on the matching ".qweight" row either -- "skip any
module whose tag we did not rewrite ourselves" is the whole rule, enforced by
only ever consulting our own record, never vllm_gguf_plugin's or qwen35's.

Shard safety: transform_weight sees each raw GGUF tensor row before vllm
assembles fused-module shards (QKVParallelLinear, MergedColumnParallelLinear,
...), same as patch_gguf_qwen35_transforms.py -- so a per-row, per-GGUF-tensor
repack here never has to reason about shard boundaries; it is invisible to
(and runs before) the shard-assembly code entirely.

Env var, CUSTOM_VLLM_GGUF_REPACK, unset by default (no-op, zero overhead --
the call site check happens before the hook function is even invoked):

    CUSTOM_VLLM_GGUF_REPACK=1            repack Q6_K only (the default set)
    CUSTOM_VLLM_GGUF_REPACK=q6_k         same, spelled out
    CUSTOM_VLLM_GGUF_REPACK=q6_k,q5_k    repack both Q6_K and Q5_K tensors

Must be listed in setup_env.sh AFTER both patch_gguf_qwen35_transforms.py and
patch_gguf_hybrid_dispatch.py: the former for the anchor this patch depends
on (see below -- it deliberately fails loudly rather than silently mis-order
if that patch has not run yet), the latter because it patches a different
file (quantization/linear.py) and has no ordering interaction, but grouping
every load-time GGUF patch after the runtime-dispatch ones keeps the list's
"static rewrite, then runtime flags" shape.

Embedding/lm_head exclusion (post L4 crash fix):
An L4 run with CUSTOM_VLLM_GGUF_REPACK=1 crashed at engine-core init inside
the torch.compile'd embedding op, vocal_embeds.py:_apply_gguf_embedding's
"assert hidden_size == qweight.shape[1] // type_size * block_size". Both the
embedding consumer (quantization/vocal_embeds.py) and the linear consumer
(quantization/linear.py) read weight_type from the exact same place --
layer.qweight_type.weight_type, populated uniformly by
_store_gguf_weight_type() in quantization/params.py -- and
weight_utils.py's gguf_quant_weights_iterator_multi is a plain generator
that always yields a tensor's ".qweight_type" row immediately before its
".qweight" row with no possibility of interleaving from another tensor, so
the "tag before data" ordering this hook (and qwen35's) relies on is sound
in general. Re-encoding one tensor's bytes in isolation was also verified
byte-consistent (Q6_K -> dequant -> Q8_0 requantize preserves element
count exactly). None of that, however, guarantees the embedding/tied-lm_head
tensor's tag and data stay paired through every consumer that can touch it
outside this hook's own bookkeeping -- and embedding is the one tensor class
with anything else in play (weight tying, vocab-embedding-specific load
path in params.py's _gguf_embedding_weight_loader). The embedding op is also
the only consumer that self-checks its shape (the assert above); the linear
path's _fused_mul_mat_gguf (linear.py) recomputes its dequant width from the
very same (qweight.shape[1], type_size) pair with no cross-check against the
real hidden size at all -- so an equivalent tag/data desync on an ordinary
Q6_K linear tensor would not raise this assert; it would either blow up
later with an unrelated matmul-shape RuntimeError in `x @ weight.T`, or, if
the miscalculated width happened to coincide, silently compute garbage. And
because engine-core crashes on the very first embedding op before any
decoder linear layer's matmul is ever exercised, this run is not evidence
either way about whether linear tensors hit the same class of problem.

Given that, and since matmul speed is the entire point of this repack and
embeddings/lm_head gain nothing from it (embedding is a gather, not a
matmul; a tied lm_head is never repacked as a distinct tensor since it
shares the embedding module instead of loading its own qweight/qweight_type
pair -- see GGUFEmbeddingMethod.tie_weights), the fix removes token
embedding and (untied) lm_head tensors from repack scope entirely by name,
both by GGUF-native name (token_embd.weight / output.weight, when
gguf_to_hf_name_map is None and transform_weight sees raw GGUF names) and by
the HF-mapped name vllm_gguf_plugin's default adapter normally hands
transform_weight (model.embed_tokens.weight / lm_head.weight). This closes
the crash regardless of the exact mechanism behind the embedding-specific
desync, without touching any plugin source directly (this script only ever
edits its own generated PATCH/HELPERS text).

HARD GUARD -- L4 linear-layer crash, same desync family, repack disabled at
runtime (2026-08-09):
An L4 run with CUSTOM_VLLM_GGUF_REPACK=1 stacked on top of the three-way
champion config (CUSTOM_VLLM_GGUF_HYBRID=1 + CUSTOM_VLLM_GGUF_TRITON_MID=1,
_C_gguf CUDA extension active) got past the embedding fix above -- model
load succeeded -- and then crashed during warmup, first prefill step:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied
    (16384x6144 and 7936x2048)

16384 = the prefill chunk's row count (x.shape[0], well over the hybrid
dispatch's 1024-row threshold, so this went through the dequant+cuBLAS
branch shared by patch_gguf_hybrid_dispatch.py / patch_gguf_prefer_dequant.py
and the plugin's own stock DEQUANT_TYPES branch -- all three compute
`shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)`
from `gguf.GGML_QUANT_SIZES[qweight_type]`). 2048 is Qwen3.5-2B's
hidden_size; 6144 is its intermediate_size (confirmed from
transcode_gguf_to_gptq.py's own worked example, `blk.0.ffn_gate.weight,
ne=[2048,6144]`), so mat1 (x) is the correct (16384, 6144) activation
entering a down_proj matmul -- and down_proj is exactly llama.cpp's
Q4_K_M convention for an always-Q6_K tensor (alongside attn.wv), i.e.
exactly this patch's target. mat2 being (7936, 2048) means the dequant
branch decided down_proj's dequantised weight had K=7936, not 6144 --
proof follows from GGML_QUANT_SIZES arithmetic done independently of any
plugin internals:

    Q8_0:  block_size=32,  type_size=34   (gguf-py GGML_QUANT_SIZES[8])
    Q6_K:  block_size=256, type_size=210  (gguf-py GGML_QUANT_SIZES[14])

    correct repacked bytes for K=6144 in Q8_0: 6144 // 32 * 34 = 6528
    those 6528 bytes misread with Q6_K's (block_size, type_size):
        6528 // 210 * 256 = 7936          <-- exactly the crash's mat2 K

That is not a rounding coincidence; it is the unique arithmetic signature
of "the qweight BYTES were correctly repacked to Q8_0 width for K=6144,
but the qweight_type TAG consumed at the matmul call site still reads
Q6_K (14) instead of the rewritten Q8_0 (8)" -- a tag/data desync in the
same family as the embedding one above, now hit on an ordinary linear
layer instead of the embedding gather.

Where exactly the tag goes stale could not be pinned to one line by static
reading of the plugin alone. `_custom_vllm_gguf_repack()`'s own
bookkeeping (tag row records original type + rewrites to 8, data row pops
the record and re-encodes to match) is self-consistent for a standalone,
unsharded module like down_proj, and vllm_gguf_plugin's own generator
(`gguf_quant_weights_iterator_multi`) guarantees a tensor's ".qweight_type"
row is always yielded immediately before its ".qweight" row with no
interleaving -- so the desync has to be happening downstream, inside
vLLM/plugin machinery this repo does not vendor: GGUFWeightTypeParameter's
materialization in quantization/params.py copies `weight_type` forward
from an "uninitialized" parameter to a real one
(`_materialize_gguf_weight_type_parameter`), `apply()` in
quantization/linear.py reads it from two different attributes depending on
whether the layer is sharded (`layer.qweight_type.weight_type` vs.
`.shard_weight_type[idx]`), and separately `_create_padded_weight_param`
can swap `layer.qweight` for a brand-new padded Parameter object without
touching `qweight_type` at all. Any of those is a plausible point for a
byte-for-byte-correct repack to end up paired with a stale type integer,
and confirming which one requires tracing an actual GPU run step by step
-- this dev environment has no GPU (see STATUS.md), so that tracing cannot
be done and verified here.

Per this script's own logic, "never let it produce silent garbage" beats
"ship an unverified fix that only looks right": if the crash above -- a
loud shape error -- is possible at all, the same class of desync feeding
a shape that happens to still be *mm-compatible* would silently produce
wrong numbers instead, on any consumer that reads qweight_type (dequant,
but also the CUDA/Triton mmq/mmvq kernels this repack exists to speed up
in the first place -- they take qweight_type as a plain integer argument
too, with no shape self-check to catch a desync the way `x @ weight.T`
did here). So the guard below is unconditional, not just "when hybrid/
dequant is also on": CUSTOM_VLLM_GGUF_REPACK now refuses to run at all,
everywhere, until each qweight_type consumer above is audited or an
assertion is added at every consumer site that qweight's byte width
matches GGML_QUANT_SIZES[qweight_type] for the type actually observed
there, and that assertion is confirmed to never fire under GPU load.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: repack Q6_K/Q5_K tensors to Q8_0 at load time ---"

# This exact text is the tail of TRANSFORM_PATCH in patch_gguf_qwen35_transforms.py.
# It only exists in base.py once that patch has run, which is exactly the
# ordering guarantee this patch needs: anchoring here (rather than on some
# anchor also present in the pristine file) makes a mis-ordered setup_env.sh
# fail loudly at patch time instead of silently composing in the wrong order.
ANCHOR = (
    '        elif _QWEN35_CFG is not None and hf_name.endswith("norm.weight"):\n'
    "            # llama.cpp writes 1 + w for every zero-centred RMSNorm.\n"
    "            weight = weight - 1.0\n"
)
PATCH = (
    ANCHOR
    + f"        {PATCH_MARKER}\n"
    "        if _CUSTOM_VLLM_GGUF_REPACK_TYPES:\n"
    "            weight = _custom_vllm_gguf_repack(hf_name, weight)\n"
)

HELPERS = f'''

{PATCH_MARKER}
import os as _custom_vllm_repack_os

_CUSTOM_VLLM_REPACK_NAME_TO_TYPE = {{"q5_k": 13, "q6_k": 14}}  # GGMLQuantizationType

# Token-embedding / lm_head tensors never benefit from a matmul-oriented
# repack (embedding is a gather, not a matmul) and are excluded from repack
# scope entirely -- see module docstring "Embedding/lm_head exclusion" for
# why. Matches both raw GGUF tensor names (token_embd.weight / output.weight)
# and the HF-mapped names transform_weight normally receives
# (model.embed_tokens.weight / lm_head.weight), by checking the last
# dot-separated component of the module base name.
_CUSTOM_VLLM_REPACK_SKIP_BASENAMES = frozenset(
    {{"embed_tokens", "token_embd", "lm_head", "output"}}
)


def _custom_vllm_gguf_repack_types():
    raw = _custom_vllm_repack_os.environ.get("CUSTOM_VLLM_GGUF_REPACK", "")
    if not raw:
        return frozenset()
    if raw == "1":
        raw = "q6_k"
    types = set()
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if tok in _CUSTOM_VLLM_REPACK_NAME_TO_TYPE:
            types.add(_CUSTOM_VLLM_REPACK_NAME_TO_TYPE[tok])
    return frozenset(types)


_CUSTOM_VLLM_GGUF_REPACK_TYPES = _custom_vllm_gguf_repack_types()

if _CUSTOM_VLLM_GGUF_REPACK_TYPES:
    # {PATCH_MARKER}
    # HARD GUARD -- see module docstring "HARD GUARD" section for the full
    # root-cause arithmetic. Confirmed on L4 (Qwen3.5-2B, three-way champion
    # dispatch: CUSTOM_VLLM_GGUF_HYBRID=1 + CUSTOM_VLLM_GGUF_TRITON_MID=1):
    # down_proj's qweight bytes get correctly repacked to Q8_0 width for
    # K=6144 (6528 bytes), but the qweight_type consumed at the matmul call
    # site still reads Q6_K -- 6528 // 210 * 256 == 7936, exactly the
    # crashing "mat1 and mat2 shapes cannot be multiplied (16384x6144 and
    # 7936x2048)" error. That is a tag/data desync between the repacked
    # qweight bytes and whichever copy of qweight_type a given consumer
    # reads, and it is not provably confined to the dequant+cuBLAS path
    # that happened to crash loudly: the CUDA/Triton mmq/mmvq kernels this
    # repack exists to speed up also take qweight_type as a plain integer
    # with no shape self-check, so the same desync there would silently
    # compute wrong numbers instead of raising. Refusing to run rather than
    # risk that. Do not remove this guard without auditing every
    # qweight_type consumer (vllm_gguf_plugin's quantization/params.py
    # GGUFWeightTypeParameter materialization and quantization/linear.py's
    # apply()/_fused_mul_mat_gguf) under real GPU load, or without adding
    # -- and confirming never fires -- an assertion at each consumer that
    # qweight's byte width matches GGML_QUANT_SIZES[qweight_type] for the
    # type actually observed there.
    raise RuntimeError(
        "CUSTOM_VLLM_GGUF_REPACK is disabled: it is confirmed incompatible "
        "with the current plugin/patch stack. Repacking a K-quant tensor "
        "to Q8_0 rewrites its on-disk byte width, and at least one "
        "consumer of layer.qweight_type.weight_type observes the "
        "ORIGINAL (pre-repack) type instead of the rewritten one -- "
        "reproduced on L4 as 'RuntimeError: mat1 and mat2 shapes cannot "
        "be multiplied (16384x6144 and 7936x2048)' during warmup, where "
        "7936 is exactly what you get by taking the correctly-repacked "
        "Q8_0 byte width for K=6144 (6528 bytes) and misinterpreting it "
        "with Q6_K's block layout (6528 // 210 * 256 == 7936). See "
        "scripts/patch_gguf_repack_q6k.py's module docstring, 'HARD "
        "GUARD' section, for the full arithmetic and the audit required "
        "before re-enabling this flag. Unset CUSTOM_VLLM_GGUF_REPACK to "
        "continue."
    )

_CUSTOM_VLLM_REPACK_ORIG_TYPE = {{}}  # module base name -> original ggml type,
                                       # only for modules whose tag we rewrote


def _custom_vllm_gguf_repack_base(hf_name):
    base = hf_name
    for _suf in (".qweight_type", ".qweight", ".weight"):
        if base.endswith(_suf):
            return base[: -len(_suf)]
    return base


def _custom_vllm_gguf_repack_is_excluded(base):
    tail = base.rsplit(".", 1)[-1]
    return tail in _CUSTOM_VLLM_REPACK_SKIP_BASENAMES


def _custom_vllm_gguf_repack(hf_name, weight):
    """Re-encode a Q6_K/Q5_K tensor to Q8_0, tag row then data row.

    Runs unconditionally (general GGUF, not gated on any qwen3.5 config) so
    it works for any architecture the plugin loads. Only touches modules
    whose ".qweight_type" tag it itself rewrote -- see module docstring for
    why that is sufficient to compose correctly with earlier transforms.
    """
    import torch

    if hf_name.endswith(".qweight_type"):
        base = _custom_vllm_gguf_repack_base(hf_name)
        if _custom_vllm_gguf_repack_is_excluded(base):
            # Embedding / lm_head: never repacked, see module docstring.
            _CUSTOM_VLLM_REPACK_ORIG_TYPE.pop(base, None)
            return weight
        raw_type = int(weight.flatten()[0])
        if raw_type in _CUSTOM_VLLM_GGUF_REPACK_TYPES:
            _CUSTOM_VLLM_REPACK_ORIG_TYPE[base] = raw_type
            from gguf import GGMLQuantizationType

            return torch.full_like(weight, int(GGMLQuantizationType.Q8_0))
        _CUSTOM_VLLM_REPACK_ORIG_TYPE.pop(base, None)
        return weight

    if hf_name.endswith(".qweight"):
        base = _custom_vllm_gguf_repack_base(hf_name)
        orig_type = _CUSTOM_VLLM_REPACK_ORIG_TYPE.pop(base, None)
        if orig_type is None:
            # Tag wasn't rewritten by us: either not in the repack set, or an
            # earlier transform (e.g. qwen35's out_proj branch) already
            # re-encoded it to Q8_0 before we saw the tag.
            return weight

        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize, quantize

        qtype = GGMLQuantizationType(orig_type)
        raw = weight.cpu().numpy()
        deq = dequantize(raw, qtype)
        requant = quantize(np.ascontiguousarray(deq), GGMLQuantizationType.Q8_0)
        return torch.from_numpy(requant).to(weight.device)

    return weight
'''


def patch(path, marker, anchor, replacement, *, append=None):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if marker in src:
        print(f"Already patched: {path}")
        return
    if anchor not in src:
        raise SystemExit(
            f"Anchor not found in {path}; either the plugin source changed, or "
            "patch_gguf_qwen35_transforms.py has not run yet (this patch must "
            "be listed after it in setup_env.sh)"
        )
    src = src.replace(anchor, replacement, 1)
    if append:
        src += append
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")


site_packages = sysconfig.get_paths()["purelib"]
base_py = glob.glob(f"{site_packages}/vllm_gguf_plugin/weights_adapter/base.py")
if not base_py:
    raise SystemExit(f"vllm_gguf_plugin/weights_adapter/base.py not found under {site_packages}")

patch(base_py[0], PATCH_MARKER, ANCHOR, PATCH, append=HELPERS)
