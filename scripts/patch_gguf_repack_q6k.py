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
_CUSTOM_VLLM_REPACK_ORIG_TYPE = {{}}  # module base name -> original ggml type,
                                       # only for modules whose tag we rewrote


def _custom_vllm_gguf_repack_base(hf_name):
    base = hf_name
    for _suf in (".qweight_type", ".qweight", ".weight"):
        if base.endswith(_suf):
            return base[: -len(_suf)]
    return base


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
