"""
vllm-gguf-plugin's _gguf_weight_type_loader_v2 only short-circuits to the
parameter's own _store() when there is no shard id:

    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):
        if loaded_shard_id is None and hasattr(param, "_store"):
            param._store(loaded_weight)
            return
        base_loader(param, loaded_weight, loaded_shard_id)

For a fused layer (QKVParallelLinear, MergedColumnParallelLinear — i.e. any
model with fused qkv_proj / gate_up_proj, which includes Qwen3.5), the weight
type IS loaded per shard, so loaded_shard_id is set and the call falls through
to vllm's generic weight_loader_v2 → _load_fused_module_from_checkpoint, which
reads param.output_dim:

    AttributeError: 'GGUFUninitializedWeightTypeParameter' object has no
    attribute 'output_dim'

A GGUF weight-type parameter is a scalar ggml-dtype tag, not a sharded weight
tensor, so it legitimately has no input_dim/output_dim — routing it through
the fused-weight path is the bug. _store_gguf_weight_type() already handles the
sharded case (it writes into param.shard_weight_type[shard_id]), so the fix is
to use _store() whenever the parameter provides it, shard id or not.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: weight-type params have no output_dim, always use _store ---"

ANCHOR = (
    "    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):\n"
    '        if loaded_shard_id is None and hasattr(param, "_store"):\n'
    "            param._store(loaded_weight)\n"
    "            return\n"
    "        base_loader(param, loaded_weight, loaded_shard_id)\n"
)
PATCH = (
    "    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):\n"
    f"        {PATCH_MARKER}\n"
    '        if hasattr(param, "_store"):\n'
    "            param._store(loaded_weight, shard_id=loaded_shard_id)\n"
    "            return\n"
    "        base_loader(param, loaded_weight, loaded_shard_id)\n"
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/quantization/params.py")
if not matches:
    raise SystemExit(f"vllm_gguf_plugin/quantization/params.py not found under {site_packages}")
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
elif ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {path}; plugin source may have changed")
else:
    src = src.replace(ANCHOR, PATCH, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
