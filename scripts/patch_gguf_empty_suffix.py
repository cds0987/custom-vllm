"""
build_name_map() registers a trailing dot for parameters that have no
".weight"/".bias" suffix, so those tensors are never loaded.

In vllm_gguf_plugin/weights_adapter/default.py, find_hf_name_in_tensor_map()
splits an HF parameter name into (base_name, suffix), where suffix is "" for a
bare parameter, and then unconditionally rejoins:

    return gguf_name + "." + suffix

For a bare parameter that yields "blk.1.ssm_a." while the GGUF file stores
"blk.1.ssm_a" — one character off, so the lookup at load time never matches.

Nothing catches it. The mapping *is* added to gguf_to_hf_name_map (under the
bad key), so the "Failed to map GGUF parameters" check, which only tests
`hf_name not in gguf_to_hf_name_map.values()`, is satisfied. The model loads
cleanly and the parameter silently keeps its initialised value.

For Qwen3.5 the casualty is linear_attn.A_log in all 24 gated-delta-net layers.
vllm computes the decay gate as

    g = -A_log.float().exp() * softplus(a + dt_bias)

so an unloaded (zero) A_log makes -exp(0) = -1 uniformly across every head,
destroying the recurrence in three quarters of the network. The model still
runs and still emits well-formed responses structurally — it just degenerates
to argmax over a broken distribution, printing token 0 ("!") forever.

Every SSM/hybrid architecture is affected the same way, since A_log is bare in
all of them (mamba, jamba, qwen3next, qwen3.5, ...).
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: bare parameters have no suffix; don't append a trailing dot ---"

ANCHOR = '            return gguf_name + "." + suffix\n'
PATCH = (
    f"            {PATCH_MARKER}\n"
    '            return gguf_name + "." + suffix if suffix else gguf_name\n'
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/weights_adapter/default.py")
if not matches:
    raise SystemExit(f"vllm_gguf_plugin/weights_adapter/default.py not found under {site_packages}")
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
