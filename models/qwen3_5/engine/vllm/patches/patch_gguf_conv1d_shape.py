"""
GGUF stores the SSM/Mamba depthwise conv1d kernel as a 2-D tensor of shape
(channels, kernel_size), while vllm's MambaMixer2 allocates conv1d.weight
with torch.nn.Conv1d's canonical 3-D layout (channels, 1, kernel_size)
(groups == channels, so the per-group input channel dim is 1). The
mamba_mixer2 sharded weight loader assigns the loaded tensor into a slice of
the parameter without reconciling rank, so loading a GGUF hybrid model
(Qwen3.5, qwen3next, ...) fails with:

    RuntimeError: The expanded size of the tensor (1) must match the existing
    size (2048) at non-singleton dimension 1.
    Target sizes: [2048, 1, 4]. Tensor sizes: [2048, 4]

This is purely a storage-layout difference, so it belongs in the GGUF
adapter's transform_weight() hook (the same place gemma3's adapter already
fixes up its own GGUF-vs-HF weight discrepancies): unsqueeze the middle dim
back in for conv1d weights.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: GGUF stores conv1d weight 2-D, vllm expects 3-D ---"

ANCHOR = (
    "        \"\"\"Transform one loaded weight.\"\"\"\n"
    "        del hf_name\n"
    "        return weight\n"
)
PATCH = (
    "        \"\"\"Transform one loaded weight.\"\"\"\n"
    f"        {PATCH_MARKER}\n"
    '        if hf_name.endswith("conv1d.weight") and weight.ndim == 2:\n'
    "            return weight.unsqueeze(1)\n"
    "        del hf_name\n"
    "        return weight\n"
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/weights_adapter/base.py")
if not matches:
    raise SystemExit(f"vllm_gguf_plugin/weights_adapter/base.py not found under {site_packages}")
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
