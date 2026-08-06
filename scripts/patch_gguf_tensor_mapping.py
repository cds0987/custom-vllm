"""
The `gguf` package (llama.cpp's Python tensor-naming library) ships a fixed
table of known HF tensor-name conventions per architecture in
tensor_mapping.py. Qwen3.5 uses two naming conventions this table doesn't
yet recognize, causing vllm-gguf-plugin's build_name_map() to fail with
"Failed to map GGUF parameters" for 30 tensors:

1. SSM_DT: other hybrid-attention models (qwen3next, mamba, jamba, ...) name
   this tensor "...dt_proj" (a Linear layer with separate .weight/.bias),
   but Qwen3.5 names it "...linear_attn.dt_bias" directly — a standalone
   bias parameter with no matching ".weight" counterpart and no "_proj"
   suffix to strip. This is a materially different naming shape, not just
   a missing alias, so it needs its own template entry.

2. Vision merger (V_DS_NORM / V_DS_FC1 / V_DS_FC2): qwen3vl's "deepstack"
   merger tensors are indexed per-block ("deepstack_merger_list.{bid}.norm"
   etc.), but Qwen3.5's merger is a single non-indexed module named
   "visual.merger.norm" / "visual.merger.linear_fc1" / "visual.merger.linear_fc2".
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: qwen3.5 tensor name additions ---"

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/gguf/tensor_mapping.py")
if not matches:
    raise SystemExit(f"gguf/tensor_mapping.py not found under {site_packages}")
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
    raise SystemExit(0)

replacements = [
    (
        '            "model.layers.{bid}.linear_attn.dt_proj",   # qwen3next\n',
        '            "model.layers.{bid}.linear_attn.dt_proj",   # qwen3next\n'
        f'            "model.layers.{{bid}}.linear_attn.dt_bias",   {PATCH_MARKER} qwen3.5\n',
    ),
    (
        '            "model.visual.deepstack_merger_list.{bid}.norm", # deepstack in qwen3vl\n',
        '            "model.visual.deepstack_merger_list.{bid}.norm", # deepstack in qwen3vl\n'
        f'            "visual.merger.norm", {PATCH_MARKER} qwen3.5\n',
    ),
    (
        '            "model.visual.deepstack_merger_list.{bid}.linear_fc1", # deepstack in qwen3vl\n',
        '            "model.visual.deepstack_merger_list.{bid}.linear_fc1", # deepstack in qwen3vl\n'
        f'            "visual.merger.linear_fc1", {PATCH_MARKER} qwen3.5\n',
    ),
    (
        '            "model.visual.deepstack_merger_list.{bid}.linear_fc2", # deepstack in qwen3vl\n',
        '            "model.visual.deepstack_merger_list.{bid}.linear_fc2", # deepstack in qwen3vl\n'
        f'            "visual.merger.linear_fc2", {PATCH_MARKER} qwen3.5\n',
    ),
]

for anchor, patch in replacements:
    if anchor not in src:
        raise SystemExit(f"Anchor not found in {path}: {anchor!r}; gguf source may have changed")
    src = src.replace(anchor, patch, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print(f"Patched: {path}")
