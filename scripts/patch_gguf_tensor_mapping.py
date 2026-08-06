"""
The `gguf` package (llama.cpp's Python tensor-naming library) ships a fixed
table of known HF tensor-name conventions per architecture in
tensor_mapping.py. Other hybrid-attention models (qwen3next, mamba, jamba,
...) name the SSM dt-bias tensor "...dt_proj" (a Linear layer with separate
.weight/.bias), but Qwen3.5 names it "...linear_attn.dt_bias" directly — a
standalone bias parameter. Once vllm-gguf-plugin's suffix-stripping treats
the trailing "_bias" the same way it already treats "_weight" (see
patch_gguf_plugin.py), the remaining base name is "...linear_attn.dt",
which still isn't a recognized template, causing "Failed to map GGUF
parameters" for every dt_bias tensor. This adds that missing template.
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

anchor = '            "model.layers.{bid}.linear_attn.dt_proj",   # qwen3next\n'
patch = (
    anchor
    + f'            "model.layers.{{bid}}.linear_attn.dt", {PATCH_MARKER}\n'
)

if anchor not in src:
    raise SystemExit(f"Anchor not found in {path}: {anchor!r}; gguf source may have changed")

src = src.replace(anchor, patch, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print(f"Patched: {path}")
