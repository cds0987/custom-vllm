"""
vllm-gguf-plugin's build_name_map() matches HF `model_type` strings directly
against gguf.MODEL_ARCH_NAMES values. Newer Qwen releases use an HF model_type
with an underscore (e.g. "qwen3_5") while the gguf package's architecture name
has none (e.g. "qwen35"), so the exact-match lookup fails with
"Unknown gguf model_type: qwen3_5" even though the architecture is supported.

This patches the installed plugin to normalize those names, mirroring the
existing gemma3_text/cohere aliasing already done in the same function.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: qwen3.5 model_type normalization ---"

ANCHOR = '        if model_type == "gemma3_text":\n            model_type = "gemma3"\n'

PATCH = (
    ANCHOR
    + PATCH_MARKER
    + "\n"
    + '        if model_type == "qwen3_5":\n'
    + '            model_type = "qwen35"\n'
    + '        if model_type == "qwen3_5_moe":\n'
    + '            model_type = "qwen35moe"\n'
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(
    f"{site_packages}/vllm_gguf_plugin/weights_adapter/default.py"
)
if not matches:
    raise SystemExit(
        "vllm_gguf_plugin/weights_adapter/default.py not found under "
        f"{site_packages}; is vllm-gguf-plugin installed?"
    )
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
