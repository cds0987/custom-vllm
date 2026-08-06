"""
Patches for vllm-gguf-plugin to support the Qwen3.5 architecture, which the
plugin doesn't yet recognize:

1. build_name_map() matches HF `model_type` strings directly against
   gguf.MODEL_ARCH_NAMES values. Qwen3.5's HF model_type is "qwen3_5" (with
   underscore) while the gguf package's architecture name has none
   ("qwen35"), so the exact-match lookup fails with
   "Unknown gguf model_type: qwen3_5" even though the architecture itself
   is supported by the gguf package.

2. For multimodal (vision) configs, the plugin reads
   `config.vision_config.num_hidden_layers`, but Qwen3.5's vision config
   (Qwen3_5VisionConfig, like other Qwen-VL configs) exposes this as
   `depth` instead, causing an AttributeError.

Both are patched in place, mirroring the existing gemma3_text/cohere
aliasing already done in the same function.
"""

import glob
import sysconfig

MODEL_TYPE_MARKER = "# --- custom_vllm: qwen3.5 model_type normalization ---"
VISION_LAYERS_MARKER = "# --- custom_vllm: qwen3.5 vision depth fallback ---"

MODEL_TYPE_ANCHOR = (
    '        if model_type == "gemma3_text":\n            model_type = "gemma3"\n'
)
MODEL_TYPE_PATCH = (
    MODEL_TYPE_ANCHOR
    + MODEL_TYPE_MARKER
    + "\n"
    + '        if model_type == "qwen3_5":\n'
    + '            model_type = "qwen35"\n'
    + '        if model_type == "qwen3_5_moe":\n'
    + '            model_type = "qwen35moe"\n'
)

VISION_LAYERS_ANCHOR = (
    "            vision_name_map = gguf.get_tensor_name_map(\n"
    "                mm_proj_arch, config.vision_config.num_hidden_layers\n"
    "            )\n"
)
VISION_LAYERS_PATCH = (
    f"            {VISION_LAYERS_MARKER}\n"
    "            vision_num_layers = getattr(\n"
    '                config.vision_config, "num_hidden_layers",\n'
    '                getattr(config.vision_config, "depth", None),\n'
    "            )\n"
    "            vision_name_map = gguf.get_tensor_name_map(\n"
    "                mm_proj_arch, vision_num_layers\n"
    "            )\n"
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

changed = False

if MODEL_TYPE_MARKER in src:
    print("model_type patch already applied")
elif MODEL_TYPE_ANCHOR not in src:
    raise SystemExit(f"model_type anchor not found in {path}; plugin source may have changed")
else:
    src = src.replace(MODEL_TYPE_ANCHOR, MODEL_TYPE_PATCH, 1)
    changed = True
    print("Applied model_type patch")

if VISION_LAYERS_MARKER in src:
    print("vision depth patch already applied")
elif VISION_LAYERS_ANCHOR not in src:
    raise SystemExit(f"vision layers anchor not found in {path}; plugin source may have changed")
else:
    src = src.replace(VISION_LAYERS_ANCHOR, VISION_LAYERS_PATCH, 1)
    changed = True
    print("Applied vision depth patch")

if changed:
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
else:
    print(f"No changes needed: {path}")
