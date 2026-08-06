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

3. build_name_map() always builds its dummy (meta-device) model via
   AutoModelForCausalLM.from_config(config, ...), even for multimodal
   configs. For Qwen3.5, AutoModelForCausalLM resolves the composite
   (vision+text) Qwen3_5Config to Qwen3_5ForCausalLM, a text-only class
   that expects a Qwen3_5TextConfig and crashes reading config.vocab_size
   off the composite config. AutoModelForImageTextToText correctly
   resolves the same composite config to Qwen3_5ForConditionalGeneration.

All three are patched in place, mirroring the existing gemma3_text/cohere
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

AUTOMODEL_MARKER = "# --- custom_vllm: use AutoModelForImageTextToText for multimodal configs ---"
AUTOMODEL_ANCHOR = (
    "        with torch.device(\"meta\"):\n"
    "            dummy_model = AutoModelForCausalLM.from_config(\n"
    "                config, trust_remote_code=model_config.trust_remote_code\n"
    "            )\n"
)
AUTOMODEL_PATCH = (
    "        with torch.device(\"meta\"):\n"
    f"            {AUTOMODEL_MARKER}\n"
    "            if is_multimodal:\n"
    "                from transformers import AutoModelForImageTextToText\n"
    "                dummy_model = AutoModelForImageTextToText.from_config(\n"
    "                    config, trust_remote_code=model_config.trust_remote_code\n"
    "                )\n"
    "            else:\n"
    "                dummy_model = AutoModelForCausalLM.from_config(\n"
    "                    config, trust_remote_code=model_config.trust_remote_code\n"
    "                )\n"
)

if AUTOMODEL_MARKER in src:
    print("AutoModel class-selection patch already applied")
elif AUTOMODEL_ANCHOR not in src:
    raise SystemExit(f"AutoModel anchor not found in {path}; plugin source may have changed")
else:
    src = src.replace(AUTOMODEL_ANCHOR, AUTOMODEL_PATCH, 1)
    changed = True
    print("Applied AutoModel class-selection patch")

if changed:
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
else:
    print(f"No changes needed: {path}")
