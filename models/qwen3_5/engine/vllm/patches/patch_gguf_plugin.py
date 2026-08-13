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

3. build_name_map() decides "is this multimodal?" from the config alone
   (`config.vision_config is not None`) and always builds its dummy
   (meta-device) model with AutoModelForCausalLM.from_config(config, ...).
   Both halves are wrong for a text-only GGUF of a multimodal model:

   - vllm resolves the *architecture* (e.g. Qwen3_5ForCausalLM, see
     patch_vllm_qwen35_registry.py) and builds only the language tower,
     so the name map must describe that tower. Deriving multimodality
     from the config instead produces "model.language_model.layers.*"
     names and loading fails with "There is no module or parameter named
     'language_model'". is_multimodal is therefore cleared whenever every
     resolved architecture is a plain ForCausalLM.
   - passing the composite (vision+text) config to AutoModelForCausalLM
     resolves to the text-only class, which expects a Qwen3_5TextConfig
     and crashes reading config.vocab_size. The already-computed
     `text_config` (config.get_text_config(), identity for non-composite
     configs) is the right thing to hand it.

   AutoModelForImageTextToText is used for the genuinely multimodal case,
   where AutoModelForCausalLM would resolve the composite config to the
   wrong class.

4. Suffix stripping only special-cases a trailing "_weight" (no dot) on
   top of the normal ".weight"/".bias" split. Qwen3.5's SSM dt-bias tensor
   is named "...linear_attn.dt_bias" (trailing "_bias", no dot), so it falls
   through unstripped and never matches gguf's tensor template (see
   patch_gguf_tensor_mapping.py, which adds the matching template for the
   stripped form). This adds the symmetric "_bias" handling.

5. Text-only GGUF quants of Qwen3.5 (no separate mmproj file loaded) have
   no vision tensors at all, so the dummy multimodal model's
   "model.visual.merger.{norm,linear_fc1,linear_fc2}" params can never be
   mapped — not a bug, just genuinely absent weights. These are added to
   sideload_params (the same allowlist mechanism already used for MoE
   expert weights) so they're tolerated as expected-missing instead of
   raising "Failed to map GGUF parameters".

All are patched in place, mirroring the existing gemma3_text/cohere
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

MULTIMODAL_MARKER = "# --- custom_vllm: only multimodal if vllm actually builds a vision tower ---"
MULTIMODAL_ANCHOR = (
    "        is_multimodal = (\n"
    '            hasattr(config, "vision_config") and config.vision_config is not None\n'
    "        )\n"
)
MULTIMODAL_PATCH = (
    MULTIMODAL_ANCHOR
    + f"        {MULTIMODAL_MARKER}\n"
    "        _archs = getattr(model_config, \"architectures\", None) or []\n"
    "        if is_multimodal and _archs and all(\n"
    '            a.endswith("ForCausalLM") for a in _archs\n'
    "        ):\n"
    "            is_multimodal = False\n"
)

if MULTIMODAL_MARKER in src:
    print("is_multimodal architecture patch already applied")
elif MULTIMODAL_ANCHOR not in src:
    raise SystemExit(f"is_multimodal anchor not found in {path}; plugin source may have changed")
else:
    src = src.replace(MULTIMODAL_ANCHOR, MULTIMODAL_PATCH, 1)
    changed = True
    print("Applied is_multimodal architecture patch")

AUTOMODEL_MARKER = "# --- custom_vllm: pick the AutoModel class vllm's architecture implies ---"
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
    "                    text_config, trust_remote_code=model_config.trust_remote_code\n"
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

BIAS_SUFFIX_MARKER = "# --- custom_vllm: qwen3.5 _bias suffix stripping ---"
BIAS_SUFFIX_ANCHOR = (
    '                if base_name.endswith("_weight"):\n'
    "                    base_name = base_name[:-7]\n"
    '                    suffix = "weight"\n'
)
BIAS_SUFFIX_PATCH = (
    BIAS_SUFFIX_ANCHOR
    + f"                {BIAS_SUFFIX_MARKER}\n"
    + '                elif base_name.endswith("_bias"):\n'
    + "                    base_name = base_name[:-5]\n"
    + '                    suffix = "bias"\n'
)

if BIAS_SUFFIX_MARKER in src:
    print("_bias suffix patch already applied")
elif BIAS_SUFFIX_ANCHOR not in src:
    raise SystemExit(f"_bias suffix anchor not found in {path}; plugin source may have changed")
else:
    src = src.replace(BIAS_SUFFIX_ANCHOR, BIAS_SUFFIX_PATCH, 1)
    changed = True
    print("Applied _bias suffix patch")

SIDELOAD_MARKER = "# --- custom_vllm: qwen3.5 text-only gguf has no vision merger tensors ---"
SIDELOAD_ANCHOR = "\n        arch = None\n"
SIDELOAD_PATCH = (
    "\n"
    f"        {SIDELOAD_MARKER}\n"
    '        if is_multimodal and model_type == "qwen35":\n'
    "            sideload_params.append(\n"
    "                regex.compile(\n"
    r'                    r"model\.visual\.merger\.(norm|linear_fc1|linear_fc2)\.(weight|bias)"'
    "\n"
    "                )\n"
    "            )\n"
    "\n        arch = None\n"
)

if SIDELOAD_MARKER in src:
    print("vision sideload patch already applied")
elif SIDELOAD_ANCHOR not in src:
    raise SystemExit(f"sideload anchor not found in {path}; plugin source may have changed")
else:
    src = src.replace(SIDELOAD_ANCHOR, SIDELOAD_PATCH, 1)
    changed = True
    print("Applied vision sideload patch")

if changed:
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
else:
    print(f"No changes needed: {path}")
