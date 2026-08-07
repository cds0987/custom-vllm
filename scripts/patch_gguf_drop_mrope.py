"""
A text-only GGUF keeps the multimodal config's M-RoPE settings, which vllm
then asserts the model must implement:

    File "vllm/v1/worker/gpu/mm/rope.py", line 144, in get_rope_state
        assert isinstance(model, SupportsMRoPE)
    AssertionError

ModelConfig.uses_mrope is decided purely from the config
(rope_scaling["mrope_section"] on the text config), but the text-only class
vllm builds -- Qwen3_5ForCausalLM, see patch_vllm_qwen35_registry.py -- does
not declare SupportsMRoPE, and rightly so: M-RoPE only exists to give image
and video tokens separate temporal/height/width position components. With no
vision tower every token is a text token, so t == h == w == position, and
partitioning the frequency dimensions across three identical position values
is exactly standard 1-D RoPE. Dropping mrope_section here is therefore a
rewrite of the config to what the loaded model actually is, not an
approximation -- the same thing llama.cpp does when it loads the language
tower of a Qwen3.5 GGUF without an mmproj file.

This is applied in GGUFConfigParser.parse(), right after it pins the
architecture to the text-only class, and only when that architecture really
is a plain ForCausalLM.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: no vision tower => M-RoPE degenerates to 1-D RoPE ---"

ANCHOR = """        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config_dict["architectures"] = [model_type]
        config.update({"architectures": [model_type]})
"""

PATCH = (
    ANCHOR
    + f"""
        {PATCH_MARKER}
        if model_type.endswith("ForCausalLM"):
            _text_cfg = config.get_text_config()
            _rope = getattr(_text_cfg, "rope_scaling", None)
            if isinstance(_rope, dict) and "mrope_section" in _rope:
                _rope = {{
                    k: v
                    for k, v in _rope.items()
                    if k not in ("mrope_section", "mrope_interleaved")
                }}
                if not _rope or set(_rope) <= {{"rope_type", "type"}}:
                    _rope = None
                _text_cfg.rope_scaling = _rope
                if "rope_scaling" in config_dict:
                    config_dict["rope_scaling"] = _rope
                _text_dict = config_dict.get("text_config")
                if isinstance(_text_dict, dict) and "rope_scaling" in _text_dict:
                    _text_dict["rope_scaling"] = _rope
"""
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/config_parser.py")
if not matches:
    raise SystemExit(f"vllm_gguf_plugin/config_parser.py not found under {site_packages}")
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
