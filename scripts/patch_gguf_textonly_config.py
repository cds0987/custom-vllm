"""
A GGUF file for a multimodal model contains only the language tower; the
vision tower ships as a separate "mmproj-*.gguf". llama.cpp models this
correctly — its Qwen3.5 loader (src/models/qwen35.cpp) builds a pure text
model and treats mmproj as an independent, optional file.

vllm-gguf-plugin instead keeps the HF config's `vision_config` intact, so
vllm builds the full Qwen3_5ForConditionalGeneration with a vision tower
whose weights were never in the GGUF. Loading "succeeds" (the vision params
stay empty), then the startup profile run pushes a dummy image through the
empty tower and dies inside the quantized matmul:

    RuntimeError: size mismatch, got input (65536), mat (65536x1024), vec (0)

This drops `vision_config` from the parsed config when the GGUF being loaded
has no vision tensors, so vllm instantiates the text-only model — matching
what the checkpoint actually contains.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: drop vision_config for text-only GGUF checkpoints ---"

ANCHOR = """        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config
"""

PATCH = '''        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        {marker}
        if is_gguf(original_model) and getattr(config, "vision_config", None) is not None:
            if not self._gguf_has_vision_tensors(original_model):
                config.vision_config = None
                config_dict.pop("vision_config", None)

        return config_dict, config

    @staticmethod
    def _gguf_has_vision_tensors(model) -> bool:
        """True if the GGUF file itself carries vision-tower tensors."""
        import gguf

        path = str(model)
        if not check_gguf_file(path):
            return True
        try:
            reader = gguf.GGUFReader(path)
        except Exception:
            return True
        return any(
            t.name.startswith("v.") or t.name.startswith("mm.")
            for t in reader.tensors
        )
'''.format(marker=PATCH_MARKER)

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
