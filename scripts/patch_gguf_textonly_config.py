"""
A GGUF file for a multimodal model contains only the language tower; the
vision tower ships as a separate "mmproj-*.gguf". llama.cpp models this
correctly — its Qwen3.5 loader (src/models/qwen35.cpp) builds a pure text
model, and mmproj is an independent, optional file.

vllm-gguf-plugin already knows this (gguf_utils.detect_gguf_multimodal
looks for an mmproj file next to the model), but only acts on it for
Gemma3. For every other architecture it leaves the HF config's
`vision_config` intact, so vllm builds the full multimodal model with a
vision tower whose weights were never in the GGUF. Loading "succeeds" (the
vision params stay empty), then the startup profile run pushes a dummy
image through the empty tower and dies inside the quantized matmul:

    RuntimeError: size mismatch, got input (65536), mat (65536x1024), vec (0)

This drops `vision_config` when no mmproj file accompanies the GGUF, so
vllm instantiates the text-only model — matching what was actually
downloaded. Fetch an "mmproj-*.gguf" into the same snapshot directory and
the vision config is kept, re-enabling image input.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: drop vision_config when no mmproj file accompanies the GGUF ---"

ANCHOR = """        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config
"""

PATCH = '''        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        {marker}
        if is_gguf(original_model) and getattr(config, "vision_config", None) is not None:
            # maybe_patch_hf_config_from_gguf() only actually wires up an mmproj
            # vision tower for Gemma3; for every other architecture the vision
            # weights are never loaded, so keeping vision_config would build an
            # empty tower that crashes on the first (profile-run) image.
            mmproj_supported = config.model_type in ("gemma3", "gemma3_text")
            if not (mmproj_supported and self._gguf_has_mmproj(original_model)):
                config.vision_config = None
                config_dict.pop("vision_config", None)

        return config_dict, config

    @staticmethod
    def _gguf_has_mmproj(model) -> bool:
        """True if an mmproj (vision tower) GGUF sits alongside the model file."""
        from pathlib import Path

        from .gguf_utils import detect_gguf_multimodal

        path = str(model)
        if check_gguf_file(path):
            return detect_gguf_multimodal(path) is not None

        # Remote "repo_id:quant" form: the weights land in the HF cache, so
        # look for an mmproj file inside the downloaded snapshot.
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
        else:
            repo_id = path
        cache_name = "models--" + str(repo_id).replace("/", "--")
        pattern = str(
            Path.home() / ".cache" / "huggingface" / "hub" / cache_name
            / "snapshots" / "*" / "*mmproj*.gguf"
        )
        return bool(glob.glob(pattern))
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
    raise SystemExit(0)

if "custom_vllm: drop vision_config for text-only GGUF checkpoints" in src:
    raise SystemExit(
        f"{path} still carries the earlier text-only patch; reinstall "
        "vllm-gguf-plugin (pip install --force-reinstall --no-deps vllm-gguf-plugin) "
        "and rerun."
    )

if ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {path}; plugin source may have changed")

src = src.replace(ANCHOR, PATCH, 1)
if "\nimport glob\n" not in src:
    src = src.replace("from pathlib import Path", "import glob\nfrom pathlib import Path", 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print(f"Patched: {path}")
