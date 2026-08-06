"""
transformers' Qwen3_5Config (a composite config with separate text_config/
vision_config sub-configs, like other Qwen-VL configs) does not expose
`vocab_size` at the top level. Qwen3_5TextModel.__init__ nevertheless reads
`config.vocab_size` directly, which crashes with:

    AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'

whenever code (e.g. vllm-gguf-plugin building a dummy model on the meta
device to inspect its state_dict) constructs a Qwen3_5ForConditionalGeneration
from the full composite config instead of config.text_config.

This adds a `vocab_size` property to Qwen3_5Config that forwards to
config.text_config.vocab_size, matching the delegation pattern used by
composite configs elsewhere in transformers.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: Qwen3_5Config.vocab_size forwarding ---"

ANCHOR = "    tie_word_embeddings: bool = False\n"

PATCH = (
    ANCHOR
    + "\n"
    + PATCH_MARKER
    + "\n"
    + "    @property\n"
    + "    def vocab_size(self):\n"
    + "        return self.text_config.vocab_size\n"
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(
    f"{site_packages}/transformers/models/qwen3_5/configuration_qwen3_5.py"
)
if not matches:
    print("configuration_qwen3_5.py not found; skipping (model may not be installed)")
    raise SystemExit(0)

path = matches[0]
with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
elif ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {path}; transformers source may have changed")
else:
    src = src.replace(ANCHOR, PATCH, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
