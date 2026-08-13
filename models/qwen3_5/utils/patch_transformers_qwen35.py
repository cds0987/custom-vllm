"""
transformers' Qwen3_5Config (a composite config with separate text_config/
vision_config sub-configs, like other Qwen-VL configs) does not expose
`vocab_size` at the top level. Qwen3_5TextModel.__init__ nevertheless reads
`config.vocab_size` directly, which crashes with:

    AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'

whenever code (e.g. vllm-gguf-plugin building a dummy model on the meta
device to inspect its state_dict) constructs a Qwen3_5ForConditionalGeneration
from the full composite config instead of config.text_config.

PreTrainedConfig is a "strict dataclass" (huggingface_hub.dataclasses) that
type-checks every attribute against its declared type on assignment, and
`vocab_size: int` is already declared as a real field somewhere up the MRO.
So a `@property` override doesn't work here (the strict-dataclass __init__
tries to assign the property object itself as vocab_size's value and fails
type validation). Instead, this sets `self.vocab_size` to a real int inside
__post_init__, once text_config has been resolved.
"""

import glob
import sysconfig

OLD_PROPERTY_MARKER = "# --- custom_vllm: Qwen3_5Config.vocab_size forwarding ---"
PATCH_MARKER = "# --- custom_vllm: Qwen3_5Config.vocab_size assignment ---"

OLD_PROPERTY_BLOCK = (
    "\n"
    + OLD_PROPERTY_MARKER
    + "\n"
    + "    @property\n"
    + "    def vocab_size(self):\n"
    + "        return self.text_config.vocab_size\n"
)

ANCHOR = (
    "        elif self.text_config is None:\n"
    '            self.text_config = self.sub_configs["text_config"]()\n'
    "\n"
    "        super().__post_init__(**kwargs)\n"
)

PATCH = (
    "        elif self.text_config is None:\n"
    '            self.text_config = self.sub_configs["text_config"]()\n'
    "\n"
    f"        {PATCH_MARKER}\n"
    "        self.vocab_size = self.text_config.vocab_size\n"
    "\n"
    "        super().__post_init__(**kwargs)\n"
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

if OLD_PROPERTY_BLOCK in src:
    src = src.replace(OLD_PROPERTY_BLOCK, "", 1)
    print("Removed broken property-based patch from a previous run")

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
elif ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {path}; transformers source may have changed")
else:
    src = src.replace(ANCHOR, PATCH, 1)
    print(f"Patched: {path}")

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
