"""
The plugin's GGUFConfig breaks serving of every NON-GGUF quantized checkpoint.

register() registers GGUFConfig as a global quantization config, so vLLM's
quant-method resolution calls override_quantization_method() on it for every
model load — including AWQ/GPTQ safetensors checkpoints that have nothing to
do with GGUF. vLLM 0.26's interface passes a third keyword:

    override_quantization_method(hf_quant_cfg, user_quant, hf_config=...)

while the plugin still implements the two-argument form, so any attempt to
serve e.g. QuantTrio/Qwen3.5-4B-AWQ dies at startup with

    TypeError: GGUFConfig.override_quantization_method() got an unexpected
    keyword argument 'hf_config'

The method's own logic is unaffected (it only inspects user_quant); accept
and ignore the new keyword. This unblocked the AWQ-Marlin comparison and the
GGUF->GPTQ transcode validation on this stack.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: vLLM 0.26 passes hf_config; accept it or every non-GGUF quant load dies ---"

ANCHOR = '''    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None
    ) -> "QuantizationMethods | None":
        del hf_quant_cfg
'''

PATCH = f'''    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> "QuantizationMethods | None":
        {PATCH_MARKER}
        del hf_quant_cfg, hf_config
'''

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/quantization/config.py")
if not matches:
    raise SystemExit(f"vllm_gguf_plugin/quantization/config.py not found under {site_packages}")
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
