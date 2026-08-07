"""
vllm ships text-only Qwen3.5 model classes but never registers them.

vllm/model_executor/models/qwen3_5.py defines:

    class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase)         # line ~376
    class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, ...)  # line ~380

yet registry.py only lists the multimodal variants:

    "Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration")
    "Qwen3_5MoeForConditionalGeneration": ("qwen3_5", "Qwen3_5MoeForConditionalGeneration")

So a checkpoint whose config says architectures=["Qwen3_5ForCausalLM"] -- which is
exactly what a GGUF resolves to, since vllm-gguf-plugin's GGUFConfigParser sets
architectures from MODEL_FOR_CAUSAL_LM_MAPPING_NAMES -- does not match any
registered architecture. Instead of failing loudly, _ModelRegistry._normalize_arch()
falls through to try_match_architecture_defaults() and rewrites the suffix:

    "Qwen3_5ForCausalLM"  ->  (suffix "ForCausalLM" not registered)
                          ->  "Qwen3_5ForConditionalGeneration"  (registered!)

Both suffixes share the ("generate", "none") runner/convert defaults, so the
normalization is considered a legal match and vllm silently instantiates the
*multimodal* model for a text-only checkpoint. The GGUF carries no vision
weights (llama.cpp ships those separately as mmproj-*.gguf), so the vision tower
is built with empty quantized weights, and the startup profile run pushes a
dummy image through it:

    File "vllm/v1/worker/gpu_model_runner.py", in profile_run
        dummy_encoder_outputs = self.model.embed_multimodal(...)
    File "vllm/model_executor/models/qwen3_vl.py", in _process_image_input
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
    File "vllm_gguf_plugin/quantization/linear.py", in _fused_mul_mat_gguf
        return x @ qweight.T
    RuntimeError: size mismatch, got input (65536), mat (65536x1024), vec (0)

The `vec (0)` is the give-away: the vision projection weight has zero elements.

Registering the two text-only classes makes the architecture resolve directly and
skips the suffix-rewrite entirely, so text-only Qwen3.5 checkpoints (GGUF and
otherwise) load the model they actually are.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: text-only Qwen3.5 classes exist in qwen3_5.py but were never registered ---"

ANCHOR = '    "Qwen3NextForCausalLM": ("qwen3_next", "Qwen3NextForCausalLM"),\n'
PATCH = (
    ANCHOR
    + f"    {PATCH_MARKER}\n"
    '    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
    '    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),\n'
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm/model_executor/models/registry.py")
if not matches:
    raise SystemExit(f"vllm/model_executor/models/registry.py not found under {site_packages}")
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
elif ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {path}; vllm registry may have changed")
else:
    src = src.replace(ANCHOR, PATCH, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
