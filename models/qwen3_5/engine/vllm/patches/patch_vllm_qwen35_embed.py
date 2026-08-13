"""
vllm's Qwen3_5Model.__init__ constructs its token embedding without passing
quant_config:

    self.embed_tokens = VocabParallelEmbedding(
        self.vocab_size,
        config.hidden_size,
    )

VocabParallelEmbedding accepts an optional quant_config that makes it
quantization-aware (able to receive the ".qweight_type" metadata GGUF
weight loading attaches to quantized parameters). Without it, loading a
GGUF checkpoint whose token embedding is stored in a quantized ggml type
(as opposed to F16/F32, which some quants keep unquantized) fails with:

    ValueError: There is no module or parameter named 'embed_tokens.qweight_type'
    in Qwen3_5Model.

vllm's own Qwen3Model (via Qwen2Model) already passes quant_config for this
exact reason; Qwen3_5Model (and its parent Qwen3NextModel, which has the
same gap) just omits it. This adds it back.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: pass quant_config to embed_tokens ---"

ANCHOR = (
    "        self.embed_tokens = VocabParallelEmbedding(\n"
    "            self.vocab_size,\n"
    "            config.hidden_size,\n"
    "        )\n"
)
PATCH = (
    "        self.embed_tokens = VocabParallelEmbedding(\n"
    "            self.vocab_size,\n"
    "            config.hidden_size,\n"
    f"            quant_config=self.quant_config,  {PATCH_MARKER}\n"
    "        )\n"
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm/model_executor/models/qwen3_5.py")
if not matches:
    raise SystemExit(f"vllm/model_executor/models/qwen3_5.py not found under {site_packages}")
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
elif ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {path}; vllm source may have changed")
else:
    src = src.replace(ANCHOR, PATCH, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
