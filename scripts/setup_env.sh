#!/usr/bin/env bash
# Install vllm + vllm-gguf-plugin and apply every patch, from scratch.
# Split out of serve_test.sh so a recycled Colab runtime can be rebuilt in one
# command, and so serving options can be varied without redoing the setup.
set -e

echo "=== Installing vllm ==="
pip install -q vllm requests

echo "=== Installing llmcompressor + datasets (quantize_*.py / eval_quality_swebench.py's lazy load_dataset) ==="
# Light guard, not a hard dependency of serving itself: several scripts/ tools
# (quantize_gptq_9b.py, quantize_awq_*.py, bench_serving.py, bench_swebench.py,
# eval_quality_swebench.py's lazy `from datasets import load_dataset`) need
# these but the core serve/patch path does not, so failures here must not
# abort setup for a pure-serving session.
pip install -q llmcompressor datasets || echo "WARNING: llmcompressor/datasets install failed (non-fatal for serving-only sessions)"
# datasets pins an older huggingface_hub (observed: downgraded to 1.23.0,
# which predates the `ResolvedRevision` symbol) and silently overwrites the
# newer huggingface_hub vllm/vllm_gguf_plugin need -- vllm_gguf_plugin's
# loader.py does `from huggingface_hub import ResolvedRevision, ...` at
# import time, so any server load of ANY model (not just GGUF) then dies
# with ImportError before it even reaches gguf-specific code. Force the
# newer huggingface_hub back after datasets has had its say; harmless if
# datasets didn't touch it.
pip install -q -U huggingface_hub || echo "WARNING: huggingface_hub re-pin failed"

echo "=== Installing vllm-gguf-plugin (GGUF moved out-of-tree as of vllm 0.26) ==="
# once for the dependencies (gguf, ...), then package-only so the in-place
# patches below always apply to pristine sources
pip install -q vllm-gguf-plugin

echo "=== Rebuilding plugin from sdist so _C_gguf matches this torch ABI ==="
# The prebuilt wheel's _C_gguf.abi3.so was compiled against a different torch
# ABI (ImportError: undefined symbol torch_exception_get_what_without_backtrace
# on torch 2.11.0+cu128), silently dropping every GGUF matmul to the Triton
# fallback — which has no GEMV path and cost 3.9x at conc1 (TEST 8). Building
# the sdist locally against the installed torch fixes the import. Fall back to
# the wheel if the build fails so the environment still comes up.
SDIST_URL=$(python - <<'EOF'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/vllm-gguf-plugin/json"))
print(next(u["url"] for u in d["urls"] if u["packagetype"] == "sdist"))
EOF
)
SDIST_OK=0
curl -sL "$SDIST_URL" -o /tmp/vllm_gguf_plugin_sdist.tar.gz && SDIST_OK=1
if [ "$SDIST_OK" = "1" ] && TORCH_CUDA_ARCH_LIST="8.9" pip install -q --no-build-isolation --force-reinstall --no-deps /tmp/vllm_gguf_plugin_sdist.tar.gz; then
  echo "Built plugin from sdist (sm_89)"
else
  echo "WARNING: sdist build failed; falling back to prebuilt wheel (Triton-only kernels)"
  pip install -q --force-reinstall --no-deps vllm-gguf-plugin
fi
python - <<'EOF'
try:
    from vllm_gguf_plugin import _C_gguf  # noqa: F401
    print("_C_gguf import OK — CUDA kernels active")
except ImportError as e:
    print(f"_C_gguf NOT available ({e}); Triton fallback in use")
EOF

for s in \
  patch_gguf_plugin \
  patch_transformers_qwen35 \
  patch_gguf_tensor_mapping \
  patch_vllm_qwen35_embed \
  patch_gguf_weight_type_loader \
  patch_gguf_conv1d_shape \
  patch_vllm_qwen35_registry \
  patch_gguf_drop_mrope \
  patch_vllm_qwen35_hybrid \
  patch_gguf_empty_suffix \
  patch_gguf_qwen35_transforms \
  patch_gguf_prefer_dequant \
  patch_gguf_hybrid_dispatch \
  patch_fla_ada_shmem \
  patch_gguf_override_signature \
  patch_gguf_repack_q6k \
  patch_gguf_dequant_buffer \
  patch_gguf_threeway_dispatch \
  patch_vllm_gdn_quant_load \
  patch_gguf_auto_marlin
do
  echo "=== $s ==="
  python "scripts/$s.py"
done

echo "=== Ensuring CUDA runtime libs are on the loader path ==="
CUDART_PATH=$(find / -name "libcudart.so.13*" 2>/dev/null | head -n1)
if [ -n "$CUDART_PATH" ]; then
  echo "export LD_LIBRARY_PATH=$(dirname "$CUDART_PATH"):\$LD_LIBRARY_PATH" > /tmp/vllm_env.sh
  echo "Found $CUDART_PATH — source /tmp/vllm_env.sh before running vllm"
else
  echo "WARNING: libcudart.so.13 not found"
fi

echo "=== Setup complete ==="
