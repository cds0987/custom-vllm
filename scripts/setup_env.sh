#!/usr/bin/env bash
# Install vllm + vllm-gguf-plugin and apply every patch, from scratch.
# Split out of serve_test.sh so a recycled Colab runtime can be rebuilt in one
# command, and so serving options can be varied without redoing the setup.
set -e

echo "=== Installing vllm ==="
pip install -q vllm requests

echo "=== Installing vllm-gguf-plugin (GGUF moved out-of-tree as of vllm 0.26) ==="
# once for the dependencies (gguf, ...), then package-only so the in-place
# patches below always apply to pristine sources
pip install -q vllm-gguf-plugin
pip install -q --force-reinstall --no-deps vllm-gguf-plugin

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
  patch_fla_ada_shmem
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
