#!/usr/bin/env bash
set -e

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
GGUF_QUANT="${2:-}"
TOKENIZER="${3:-}"
MAX_MODEL_LEN="${4:-4096}"
GPU_MEM_UTIL="${5:-0.85}"

echo "=== GPU status ==="
nvidia-smi

echo "=== Installing vllm ==="
pip install -q vllm requests
if [ -n "$GGUF_QUANT" ]; then
  echo "=== Installing vllm-gguf-plugin (GGUF support moved out-of-tree as of vllm 0.26) ==="
  pip install -q vllm-gguf-plugin
  echo "=== Patching vllm-gguf-plugin for qwen3.5 model_type naming mismatch ==="
  python scripts/patch_gguf_plugin.py
  echo "=== Patching transformers Qwen3_5Config.vocab_size ==="
  python scripts/patch_transformers_qwen35.py
  echo "=== Patching gguf tensor_mapping.py for qwen3.5 tensor names ==="
  python scripts/patch_gguf_tensor_mapping.py
  echo "=== Patching vllm Qwen3_5Model to pass quant_config to embed_tokens ==="
  python scripts/patch_vllm_qwen35_embed.py
  echo "=== Patching vllm-gguf-plugin weight-type loader for fused layers ==="
  python scripts/patch_gguf_weight_type_loader.py
  echo "=== Patching vllm-gguf-plugin conv1d weight shape for SSM layers ==="
  python scripts/patch_gguf_conv1d_shape.py
fi

echo "=== Ensuring CUDA runtime libs are on the loader path ==="
CUDART_PATH=$(find / -xdev -name "libcudart.so.13*" 2>/dev/null | head -n1)
if [ -z "$CUDART_PATH" ]; then
  CUDART_PATH=$(find / -name "libcudart.so.13*" 2>/dev/null | head -n1)
fi
if [ -z "$CUDART_PATH" ]; then
  pip install -q nvidia-cuda-runtime-cu13 2>/dev/null || true
  CUDART_PATH=$(python - <<'EOF'
import os
try:
    import nvidia.cuda_runtime
    d = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib")
    for f in os.listdir(d):
        if f.startswith("libcudart.so.13"):
            print(os.path.join(d, f))
            break
except Exception:
    pass
EOF
)
fi
if [ -z "$CUDART_PATH" ]; then
  apt-get -qq update >/dev/null 2>&1 && apt-get -qq install -y cuda-cudart-13-0 >/dev/null 2>&1 || true
  CUDART_PATH=$(find / -name "libcudart.so.13*" 2>/dev/null | head -n1)
fi
if [ -n "$CUDART_PATH" ]; then
  export LD_LIBRARY_PATH="$(dirname "$CUDART_PATH"):$LD_LIBRARY_PATH"
  echo "Found $CUDART_PATH, added to LD_LIBRARY_PATH"
else
  echo "WARNING: libcudart.so.13 not found anywhere on the filesystem"
fi

SERVE_TARGET="$MODEL"
TOKENIZER_ARGS=()
if [ -n "$GGUF_QUANT" ]; then
  SERVE_TARGET="${MODEL}:${GGUF_QUANT}"
  if [ -n "$TOKENIZER" ]; then
    TOKENIZER_ARGS=(--tokenizer "$TOKENIZER")
  fi
fi

echo "=== Starting vllm serve ($SERVE_TARGET, max-model-len=$MAX_MODEL_LEN, gpu-mem-util=$GPU_MEM_UTIL) ==="
nohup vllm serve "$SERVE_TARGET" "${TOKENIZER_ARGS[@]}" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enforce-eager \
  --port 8000 > vllm_server.log 2>&1 &
SERVER_PID=$!

echo "=== Waiting for server readiness ==="
READY=0
for i in $(seq 1 180); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models | grep -q 200; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process exited early."
    break
  fi
  sleep 10
done

if [ "$READY" -ne 1 ]; then
  echo "Server failed to start within timeout. Last log lines:"
  tail -n 150 vllm_server.log
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

echo "=== Running inference test ==="
VLLM_MODEL="$MODEL" python scripts/test_inference.py

kill "$SERVER_PID" 2>/dev/null || true
