#!/usr/bin/env bash
set -e

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
GGUF_FILE="${2:-}"
TOKENIZER="${3:-}"

echo "=== GPU status ==="
nvidia-smi

echo "=== Installing vllm ==="
pip install -q vllm requests

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
if [ -n "$GGUF_FILE" ]; then
  echo "=== Downloading GGUF file $GGUF_FILE from $MODEL ==="
  pip install -q huggingface_hub
  SERVE_TARGET=$(python - <<EOF
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id="$MODEL", filename="$GGUF_FILE"))
EOF
)
  if [ -n "$TOKENIZER" ]; then
    TOKENIZER_ARGS=(--tokenizer "$TOKENIZER")
  fi
fi

echo "=== Starting vllm serve ($SERVE_TARGET) ==="
nohup vllm serve "$SERVE_TARGET" "${TOKENIZER_ARGS[@]}" --port 8000 > vllm_server.log 2>&1 &
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
