#!/usr/bin/env bash
set -e

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "=== GPU status ==="
nvidia-smi

echo "=== Installing vllm ==="
pip install -q vllm requests

echo "=== Ensuring CUDA runtime libs are on the loader path ==="
pip install -q nvidia-cuda-runtime-cu13 || true
CUDART_LIB_DIR=$(python - <<'EOF'
import os
try:
    import nvidia.cuda_runtime
    print(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib"))
except ImportError:
    pass
EOF
)
if [ -n "$CUDART_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$CUDART_LIB_DIR:$LD_LIBRARY_PATH"
  echo "Added $CUDART_LIB_DIR to LD_LIBRARY_PATH"
fi

echo "=== Starting vllm serve ($MODEL) ==="
nohup vllm serve "$MODEL" --port 8000 > vllm_server.log 2>&1 &
SERVER_PID=$!

echo "=== Waiting for server readiness ==="
READY=0
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models | grep -q 200; then
    READY=1
    break
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "Server failed to start within timeout. Last log lines:"
  tail -n 100 vllm_server.log
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

echo "=== Running inference test ==="
python scripts/test_inference.py

kill "$SERVER_PID" 2>/dev/null || true
