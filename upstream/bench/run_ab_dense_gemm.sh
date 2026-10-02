#!/usr/bin/env bash
# A/B do cho PR "dense GEMM cho lo lon" tren vllm-gguf-plugin MAIN + dung ban va.
# Moi truong RIENG (/content/pr_env) -- khong dung env da patch cua custom-vllm,
# de so do la cua upstream sach + MOT thay doi.
#
#   bash upstream/bench/run_ab_dense_gemm.sh            # setup + pytest + A/B
#   MODEL=... TOKENIZER=... THRESHOLDS="0 1024" bash ...
#
# Idempotent: buoc nao xong roi thi bo qua. Ket qua: /content/pr_bench/*.json
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV=/content/pr_env
SRC=/content/pr_plugin
OUT=/content/pr_bench
MODEL="${MODEL:-unsloth/Qwen3-8B-GGUF:Q4_K_M}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
THRESHOLDS="${THRESHOLDS:-0 1024}"
PORT="${PORT:-8100}"
mkdir -p "$OUT"

echo "=== [1/4] moi truong"
# Hai bay da dinh tren Colab (2026-10-02):
#  1. Colab dat UV_SYSTEM_PYTHON=true -> `uv pip` cai vao python he thong, BO QUA
#     venv dang activate. Phai unset.
#  2. Cai thang vao he thong thi torch moi dung torchaudio cai san (lech ban CUDA)
#     -> vllm khong import noi. => BAT BUOC venv rieng.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
unset UV_SYSTEM_PYTHON
export PATH="$HOME/.local/bin:$PATH"
[ -x "$ENV/bin/python" ] || uv venv --python 3.12 "$ENV"
export VIRTUAL_ENV="$ENV" PATH="$ENV/bin:$PATH"
PY="$ENV/bin/python"
$PY -c "import vllm" 2>/dev/null || uv pip install vllm --torch-backend=auto
$PY -c "import pytest, requests" 2>/dev/null || uv pip install pytest requests
[ "$(command -v vllm)" = "$ENV/bin/vllm" ] || { echo "vllm KHONG nam trong venv: $(command -v vllm)"; exit 1; }

echo "=== [2/4] plugin main + ban va"
if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/vllm-project/vllm-gguf-plugin.git "$SRC"
fi
cd "$SRC"
if ! grep -q VLLM_GGUF_DENSE_GEMM_MIN_ROWS vllm_gguf_plugin/quantization/linear.py; then
  git apply "$HERE/../patches/06-dense-gemm-large-batch.patch" || { echo "BAN VA KHONG AP DUOC"; exit 1; }
fi
git log --oneline -1 | tee "$OUT/plugin_commit.txt"
uv pip show vllm-gguf-plugin >/dev/null 2>&1 || uv pip install -e . --no-build-isolation
$PY - <<'PYEOF' | tee "$OUT/env.txt"
import torch, vllm, vllm_gguf_plugin.ops as ops
print("vllm", vllm.__version__, "| torch", torch.__version__,
      "| gpu", torch.cuda.get_device_name(0),
      "| cuda_ext_loaded", ops._CUDA_AVAILABLE, "| cuda_enabled", ops._CUDA_ENABLED)
PYEOF

echo "=== [3/4] pytest (GPU)"
$PY -m pytest tests/test_kernels.py -k "dense_gemm" -q 2>&1 | tail -15 | tee "$OUT/pytest.txt"

echo "=== [4/4] A/B"
for THR in $THRESHOLDS; do
  [ -f "$OUT/thr${THR}.json" ] && { echo "thr=$THR da co, bo qua"; continue; }
  LOG="$OUT/serve_thr${THR}.log"
  VLLM_GGUF_DENSE_GEMM_MIN_ROWS="$THR" nohup vllm serve "$MODEL" --tokenizer "$TOKENIZER" \
    --served-model-name ab --port "$PORT" --max-model-len 16384 --max-num-seqs 64 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching > "$LOG" 2>&1 &
  SPID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null && break
    kill -0 "$SPID" 2>/dev/null || { echo "SERVER CHET (thr=$THR):"; tail -30 "$LOG"; exit 1; }
    sleep 5
  done
  curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "SERVER KHONG LEN"; tail -30 "$LOG"; kill "$SPID"; exit 1; }
  $PY -u "$HERE/ab_dense_gemm.py" --url "http://localhost:$PORT/v1/completions" \
    --model ab --label "thr${THR}" --out "$OUT/thr${THR}.json"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee "$OUT/vram_thr${THR}.txt"
  kill "$SPID"; wait "$SPID" 2>/dev/null
  while ss -tln | grep -q ":$PORT "; do sleep 2; done
done
echo "=== XONG -> $OUT"; ls -la "$OUT"
