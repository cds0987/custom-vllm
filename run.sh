#!/usr/bin/env bash
# custom-vllm — ONE command for everything, vLLM-style.
#
# The whole product on Colab is exactly ONE cell:
#
#     !git clone -q https://github.com/cds0987/custom-vllm.git /content/custom_vllm 2>/dev/null; \
#      cd /content/custom_vllm && git pull -q && bash run.sh serve 9b
#
# Need something else? ADD A COMMAND to that cell, not a new cell:
#
#     bash run.sh setup              # env + all patches (idempotent, auto-run by serve)
#     bash run.sh serve 9b|9b-spec   # pull champion 9B + serve (-spec: ngram speculative)
#     bash run.sh serve 27b          # pull 27B frame + serve with tuned L4 config
#     bash run.sh status             # GPU / server / model state, tail of logs
#     bash run.sh logs               # follow server log
#     bash run.sh bench <script> ... # run any bench/ script against the live server
#     bash run.sh eval 27b|9b        # ppl quality gate (99 SWE-bench instances)
#     bash run.sh stop               # stop the server
#
# Every command is idempotent: rerunning is always safe.
set -e
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
LOGS=/content/logs; mkdir -p "$LOGS" 2>/dev/null || LOGS="$REPO_DIR/out/logs"; mkdir -p "$LOGS"

MODELS_DIR="${MODELS_DIR:-/content/models}"
CHAMPION_9B_REPO="${CHAMPION_9B_REPO:-gunnybd01/qwen35-9b-champion}"
FRAME_27B_REPO="${FRAME_27B_REPO:-apolo13x/Qwen3.5-27B-quantized.w4a16}"

say() { echo; echo "=== [run.sh] $* ==="; }

port_open() { curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; }

ensure_setup() {
  # Marker-based: setup once per runtime, cheap check afterwards.
  if python -c "import vllm" 2>/dev/null && [ -f /tmp/custom_vllm_setup_done ]; then
    say "setup: OK (cached)"; return 0
  fi
  say "setup: installing env + patches (~3-5 min on a fresh runtime)"
  bash loading/setup_env.sh 2>&1 | tail -15
  touch /tmp/custom_vllm_setup_done
}

pull_model() {  # $1 = 9b|27b  -> prints local dir
  case "$1" in
    9b)
      python - "$CHAMPION_9B_REPO" "$MODELS_DIR/champion9b" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=8))
EOF
      ;;
    27b)
      python - "$FRAME_27B_REPO" "$MODELS_DIR/frame27b" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=8))
EOF
      ;;
    *) echo "unknown model: $1 (want 9b|27b)"; exit 1;;
  esac
}

serve() {  # $1 = 9b|27b, optionally with -spec suffix (ngram speculative decoding)
  local variant="$1"; shift || true
  local base="${variant%-spec}"
  ensure_setup
  say "pulling model $base"
  pull_model "$base" >/dev/null
  local model flags
  # Tuned configs — measured on L4 22.5GB, see models/qwen3_5/hardware/l4.py and STATUS.md
  case "$base" in
    9b)
      model="$MODELS_DIR/champion9b"
      flags="--max-model-len 65536 --max-num-batched-tokens 1088" ;;
    27b)
      model="$MODELS_DIR/frame27b"
      flags="--max-model-len 8192 --max-num-batched-tokens 512 --max-num-seqs 8 \
             --compilation-config {\"cudagraph_capture_sizes\":[1,2,4,8],\"max_cudagraph_capture_size\":8} \
             --gpu-memory-utilization 0.97" ;;
    *) echo "unknown model: $base (want 9b|27b, optional -spec)"; exit 1;;
  esac
  [ "$base" = "9b" ] && flags="$flags --gpu-memory-utilization 0.85"
  if [ "$variant" != "$base" ]; then
    say "speculative decoding: ngram k=4 (prompt-lookup 2-4)"
    flags="$flags --speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"
  fi
  say "stopping any old server"
  pkill -f "vllm serve" 2>/dev/null || true
  local t0=$(date +%s)
  while port_open && [ $(( $(date +%s) - t0 )) -lt 120 ]; do sleep 3; done
  sleep 5
  say "starting vllm serve $model"
  # shellcheck disable=SC2086
  ( source /tmp/vllm_env.sh 2>/dev/null; \
    nohup vllm serve "$model" $flags \
      --enable-prefix-caching --mamba-cache-mode align \
      --kv-cache-dtype fp8_e4m3 --port 8000 \
      > "$LOGS/serve.log" 2>&1 & echo "PID:$!" )
  t0=$(date +%s)
  while ! port_open; do
    if [ $(( $(date +%s) - t0 )) -gt 900 ]; then say "TIMEOUT — last log:"; tail -20 "$LOGS/serve.log"; exit 1; fi
    if ! pgrep -f "vllm serve" >/dev/null; then say "SERVER DIED — root cause:"; grep -m3 -B2 "ERROR\|OOM\|ValueError" "$LOGS/serve.log" | tail -12; exit 1; fi
    sleep 10
  done
  say "SERVER READY after $(( $(date +%s) - t0 ))s"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  grep -m1 "GPU KV cache size" "$LOGS/serve.log" || true
  echo "OpenAI-compatible API: http://127.0.0.1:8000/v1  (model: $model)"
}

case "${1:-help}" in
  setup)  ensure_setup ;;
  serve)  shift; serve "${1:-9b}" ;;
  stop)   pkill -f "vllm serve" 2>/dev/null && echo "stopped" || echo "nothing running" ;;
  status)
    echo "--- GPU:"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "no GPU"
    echo "--- server:"; if port_open; then curl -s -m 2 http://127.0.0.1:8000/v1/models | head -c 300; echo; else echo "not running"; fi
    echo "--- serve.log tail:"; tail -15 "$LOGS/serve.log" 2>/dev/null || echo "(no log)"
    ;;
  logs)   tail -f "$LOGS/serve.log" ;;
  bench)  shift; s="$1"; shift; python "bench/$s.py" "$@" ;;
  eval)
    shift; v="${1:-9b}"
    m="$MODELS_DIR/champion9b"; [ "$v" = "27b" ] && m="$MODELS_DIR/frame27b"
    VLLM_MODEL="$m" python bench/eval_quality_swebench.py run --num-instances 100 --concurrency 2 \
      --output "$LOGS/quality_$v.json" && python -c "import json;d=json.load(open('$LOGS/quality_$v.json'));print('ppl:',d['ppl'])"
    ;;
  registry) shift; python register.py "$@" ;;
  help|*)
    grep '^#' "$0" | sed -n '2,20p' | sed 's/^# \{0,1\}//'
    ;;
esac
