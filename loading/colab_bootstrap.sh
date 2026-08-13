#!/usr/bin/env bash
# One-command fast rebuild of a wiped Colab runtime, from nothing to a served
# champion. Written after nine runtime wipes, each costing ~40 minutes of
# sequential setup.
#
#   bash loading/colab_bootstrap.sh
#
# Where the old ~40 minutes went, and what this changes:
#
#   step                       before   after   how
#   -------------------------- -------  ------  -----------------------------
#   pip install vllm           ~12 min  ~12 min  (unavoidable)
#   llmcompressor + datasets   ~4 min   0       skipped unless CUSTOM_VLLM_TOOLS=1
#   plugin sdist rebuild       ~6 min   0       only rebuilt if the wheel is broken
#   download frame + GGUF      ~11 min  ~4 min  hf_transfer + both in parallel
#   graft                      ~5 min   ~5 min  (CPU-bound, already minimal)
#
# Set CHAMPION_REPO to a HF repo holding a prebuilt champion to skip the
# download+graft entirely (~9 min -> ~3 min); requires `huggingface-cli login`.
set -e
START=$(date +%s)
step() { echo; echo "=== [$(( $(date +%s) - START ))s] $* ==="; }

REPO_DIR="${REPO_DIR:-/content/custom_vllm}"
OUT="${OUT:-/content/champion}"
FRAME_REPO="${FRAME_REPO:-RedHatAI/Qwen3.5-9B-quantized.w4a16}"
GGUF_REPO="${GGUF_REPO:-unsloth/Qwen3.5-9B-GGUF}"
GGUF_FILE="${GGUF_FILE:-Qwen3.5-9B-Q4_K_M.gguf}"

step "Cloning repo"
[ -d "$REPO_DIR" ] || git clone -q https://github.com/cds0987/custom-vllm.git "$REPO_DIR"
cd "$REPO_DIR"

step "Environment"
bash loading/setup_env.sh 2>&1 | tail -25

export HF_HUB_ENABLE_HF_TRANSFER=1

# A prebuilt champion turns download+graft into a single pull: 13.8 GB of
# downloads plus a 5-minute CPU graft collapse into one 9.1 GB fetch. Try it
# first and fall back to building from source, so a missing/private repo costs
# nothing but a failed lookup.
CHAMPION_REPO="${CHAMPION_REPO:-gunnybd01/qwen35-9b-champion}"
PREBUILT_OK=0
if [ -n "$CHAMPION_REPO" ]; then
  step "Trying prebuilt champion: $CHAMPION_REPO"
  if python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$CHAMPION_REPO", local_dir="$OUT")
EOF
  then
    echo "Pulled prebuilt champion — skipping download + graft"
    PREBUILT_OK=1
  else
    echo "Prebuilt champion unavailable; building from source"
  fi
fi

if [ "$PREBUILT_OK" = "0" ]; then
  step "Downloading frame + GGUF IN PARALLEL"
  # Sequential downloads leave the link idle while the other file waits; these
  # two are independent, so overlap them and wait for both.
  python -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'
from huggingface_hub import snapshot_download
print('FRAME', snapshot_download('$FRAME_REPO'))" &
  PID_FRAME=$!
  python -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'
from huggingface_hub import hf_hub_download
print('GGUF', hf_hub_download('$GGUF_REPO','$GGUF_FILE'))" &
  PID_GGUF=$!
  wait $PID_FRAME || { echo "FRAME download failed"; exit 1; }
  wait $PID_GGUF  || { echo "GGUF download failed"; exit 1; }

  step "Grafting champion (CPU)"
  FRAME=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$FRAME_REPO'))")
  GGUF=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$GGUF_REPO','$GGUF_FILE'))")
  # NOTE: never run fix_qwen35_hf_checkpoint.py on the frame -- it is a
  # multimodal checkpoint and the fix corrupts it. RMS ~9.4% mean / 11.4% max
  # is the EXPECTED figure for --bits 4 (0.55% belongs to --bits 8).
  python models/qwen3_5/load/gguf_to_marlin.py \
      --frame "$FRAME" --gguf "$GGUF" --out "$OUT" --bits 4 --group-size 32
fi

step "Patching vLLM loader"
python models/qwen3_5/engine/vllm/patches/patch_vllm_gdn_quant_load.py

step "DONE in $(( ($(date +%s) - START) / 60 )) min $(( ($(date +%s) - START) % 60 ))s"
cat <<EOF

Champion: $OUT

Serve with:
  vllm serve $OUT \\
      --max-model-len 32768 --max-num-batched-tokens 1088 \\
      --enable-prefix-caching --mamba-cache-mode align \\
      --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.85

To make the next rebuild ~3 minutes instead of ~9, upload this champion once:
  huggingface-cli login
  huggingface-cli upload <you>/qwen35-9b-champion $OUT
then rebuild with:  CHAMPION_REPO=<you>/qwen35-9b-champion bash loading/colab_bootstrap.sh
EOF
