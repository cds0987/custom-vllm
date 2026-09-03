#!/usr/bin/env bash
# eba_grpo — buoc RL (GRPO) sau SFT, tren du lieu synthetic Entity Binding
# Arithmetic (eba_gen.py). Xem docstring eba_grpo.py va TRANG-THAI.md muc
# "Unsloth GRPO" cho ly do thiet ke.
#
# MAC DINH CHI SANITY (5 buoc, khong val, khong luu ckpt) -- doc so VRAM/toc
# do TRUOC khi cam ket --steps day (dung ky luat du an: khong phong train
# dai khi chua do duoc chi phi that).
#
#   bash run_eba_grpo.sh                 # sanity 5 buoc
#   GO=1 bash run_eba_grpo.sh            # chay that (300 buoc mac dinh)
#   GO=1 STEPS=600 K=8 bash run_eba_grpo.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "0 moi truong"
for pkg in peft bitsandbytes; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done

step "1 checkpoint SFT lam warm-start (BAT BUOC -- thieu no reward se dong nhat, GRPO khong hoc duoc gi)"
CK="${CK:-joint49cc}"
[ -f "/content/$CK/mapper_best.pt" ] || python3 - "$CK" <<'PYEOF' || true
import os, sys, shutil, pathlib
from huggingface_hub import snapshot_download
name = sys.argv[1]
try:
    p = snapshot_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                          allow_patterns=[f"{name}/*"],
                          local_dir=f"/content/_hf_{name}",
                          token=os.environ.get("HF_TOKEN"))
    src = pathlib.Path(p) / name
    if any(src.glob("mapper_*.pt")):
        shutil.copytree(src, f"/content/{name}", dirs_exist_ok=True)
        print("KEO VE", name, sorted(x.name for x in src.iterdir()))
    else:
        print(f"HF chua co {name}/mapper_*.pt")
except Exception as e:
    print(f"khong lay duoc {name}:", type(e).__name__, str(e)[:80])
PYEOF
[ -f "/content/$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK, DUNG"; exit 1; }

step "2 chay eba_grpo.py"
SANITY="${SANITY:-5}"
STEPS="${STEPS:-300}"
K="${K:-6}"
N_ITEMS="${N_ITEMS:-400}"
DIFF_MAX="${DIFF_MAX:-1}"
SNAPSHOT="${SNAPSHOT:-0}"
OUT="${OUT:-eba_grpo_v1}"
if [ "${GO:-0}" = "1" ]; then
  ARGS="--steps $STEPS --sanity 0"
else
  ARGS="--sanity $SANITY"
fi
python3 -u eba_grpo.py \
  --init-mapper "/content/$CK/mapper_best.pt" \
  --init-lora "/content/$CK/lora_best" \
  --init-lora-t "/content/$CK/lorat_best" \
  --k "$K" --n-items "$N_ITEMS" --difficulty-max "$DIFF_MAX" \
  --snapshot-every "$SNAPSHOT" \
  --out "/content/$OUT" \
  --hf-prefix "$OUT" \
  $ARGS 2>&1 | tee "/content/logs/${OUT}.log"
STATUS=${PIPESTATUS[0]}

python3 - "$OUT" <<'PYEOF' || true
import os, pathlib, sys
from huggingface_hub import HfApi
out = sys.argv[1]
api = HfApi(token=os.environ.get("HF_TOKEN"))
p = pathlib.Path(f"/content/logs/{out}.log")
if p.exists():
    try:
        api.upload_file(path_or_fileobj=str(p), repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                        path_in_repo=f"{out}/{out}.log")
        print("HF-UP", f"{out}.log")
    except Exception as e:
        print("HF-UP FAIL", type(e).__name__, str(e)[:80])
PYEOF

[ "$STATUS" = 0 ] && echo "RUN_EBA_GRPO_EXIT" || echo "RUN_EBA_GRPO_FAIL status=$STATUS"
