#!/usr/bin/env bash
# gsm_grpo -- pipeline GRPO HOP NHAT (eba_grpo.py --task gsm8k): dung LAI
# dung engine RL da kiem chung tren EBA (mapper + LoRA-4B + LoRA-9B, K nhanh
# gop lo, anchor-CE) nhung tren gsm8k THAT + pseudo-gold 9B tu sinh (da co
# san tren HF joint49_cot/pseudo_gold_gsm2.json -- 9B tu giai, CHI giu quy
# dao item no lam DUNG, item sai giu gold goc, khong bao gio day mapper hoc
# theo suy luan sai). User 2026-09-04: "aggregate thanh 1 full pipeline...
# sinh report table so THAT cho gsm8k thay vi so proxy".
#
# Warm-start MAC DINH tu eba_grpo_v2c/best (da RL-tune tren EBA, buoc nay
# RL THEM tren gsm8k that de sac net hoa dung so cuoi) -- co the doi qua CK
# khac (vd joint49cc) bang bien CK.
#
#   bash run_gsm_grpo.sh                 # sanity 5 buoc
#   GO=1 bash run_gsm_grpo.sh            # chay that (400 buoc mac dinh)
#   GO=1 STEPS=800 K=4 bash run_gsm_grpo.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "0 moi truong + du lieu"
for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done
python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
for name, dest in [("joint49_cot/train_items_gsm.json", "/content/train_items_gsm.json"),
                   ("joint49_cot/pseudo_gold_gsm2.json", "/content/pseudo_gold_gsm2.json")]:
    if pathlib.Path(dest).exists():
        print("da co", dest); continue
    try:
        p = hf_hub_download(REPO, name, token=os.environ.get("HF_TOKEN"))
        pathlib.Path(dest).write_bytes(pathlib.Path(p).read_bytes())
        print("KEO VE", dest, pathlib.Path(dest).stat().st_size, "byte")
    except Exception as e:
        print("khong keo duoc", name, type(e).__name__, str(e)[:80])
PYEOF

step "1 checkpoint warm-start (BAT BUOC -- thieu se vi pham dieu kien SFT-truoc-RL)"
CK="${CK:-eba_grpo_v2c}"
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
        print("KEO VE", name)
    else:
        print(f"HF chua co {name}/mapper_*.pt")
except Exception as e:
    print(f"khong lay duoc {name}:", type(e).__name__, str(e)[:80])
PYEOF
[ -f "/content/$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK, DUNG"; exit 1; }

step "2 chay eba_grpo.py --task gsm8k"
SANITY="${SANITY:-5}"
STEPS="${STEPS:-400}"
K="${K:-4}"
GEN_LEN="${GEN_LEN:-200}"
GSM_LIMIT="${GSM_LIMIT:-1200}"
SNAPSHOT="${SNAPSHOT:-100}"
OUT="${OUT:-gsm_grpo_v1}"
if [ "${GO:-0}" = "1" ]; then
  ARGS="--steps $STEPS --sanity 0"
else
  ARGS="--sanity $SANITY"
fi
python3 -u eba_grpo.py --task gsm8k \
  --init-mapper "/content/$CK/mapper_best.pt" \
  --init-lora "/content/$CK/lora_best" \
  --init-lora-t "/content/$CK/lorat_best" \
  --k "$K" --gen-len "$GEN_LEN" --gold-cap 256 \
  --gsm-data /content/train_items_gsm.json \
  --pseudo-gold /content/pseudo_gold_gsm2.json \
  --gsm-limit "$GSM_LIMIT" \
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

[ "$STATUS" = 0 ] && echo "RUN_GSM_GRPO_EXIT" || echo "RUN_GSM_GRPO_FAIL status=$STATUS"
