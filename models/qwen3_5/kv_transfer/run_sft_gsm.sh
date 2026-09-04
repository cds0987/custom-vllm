#!/usr/bin/env bash
# SFT gsm8k DAY DU 1 EPOCH tren pool lon (user 2026-09-04: "chay lai sft voi
# du 1 epochs"). Khac lan probe (300 buoc / 300 mau): dung joint_v1/
# train_items.json co 2.882 gsm8k, trong do 2.483 co CoT 9B tu sinh giai DUNG.
# Luu checkpoint dinh ky (Colab recycle bat cu luc nao, 1 epoch ~2 gio).
#
#   bash run_sft_gsm.sh                  # sanity 5 buoc
#   GO=1 bash run_sft_gsm.sh             # chay that 1 epoch
#   GO=1 EPOCHS=1 bash run_sft_gsm.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
EPOCHS="${EPOCHS:-1}"
CK="${CK:-gsm_grpo_v1c}"
OUT="${OUT:-sft_gsm_v1}"

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done

echo "=== keo du lieu + checkpoint tu HF ==="
python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
for name, dest in [("joint_v1/train_items.json", "/content/train_items.json"),
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

[ -f "/content/$CK/mapper_best.pt" ] || python3 - "$CK" <<'PYEOF' || true
import os, sys, shutil, pathlib
from huggingface_hub import snapshot_download
name = sys.argv[1]
try:
    p = snapshot_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                          allow_patterns=[f"{name}/*"], local_dir=f"/content/_hf_{name}",
                          token=os.environ.get("HF_TOKEN"))
    src = pathlib.Path(p) / name
    if any(src.glob("mapper_*.pt")):
        shutil.copytree(src, f"/content/{name}", dirs_exist_ok=True)
        print("KEO VE", name)
except Exception as e:
    print("khong lay duoc", name, type(e).__name__, str(e)[:80])
PYEOF
[ -f "/content/$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK, DUNG"; exit 1; }

if [ "${GO:-0}" = "1" ]; then
  ARGS="--epochs $EPOCHS --sanity 0"
else
  ARGS="--epochs $EPOCHS --sanity 5"
fi
python3 -u sft_gsm.py \
  --init-dir "/content/$CK" \
  --data-file /content/train_items.json \
  --pseudo-gold /content/pseudo_gold_gsm2.json \
  --out "/content/$OUT" --hf-prefix "$OUT" \
  $ARGS 2>&1 | tee "/content/logs/${OUT}.log"
STATUS=${PIPESTATUS[0]}
[ "$STATUS" = 0 ] && echo "RUN_SFT_GSM_EXIT" || echo "RUN_SFT_GSM_FAIL status=$STATUS"
