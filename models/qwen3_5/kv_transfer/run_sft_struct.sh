#!/usr/bin/env bash
# BUOC 2 -- SFT day CAU TRUC (<think>/ENTITIES/STEPS/Final Answer).
#
# Mac dinh: DO BATCH LON NHAT chay duoc (user chon phuong an (a) 2026-09-05),
# thu B=2,4,6,8 moi cai 8 buoc sanity, cai nao OOM thi dung, roi chay THAT
# 1 epoch voi B lon nhat song + GA cho batch hieu dung ~16.
#
#   bash run_sft_struct.sh            # do batch roi in de xuat
#   GO=1 B=4 bash run_sft_struct.sh   # chay that 1 epoch voi B da chon
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CK="${CK:-gsm_grpo_v1}"
OUT="${OUT:-sft_struct_v1}"
EFF="${EFF:-16}"        # batch hieu dung mong muon = B * accum

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done

echo "=== keo du lieu + checkpoint ==="
python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
for name, dest in [("joint_v1/train_items.json", "/content/train_items.json"),
                   ("struct_gold/struct_gold_gsm.json", "/content/struct_gold_gsm.json")]:
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
  B="${B:-4}"
  ACC=$(( EFF / B )); [ "$ACC" -lt 1 ] && ACC=1
  echo "=== CHAY THAT 1 epoch: B=$B accum=$ACC (batch hieu dung $(( B * ACC ))) ==="
  python3 -u sft_struct.py --init-dir "/content/$CK" --epochs 1 \
    --batch "$B" --accum "$ACC" --out "/content/$OUT" --hf-prefix "$OUT" \
    2>&1 | tee "/content/logs/${OUT}.log"
  exit ${PIPESTATUS[0]}
fi

echo "=== DO BATCH LON NHAT CHAY DUOC ==="
BEST=1
for B in 2 4 6 8; do
  echo; echo "--- thu B=$B ---"
  if python3 -u sft_struct.py --init-dir "/content/$CK" --batch "$B" --accum 1 \
       --sanity 8 --out /content/_probe_batch --hf-repo "" \
       > "/content/logs/probe_B${B}.log" 2>&1; then
    P=$(grep -o 'peak=[0-9.]*GiB' "/content/logs/probe_B${B}.log" | tail -1)
    S=$(grep -o '[0-9.]*s/buoc' "/content/logs/probe_B${B}.log" | tail -1)
    echo "B=$B OK  $P  $S"
    BEST=$B
  else
    if grep -q 'OutOfMemory' "/content/logs/probe_B${B}.log"; then
      echo "B=$B OOM -> dung do"
    else
      echo "B=$B LOI khac:"; tail -5 "/content/logs/probe_B${B}.log"
    fi
    break
  fi
done
ACC=$(( EFF / BEST )); [ "$ACC" -lt 1 ] && ACC=1
echo
echo "=== DE XUAT: B=$BEST accum=$ACC (batch hieu dung $(( BEST * ACC ))) ==="
echo "chay that:  GO=1 B=$BEST bash run_sft_struct.sh"
echo "RUN_SFT_STRUCT_PROBE_EXIT"
