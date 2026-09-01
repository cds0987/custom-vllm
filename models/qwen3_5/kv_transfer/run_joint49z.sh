#!/usr/bin/env bash
# joint49z — pseudo-gold CoT that (user duyet 2026-09-01): 9B tu sinh CoT
# day du (ngan sach 24->200 token) cho 4 ho QUAN HE (musr, suite_rag/mid/swe),
# thay vi dap an ngan bi cat cut. Am tu joint49y (checkpoint tot nhat hien co).
#
# CHI DOI DUNG MOT BIEN so voi joint49y: --pseudo-gold. Moi thu khac GIU
# NGUYEN (max-ctx 4096, batch 2, accum 2, lora-t modules) — dung bai hoc
# "doi 4 bien cung luc" cua dot joint49x lan 1.
#
#   bash run_joint49z.sh            # sanity 40 buoc -> DUNG, cho doc so
#   GO=1 bash run_joint49z.sh       # chay that (1000 buoc, dung theo val)
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "0 moi truong"
for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done
python3 -c "import torch,peft,bitsandbytes;print('torch',torch.__version__)"

step "0b khoi phuc dau vao tu HF (recycle xoa sach /content)"
python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
pathlib.Path("/content/logs").mkdir(parents=True, exist_ok=True)
for name, dest in [("joint_v1/train_items.json", "/content/train_items.json"),
                   ("joint49_cot/pseudo_gold_v2.json", "/content/pseudo_gold_v2.json"),
                   ("evalbig/eval_big_items.json", "/content/eval_big_items.json")]:
    if pathlib.Path(dest).exists():
        print("da co", dest); continue
    try:
        p = hf_hub_download(REPO, name, token=os.environ.get("HF_TOKEN"))
        pathlib.Path(dest).write_bytes(pathlib.Path(p).read_bytes())
        print("KEO VE", dest, pathlib.Path(dest).stat().st_size, "byte")
    except Exception as e:
        print("khong keo duoc", name, type(e).__name__, str(e)[:80])
PYEOF

step "1 checkpoint am: joint49z (noi lai) hoac joint49y"
NAME="${OUTNAME:-joint49z}"
for want in "$NAME" joint49y; do
  [ -f "/content/$want/mapper_last.pt" ] && continue
  [ "$want" = joint49y ] && [ -f /content/joint49y/mapper_best.pt ] && continue
  python3 - "$want" <<'PYEOF' || true
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
done

INIT_M=/content/joint49y/mapper_best.pt
INIT_L=/content/joint49y/lora_best
INIT_T=""
if [ -f "/content/$NAME/mapper_last.pt" ]; then
  INIT_M="/content/$NAME/mapper_last.pt"
  [ -d "/content/$NAME/lora_last" ]  && INIT_L="/content/$NAME/lora_last"
  [ -d "/content/$NAME/lorat_last" ] && INIT_T="/content/$NAME/lorat_last"
  echo "NOI LAI tu $INIT_M"
else
  echo "BAT DAU MOI tu joint49y (mapper_best + lora_best)"
  [ -d /content/joint49y/lorat_best ] && INIT_T=/content/joint49y/lorat_best
fi
[ -f "$INIT_M" ] || { echo "KHONG THAY $INIT_M"; exit 1; }

MODS="${LORAT_MODS:-q_proj,o_proj,in_proj_qkvz,out_proj}"
BATCH="${BATCH:-2}"

# GIONG HET cau hinh joint49y (max-ctx 4096, batch 2, accum 2) — CHI doi
# --pseudo-gold sang ban CoT moi. Day la thi nghiem mot-bien.
COMMON="--tgt-model Qwen/Qwen3.5-9B
  --data-file /content/train_items.json
  --pseudo-gold /content/pseudo_gold_v2.json
  --max-ctx ${MAXCTX:-4096} --tbptt 128 --gold-cap 256 --gold-envelope 16384:256
  --drop-kinds ${DROP:-gsm8k,suite_math}
  --no-offload
  --batch $BATCH
  --lora-t $MODS --lora-t-r ${LORAT_R:-16}
  --init-mapper $INIT_M --init-lora $INIT_L"
[ -n "$INIT_T" ] && COMMON="$COMMON --init-lora-t $INIT_T"

if [ "${GO:-0}" != "1" ]; then
  step "SANITY 40 buoc — do toc do/VRAM voi gold DAI HON (CoT toi 200 token)"
  # Gold dai hon truoc day (24) co the doi s/buoc va peak VRAM — phai do
  # truoc khi phong 1000 buoc, dung ky luat da dung cho joint49x.
  python3 -u e9_joint.py $COMMON \
    --accum ${ACCUM:-2} --steps 40 --sanity 40 --val-every 100000 \
    --verify-meta 512 \
    --out "/content/${NAME}_sanity" \
    --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix "${NAME}_sanity"
  echo "SANITY XONG — doc s/buoc va peak GiB, so voi joint49y (2,82s/buoc, 18-19GiB)."
  echo "Chay that: GO=1 bash run_joint49z.sh"
  echo "RUN_49Z_SANITY_EXIT"
  exit 0
fi

step "TRAIN $NAME (${STEPS:-1000} buoc, dung theo val)"
python3 -u e9_joint.py $COMMON \
  --steps ${STEPS:-1000} --val-every ${VALEVERY:-200} --val-n ${VALN:-150} \
  --ce-floor 0.05 --patience ${PAT:-3} --accum ${ACCUM:-2} \
  --verify-meta 512 \
  --out "/content/$NAME" \
  --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix "$NAME"

step "XONG"
echo "RUN_49Z_EXIT"
