#!/usr/bin/env bash
# joint49cc — MO DUNG LUONG PHAN GDN CUA MAPPER (--gdn-terms 4).
#
# VI SAO: probe trich-xuat-so (2026-09-02) cho thay 9B KHONG nhac lai noi con
# so co san trong ngu canh (so DAU 50%, so CUOI 15%) -> thong tin so khong toi
# duoc nguyen ven, KHONG phai loi khau suy luan. Nen --w-entity vo dung, don
# bay dung la dung luong phan GDN: hien 0,8M/17,6M tham so va CHI MOT so hang
# A.S.B, trong khi theo dinh luat E7 thi GDN chinh la noi mang QUAN HE.
#
# CHI DOI DUNG MOT BIEN so voi joint49bb: --gdn-terms 1 -> 4. Moi thu khac
# GIU NGUYEN (max-ctx 4096, batch 2, accum 2, cung tap suite_swe+gsm8k,
# cung pseudo-gold). So hang r>0 khoi tao BANG 0 nen tai buoc 0 dau ra GIONG
# HET joint49bb -> warm-start khong tut diem (test_mapper_terms.py 23/23).
#
#   bash run_joint49cc.sh            # sanity 40 buoc -> DUNG, cho doc so
#   GO=1 bash run_joint49cc.sh       # chay that (1000 buoc, dung theo val)
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "0 moi truong"
for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done
python3 -c "import torch,peft,bitsandbytes;print('torch',torch.__version__)"

step "0b khoi phuc dau vao tu HF (runtime recycle 3 lan trong phien 02-09)"
python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
pathlib.Path("/content/logs").mkdir(parents=True, exist_ok=True)
for name, dest in [("joint49_cot/train_items_gsm.json", "/content/train_items_gsm.json"),
                   ("joint49_cot/pseudo_gold_gsm2.json", "/content/pseudo_gold_gsm2.json"),
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

step "1 checkpoint am: joint49cc (noi lai) hoac joint49bb"
NAME="${OUTNAME:-joint49cc}"
for want in "$NAME" joint49bb; do
  [ -f "/content/$want/mapper_last.pt" ] && continue
  [ "$want" = joint49bb ] && [ -f /content/joint49bb/mapper_best.pt ] && continue
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

INIT_M=/content/joint49bb/mapper_best.pt
INIT_L=/content/joint49bb/lora_best
INIT_T=""
if [ -f "/content/$NAME/mapper_last.pt" ]; then
  INIT_M="/content/$NAME/mapper_last.pt"
  [ -d "/content/$NAME/lora_last" ]  && INIT_L="/content/$NAME/lora_last"
  [ -d "/content/$NAME/lorat_last" ] && INIT_T="/content/$NAME/lorat_last"
  echo "NOI LAI tu $INIT_M"
else
  echo "BAT DAU MOI tu joint49bb (mapper_best + lora_best), gdn-terms 1 -> ${TERMS:-4}"
  [ -d /content/joint49bb/lorat_best ] && INIT_T=/content/joint49bb/lorat_best
fi
[ -f "$INIT_M" ] || { echo "KHONG THAY $INIT_M"; exit 1; }

MODS="${LORAT_MODS:-q_proj,o_proj,in_proj_qkvz,out_proj}"
BATCH="${BATCH:-2}"

# GIONG HET joint49bb, CHI THEM --gdn-terms.
COMMON="--tgt-model Qwen/Qwen3.5-9B
  --data-file /content/train_items_gsm.json
  --pseudo-gold /content/pseudo_gold_gsm2.json
  --max-ctx ${MAXCTX:-4096} --tbptt 128 --gold-cap 256 --gold-envelope 16384:256
  --drop-kinds ${DROP:-bbh,bfcl,needle,musr,suite_rag,suite_mid,suite_math,ifstruct,pbtable}
  --no-offload
  --batch $BATCH
  --gdn-terms ${TERMS:-4}
  --lora-t $MODS --lora-t-r ${LORAT_R:-16}
  --init-mapper $INIT_M --init-lora $INIT_L"
[ -n "$INIT_T" ] && COMMON="$COMMON --init-lora-t $INIT_T"

if [ "${GO:-0}" != "1" ]; then
  step "SANITY 40 buoc — do toc do/VRAM voi gdn-terms ${TERMS:-4}"
  # So voi joint49bb: 3,15s/buoc, peak 17,18GiB. GDN 0,8M -> ~3,2M tham so
  # (khong dang ke ve VRAM) nhung backward qua nhieu so hang co the cham hon.
  python3 -u e9_joint.py $COMMON \
    --accum ${ACCUM:-2} --steps 40 --sanity 40 --val-every 100000 \
    --verify-meta 512 \
    --out "/content/${NAME}_sanity" \
    --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix "${NAME}_sanity"
  echo "SANITY XONG — so voi joint49bb (3,15s/buoc, 17,18GiB)."
  echo "Chay that: GO=1 bash run_joint49cc.sh"
  echo "RUN_49CC_SANITY_EXIT"
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
echo "RUN_49CC_EXIT"
