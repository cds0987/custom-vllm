#!/usr/bin/env bash
# joint49aa — them gsm8k (tap con nho, user duyet 2026-09-01), am tu joint49z.
#
# VI SAO TAP CON: gsm8k co 3.000 mau trong train_items.json — do het se gan
# gap doi tap train hien tai (4.697), lam cham moi buoc va loang tin hieu
# musr/suite_swe (vua mang lai dot pha lon nhat chien dich). Lay 400 mau,
# UU TIEN cac mau DA CO pseudo-gold CoT dung (2.583/3.000 = 86,1% tu 9B)
# de tan dung ngay co che vua xay, khong can sinh gi them.
#
# CHI DOI DUNG HAI THU so voi joint49z: (1) bo gsm8k khoi --drop-kinds,
# (2) dung file train_items da lay tap con gsm8k. Moi thu khac GIU NGUYEN.
#
#   bash run_joint49aa.sh            # sanity 40 buoc -> DUNG, cho doc so
#   GO=1 bash run_joint49aa.sh       # chay that (1000 buoc, dung theo val)
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
# Hoc phi (dot dau, 2026-09-01): sanity do 21,74GiB/23 (sat hon joint49z
# 19,6GiB vi tron gsm8k dai hon) nhung LUOT TRAIN THAT van OOM ngay o buoc
# dau — thong bao loi "21,61GiB allocated, 171MiB reserved-but-unallocated"
# la dau hieu PHAN MANH, khong phai het bo nho that. Bat expandable_segments
# theo dung goi y trong thong bao loi.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

step "0c lay tap con gsm8k (${GSM_N:-400} mau, uu tien mau da co CoT dung)"
if [ -f /content/train_items_gsm.json ]; then
  echo "da co, bo qua"
else
  python3 - <<PYEOF
import json, random, pathlib
random.seed(42)
data = json.loads(pathlib.Path("/content/train_items.json").read_text())
pg = json.loads(pathlib.Path("/content/pseudo_gold_v2.json").read_text())
N = ${GSM_N:-400}

# ty le train:val ~ 4697:201 (~23,4x) — chia N theo dung ty le do de val
# khong bi thoi phong bat thuong so voi train.
TRAIN_LEN = len(data["train"]); VAL_LEN = len(data["val"])

def subset(split_items, n_this):
    gsm = [it for it in split_items if it["kind"] == "gsm8k"]
    other = [it for it in split_items if it["kind"] != "gsm8k"]
    have_cot = [it for it in gsm if pg.get(it.get("id",""), {}).get("gold")]
    no_cot = [it for it in gsm if not pg.get(it.get("id",""), {}).get("gold")]
    random.shuffle(have_cot); random.shuffle(no_cot)
    picked = (have_cot + no_cot)[:n_this]
    return other + picked, len(picked), sum(1 for p in picked if pg.get(p.get("id",""),{}).get("gold"))

n_tr_target = max(1, round(N * TRAIN_LEN / (TRAIN_LEN + VAL_LEN)))
n_val_target = max(1, N - n_tr_target)
data["train"], n_tr, cot_tr = subset(data["train"], n_tr_target)
data["val"], n_val, cot_val = subset(data["val"], n_val_target)
pathlib.Path("/content/train_items_gsm.json").write_text(json.dumps(data))
print(f"gsm8k subset: train {n_tr} mau ({cot_tr} co CoT dung), "
      f"val {n_val} mau ({cot_val} co CoT dung)")
PYEOF
  python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.upload_file(path_or_fileobj='/content/train_items_gsm.json',
                path_in_repo='joint49_cot/train_items_gsm.json',
                repo_id='gunnybd01/qwen35-kv-mapper-4b-27b')
print('HF-UP train_items_gsm.json OK')
"
fi

step "1 checkpoint am: joint49aa (noi lai) hoac joint49z"
NAME="${OUTNAME:-joint49aa}"
for want in "$NAME" joint49z; do
  [ -f "/content/$want/mapper_last.pt" ] && continue
  [ "$want" = joint49z ] && [ -f /content/joint49z/mapper_best.pt ] && continue
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

INIT_M=/content/joint49z/mapper_best.pt
INIT_L=/content/joint49z/lora_best
INIT_T=""
if [ -f "/content/$NAME/mapper_last.pt" ]; then
  INIT_M="/content/$NAME/mapper_last.pt"
  [ -d "/content/$NAME/lora_last" ]  && INIT_L="/content/$NAME/lora_last"
  [ -d "/content/$NAME/lorat_last" ] && INIT_T="/content/$NAME/lorat_last"
  echo "NOI LAI tu $INIT_M"
else
  echo "BAT DAU MOI tu joint49z (mapper_best + lora_best)"
  [ -d /content/joint49z/lorat_best ] && INIT_T=/content/joint49z/lorat_best
fi
[ -f "$INIT_M" ] || { echo "KHONG THAY $INIT_M"; exit 1; }

MODS="${LORAT_MODS:-q_proj,o_proj,in_proj_qkvz,out_proj}"
BATCH="${BATCH:-2}"

# CHI doi so voi joint49z: --data-file (tap con gsm8k) va --drop-kinds (bo
# gsm8k khoi danh sach loai). Moi thu khac GIONG HET.
COMMON="--tgt-model Qwen/Qwen3.5-9B
  --data-file /content/train_items_gsm.json
  --pseudo-gold /content/pseudo_gold_v2.json
  --max-ctx ${MAXCTX:-4096} --tbptt 128 --gold-cap 256 --gold-envelope 16384:256
  --drop-kinds ${DROP:-suite_math}
  --no-offload
  --batch $BATCH
  --lora-t $MODS --lora-t-r ${LORAT_R:-16}
  --init-mapper $INIT_M --init-lora $INIT_L"
[ -n "$INIT_T" ] && COMMON="$COMMON --init-lora-t $INIT_T"

if [ "${GO:-0}" != "1" ]; then
  step "SANITY 40 buoc — do toc do/VRAM voi gsm8k tron vao (item dai hon)"
  python3 -u e9_joint.py $COMMON \
    --accum ${ACCUM:-2} --steps 40 --sanity 40 --val-every 100000 \
    --verify-meta 512 \
    --out "/content/${NAME}_sanity" \
    --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix "${NAME}_sanity"
  echo "SANITY XONG — doc s/buoc va peak GiB, so voi joint49z (~6s/buoc, 19,6GiB)."
  echo "Chay that: GO=1 bash run_joint49aa.sh"
  echo "RUN_49AA_SANITY_EXIT"
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
echo "RUN_49AA_EXIT"
