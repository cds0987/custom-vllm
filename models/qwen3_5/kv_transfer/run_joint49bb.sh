#!/usr/bin/env bash
# joint49bb — thu hep pham vi CHI con suite_swe (day du) + gsm8k (user chot
# 2026-09-01, xem quy tac 14 trong .claude/rules/quy-tac.md). Am tu joint49z
# (checkpoint tot nhat hien co — KHONG tu joint49aa, da xac nhan la buoc lui).
#
# CHI doi hai thu so voi joint49aa: (1) --drop-kinds mo rong loai het
# bbh/bfcl/needle/musr/suite_rag/suite_mid (chi giu suite_swe+gsm8k),
# (2) gold gsm8k cat DAU+DUOI thay vi chi cat DAU (hoc phi joint49aa: cat
# dau lam mat ket luan dap an vi CoT gsm8k luon "ly luan -> dap an cuoi").
#
#   bash run_joint49bb.sh            # sanity 40 buoc -> DUNG, cho doc so
#   GO=1 bash run_joint49bb.sh       # chay that (1000 buoc, dung theo val)
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

step "0b khoi phuc dau vao tu HF (recycle xoa sach /content)"
python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
pathlib.Path("/content/logs").mkdir(parents=True, exist_ok=True)
for name, dest in [("joint_v1/train_items.json", "/content/train_items.json"),
                   ("joint49_cot/pseudo_gold_v2.json", "/content/pseudo_gold_v2.json"),
                   ("joint49_cot/train_items_gsm.json", "/content/train_items_gsm.json"),
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
if [ ! -f /content/train_items_gsm.json ]; then
  echo "KHONG THAY train_items_gsm.json tren HF lan ca cuc bo -> can chay lai"
  echo "buoc 0c cua run_joint49aa.sh truoc (tao tap con gsm8k 400 mau)."
  exit 1
fi

step "0c sua gold gsm8k: cat DAU+DUOI (giu ket luan dap an), khong chi cat DAU"
# Hoc phi joint49aa: cat 50 token DAU luon mat ket luan vi CoT gsm8k luon
# ket thuc bang dap an/\boxed{}. Sua: giu HEAD token dau (ly luan) + TAIL
# token cuoi (ket luan), bo phan GIUA neu dai hon ngan sach.
if [ -f /content/pseudo_gold_gsm2.json ]; then
  echo "da co, bo qua"
else
  python3 - <<PYEOF
import json, pathlib
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
pg = json.loads(pathlib.Path("/content/pseudo_gold_v2.json").read_text())
CAP = ${GSM_GOLD_CAP:-50}
HEAD = ${GSM_GOLD_HEAD:-28}
TAIL = CAP - HEAD

pg2 = dict(pg)
n_cut = 0
for k, v in pg.items():
    if v.get("kind") == "gsm8k" and v.get("gold"):
        ids = tok(v["gold"], add_special_tokens=False)["input_ids"]
        if len(ids) > CAP:
            kept = ids[:HEAD] + ids[-TAIL:]
            pg2[k] = dict(v, gold=tok.decode(kept))
            n_cut += 1
pathlib.Path("/content/pseudo_gold_gsm2.json").write_text(json.dumps(pg2, ensure_ascii=False))
print(f"cat dau+duoi gsm8k: {n_cut} mau > {CAP} token da duoc cat (head={HEAD} tail={TAIL})")
PYEOF
  python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.upload_file(path_or_fileobj='/content/pseudo_gold_gsm2.json',
                path_in_repo='joint49_cot/pseudo_gold_gsm2.json',
                repo_id='gunnybd01/qwen35-kv-mapper-4b-27b')
print('HF-UP pseudo_gold_gsm2.json')
"
fi

step "1 checkpoint am: joint49bb (noi lai) hoac joint49z"
NAME="${OUTNAME:-joint49bb}"
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

# CHI con suite_swe + gsm8k (quy tac 14): loai het cac bo khac khoi train/val.
COMMON="--tgt-model Qwen/Qwen3.5-9B
  --data-file /content/train_items_gsm.json
  --pseudo-gold /content/pseudo_gold_gsm2.json
  --max-ctx ${MAXCTX:-4096} --tbptt 128 --gold-cap 256 --gold-envelope 16384:256
  --drop-kinds ${DROP:-bbh,bfcl,needle,musr,suite_rag,suite_mid,suite_math,ifstruct,pbtable}
  --no-offload
  --batch $BATCH
  --lora-t $MODS --lora-t-r ${LORAT_R:-16}
  --init-mapper $INIT_M --init-lora $INIT_L"
[ -n "$INIT_T" ] && COMMON="$COMMON --init-lora-t $INIT_T"

if [ "${GO:-0}" != "1" ]; then
  step "SANITY 40 buoc — do toc do/VRAM voi pham vi thu hep (suite_swe+gsm8k)"
  python3 -u e9_joint.py $COMMON \
    --accum ${ACCUM:-2} --steps 40 --sanity 40 --val-every 100000 \
    --verify-meta 512 \
    --out "/content/${NAME}_sanity" \
    --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix "${NAME}_sanity"
  echo "SANITY XONG — doc s/buoc va peak GiB, so voi joint49z (~19,6GiB)."
  echo "Chay that: GO=1 bash run_joint49bb.sh"
  echo "RUN_49BB_SANITY_EXIT"
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
echo "RUN_49BB_EXIT"
