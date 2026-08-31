#!/usr/bin/env bash
# joint49x — DOT GOP (user duyet 2026-08-31): LoRA tren 9B (phia DOC) + gom lo
# theo do dai. Am tu joint49w.
#
#   bash run_joint49x.sh            # sanity 40 buoc -> DUNG, cho doc so
#   GO=1 bash run_joint49x.sh       # chay that (1000 buoc, dung theo val)
#
# VI SAO LoRA 9B: kien truc dang BAT DOI XUNG — phia ma hoa duoc thich nghi
# (LoRA 4B), phia dich duoc train (mapper), phia DOC bi DONG BANG. Mapper vi
# the phai nham trung dung phan phoi cache ban dia cua 9B, mot cai dich rat
# hep. Dang chu y hon: LoRA 4B hien chi cham q/k/v/o_proj = CHI lop attention
# (8/32 lop) — chua adapter nao cham GDN o BAT KY dau nao, ma E7 chi ra GDN
# moi la cho lech (CCA 0,23-0,9 so voi attention 0,93-0,98).
#
# CHOT CHAN BAT BUOC: them dung luong vao phia SINH lam viec "an gian" de hon
# — LoRA 9B co the hoc thang tac vu thay vi hoc doc cache. Sau train PHAI chay
# doi chung ctx-BO; bo ngu canh ma diem khong sap = dang thuoc bai -> bo.
#
# Moc phan xu dat TRUOC khi chay:
#   suite_swe > 60%  VA ctx-BO van sap  -> thanh cong
#   suite_swe <= 60%                    -> khong phai phia doc; nghi pham quay
#                                          ve DANG HAM A.S.B cua mapper
#   diem len nhung ctx-BO KHONG sap     -> an gian, bo va siet lai
set -u
cd "$(dirname "$0")"
KV="$(pwd)"
mkdir -p /content/logs

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "0 moi truong"
# CAI TUNG GOI MOT: `pip -q install A B` ma B hong build thi A CUNG khong
# duoc cai, va -q nuot loi (hoc phi trong docs/03-bug-va-cach-sua.md).
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
                   ("joint49/pseudo_gold.json", "/content/pseudo_gold.json"),
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

step "1 checkpoint am: joint49x (noi lai) hoac joint49w"
# Uu tien ban local -> ban joint49x tren HF -> joint49w. Colab thu hoi runtime
# ~1,5 gio/lan; khong noi lai thi moi lan recycle la mat sach gio train.
NAME="${OUTNAME:-joint49x}"
for want in "$NAME" joint49w; do
  [ -f "/content/$want/mapper_last.pt" ] && continue
  [ "$want" = joint49w ] && [ -f /content/joint49w/mapper_best.pt ] && continue
  python3 - "$want" <<'PYEOF' || true
import os, sys, shutil, pathlib
from huggingface_hub import snapshot_download
name = sys.argv[1]      # qua argv: heredoc trich dan khong no bien shell
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

INIT_M=/content/joint49w/mapper_best.pt
INIT_L=/content/joint49w/lora_best
INIT_T=""
if [ -f "/content/$NAME/mapper_last.pt" ]; then
  INIT_M="/content/$NAME/mapper_last.pt"
  [ -d "/content/$NAME/lora_last" ]  && INIT_L="/content/$NAME/lora_last"
  # LoRA 9B chi ton tai tu dot nay tro di -> chi noi lai khi that su co
  [ -d "/content/$NAME/lorat_last" ] && INIT_T="/content/$NAME/lorat_last"
  echo "NOI LAI tu $INIT_M"
else
  echo "BAT DAU MOI tu joint49w (mapper_best + lora_best)"
fi
[ -f "$INIT_M" ] || { echo "KHONG THAY $INIT_M"; exit 1; }

# Nham cac module TIEU THU cache: q_proj/o_proj doc K/V da ghep; in_proj_qkvz/
# out_proj cua linear_attn la noi trang thai GDN duoc dung.
MODS="${LORAT_MODS:-q_proj,o_proj,in_proj_qkvz,out_proj}"
BATCH="${BATCH:-2}"

# GHEP CUNG CAU HINH joint49w — de chi con DUNG HAI bien doi.
# Hoc phi (dot dau, 2026-08-31): phong voi max-ctx 16384 + accum 1 trong khi
# joint49w train o 4096 + accum 4, tuc doi BON thu cung luc. Val ra needle
# 0/15 (49w: 15/15) ma KHONG quy duoc cho ai: needle phu thuoc truc tiep vao
# do dai ngu canh nen ctx gap 4 lan da doi chinh cac mau val.
# accum 2 x batch 2 = 4 mau/lan cap nhat = DUNG bang 49w (1 x 4).
# LUU Y: khong dat dong '#' nao BEN TRONG chuoi COMMON — shell khong coi do la
# chu thich, no thanh tham so truyen thang cho e9_joint.py.
COMMON="--tgt-model Qwen/Qwen3.5-9B
  --data-file /content/train_items.json
  --pseudo-gold /content/pseudo_gold.json
  --max-ctx ${MAXCTX:-4096} --tbptt 128 --gold-cap 256 --gold-envelope 16384:256
  --drop-kinds ${DROP:-gsm8k,suite_math}
  --no-offload
  --batch $BATCH
  --lora-t $MODS --lora-t-r ${LORAT_R:-16}
  --init-mapper $INIT_M --init-lora $INIT_L"
[ -n "$INIT_T" ] && COMMON="$COMMON --init-lora-t $INIT_T"

if [ "${GO:-0}" != "1" ]; then
  step "SANITY 40 buoc — do toc do that + verify-meta o CA (T=512,B=1) va (512,B=2)"
  # Chieu batch trong meta la mot trong hai cho sai AM THAM (cai kia la mat na
  # CE). Lech chieu batch = cache vo nghia ma khong bao loi nao -> phai de
  # verify-meta bat truoc khi phong 1000 buoc.
  python3 -u e9_joint.py $COMMON \
    --accum ${ACCUM:-2} --steps 40 --sanity 40 --val-every 100000 \
    --verify-meta 512 \
    --out "/content/${NAME}_sanity" \
    --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix "${NAME}_sanity"
  echo "SANITY XONG — doc s/buoc (mong ~1,5-1,7 so voi 3,00 cua batch 1)."
  echo "Chay that: GO=1 bash run_joint49x.sh"
  echo "RUN_49X_SANITY_EXIT"
  exit 0
fi

step "TRAIN $NAME (${STEPS:-1000} buoc, dung theo val)"
# val-every 200: checkpoint CHI duoc luu + upload HF o moc val, ma runtime bi
# thu hoi ~1,5 gio/lan. Moc thua thi moi lan mat runtime chi mat mot mau.
python3 -u e9_joint.py $COMMON \
  --steps ${STEPS:-1000} --val-every ${VALEVERY:-200} --val-n ${VALN:-150} \
  --ce-floor 0.05 --patience ${PAT:-3} --accum ${ACCUM:-2} \
  --verify-meta 512 \
  --out "/content/$NAME" \
  --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix "$NAME"

step "XONG"
echo "RUN_49X_EXIT"
