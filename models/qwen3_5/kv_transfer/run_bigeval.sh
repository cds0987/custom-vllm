#!/usr/bin/env bash
# Chien dich "chac chan phan dang vuot tran la that" — user duyet 2026-08-29.
#
# NAM PHA NOI TIEP tren 1 L4, chay nen (quy tac 10). Moi pha idempotent: da co
# ket qua thi bo qua -> runtime bi thu hoi giua chung chi can chay lai cell.
#
#   1. ext_bench gen   — dung lai tap NIEM PHONG CU, chi de KIEM RO RI
#   2. eval_big  gen   — 2000 mau moi, doi chieu chuoi prompt voi ca train_items
#                        LAN tap e6v3.build_data() sinh luc chay
#   3. tai joint49s    — checkpoint score 67 (an toan tren HF)
#   4. eval_big  self  — cot tran, bang vLLM (dung dung token ket thuc; vong
#                        greedy tay tung cho 32% thay vi 92% tren cung 40 mau)
#   5. e9_joint  train — joint49v tu joint49s, DUNG THEO VAL (patience 3)
#                        chu KHONG theo so buoc: 3/4 luot truoc bi cat ngang
#                        vi runtime bi thu hoi hoac vi het so buoc — KHONG
#                        luot nao dung vi HOI TU (49s van dang len o buoc cuoi).
#                        Do mapper chua hoi tu thi so do chi la CAN DUOI.
#
# Pha 6 (eval_big mapped) chay rieng sau khi chon duoc checkpoint tot nhat.
set -u
cd "$(dirname "$0")"
KV="$(pwd)"
ROOT="$KV/../../.."
mkdir -p /content/logs

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

ensure_vllm() {
  # CHI cai vLLM khi that su can (pha 4). Train KHONG dung vLLM, ma cai mat
  # ~12 phut — voi nhip Colab recycle ~1,5 gio thi do la 13% thoi gian do
  # khong cho gi. Kiem theo nhu cau, khong cai mu quang o dau chuoi.
  if python3 -c "import vllm" 2>/dev/null; then
    echo "vllm da co: $(python3 -c 'import vllm;print(vllm.__version__)')"
  else
    bash "$ROOT/loading/setup_env.sh" 2>&1 | tail -6
  fi
  [ -f /tmp/vllm_env.sh ] && . /tmp/vllm_env.sh || true
}

step "0/5 moi truong (chua cai vLLM — chi cai neu pha 4 can)"
# CAI TUNG GOI MOT: `pip -q install A B` ma B hong build thi A CUNG khong duoc
# cai, va -q nuot loi (hoc phi da ghi trong docs/03-bug-va-cach-sua.md).
for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done
# setup_env.sh in ro "source /tmp/vllm_env.sh before running vllm": no dat
# LD_LIBRARY_PATH toi libcudart cu13. Thieu -> vLLM chet luc nap. `|| true`
# vi da co lan `run.sh serve` chet CAM khi file nay vang (set -e nuot loi).
[ -f /tmp/vllm_env.sh ] && . /tmp/vllm_env.sh || true
python3 -c "import torch,peft,bitsandbytes;print('torch',torch.__version__)"

step "0b/5 khoi phuc tu HF (runtime da bi thu hoi 5 lan)"
# Recycle xoa sach /content. Bo 1875 mau (~15 phut dung) va cot self (~12 phut
# vLLM) da duoc upload theo quy tac 6d — keo ve re hon nhieu lan lam lai.
python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
pathlib.Path("/content/logs").mkdir(parents=True, exist_ok=True)
# (duong dan tren HF, dich duoi runtime). train_items + pseudo_gold la DAU VAO
# cua pha train — recycle xoa /content nen phai keo ve, neu khong pha 5 chet.
for name, dest in [("evalbig/eval_big_items.json", "/content/eval_big_items.json"),
                   ("evalbig/evalbig_self.json", "/content/logs/evalbig_self.json"),
                   ("joint_v1/train_items.json", "/content/train_items.json"),
                   ("joint49/pseudo_gold.json", "/content/pseudo_gold.json")]:
    if pathlib.Path(dest).exists():
        print("da co", dest)
        continue
    try:
        p = hf_hub_download(REPO, name,
                            token=os.environ.get("HF_TOKEN"))
        pathlib.Path(dest).write_bytes(pathlib.Path(p).read_bytes())
        print("KEO VE", dest, pathlib.Path(dest).stat().st_size, "byte")
    except Exception as e:
        print("khong keo duoc", name, type(e).__name__, str(e)[:80])
PYEOF

step "1/5 tap niem phong CU (chi de kiem ro ri)"
if [ -f /content/eval_big_items.json ]; then
  echo "bo 1875 mau da co -> khong can tap niem phong cu (chi dung de kiem ro ri)"
elif [ -f /content/ext_bench_items.json ]; then echo "da co, bo qua"; else
  # --n-each 200, KHONG phai mac dinh 500: tap test da BAO CAO la 200/bo
  # (bbh 98/182 = 7 hang x 26 tac vu, musr 115/198 = 66 x 3, gsm8k 160/200)
  # va TEST_BBH_PER/TEST_MUSR_PER trong gen_data ghim cung con 200 do. Dung
  # 500 la tu bia ra mot tap niem phong TO HON tap that -> bao ro ri GIA.
  ensure_vllm
  python3 -u ext_bench.py gen --bench bbh,gsm8k,musr --n-each 200 2>&1 | tail -8
fi

step "2/5 dung 2000 mau niem phong MOI"
if [ -f /content/eval_big_items.json ]; then echo "da co, bo qua"; else
  python3 -u eval_big.py gen --n-bbh 800 --n-gsm8k 100 --n-musr 60 \
      --n-bfcl 600 --n-needle 240 --n-suite 500 || exit 1
fi

step "3/5 tai joint49s (mapper score 67 + lora)"
if [ -d /content/joint49s/lora_best ]; then echo "da co, bo qua"; else
  python3 - <<'EOF'
import os, shutil, pathlib
from huggingface_hub import snapshot_download
p = snapshot_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                      allow_patterns=["joint49s/*"], local_dir="/content/_hf",
                      token=os.environ.get("HF_TOKEN"))
shutil.copytree(pathlib.Path(p) / "joint49s", "/content/joint49s",
                dirs_exist_ok=True)
print("joint49s:", sorted(x.name for x in pathlib.Path("/content/joint49s").iterdir()))
EOF
fi

step "4/5 cot SELF bang vLLM (tran cua 2000 mau)"
if [ -f /content/logs/evalbig_self.json ]; then echo "da co, bo qua"; else
  ensure_vllm
  python3 -u eval_big.py self --tgt-model Qwen/Qwen3.5-9B --max-len 6144 || exit 1
fi

step "4b/5 PROBE gom lo (user: batch=1 la ko duoc)"
# Chay TRUOC train vi ket qua cua no quyet dinh cau hinh train (batch that hay
# chi gradient accumulation). Train xong roi moi do thi phai train lai.
if [ -f /content/logs/probe_batch.log ]; then echo "da co, bo qua"; else
  python3 -u probe_batch.py --src-model Qwen/Qwen3.5-4B \
      2>&1 | tee /content/logs/probe_batch.log | tail -40
fi

step "5/5 chuan bi: noi lai joint49v neu da co tien do"
# CONG CHO: khong tu dong train. Cau hinh train (batch that / --accum bao
# nhieu) PHU THUOC ket qua probe 4b, va do la quyet dinh cua user. Chay lien
# tay se train sai cau hinh roi phai train lai — dat hon nhieu lan viec cho.
if [ ! -f /content/train_cfg.sh ]; then
  echo "CHUA CO /content/train_cfg.sh -> DUNG o day, cho cau hinh train."
  echo "RUN_BIGEVAL_CHO_CAU_HINH"
  exit 0
fi
. /content/train_cfg.sh
echo "cau hinh train: accum=${ACCUM:-1} steps=${STEPS:-8000} patience=${PAT:-3}"
# Runtime bi thu hoi 4 lan trong ngay. Khong noi lai thi moi lan recycle la
# mat sach gio train va quay ve joint49s. Uu tien: ban local -> ban tren HF ->
# joint49s. Phai doc TU HF vi recycle xoa /content nhung HF thi con.
INIT_M=/content/joint49s/mapper_best.pt
INIT_L=/content/joint49s/lora_best
if [ ! -f /content/joint49v/mapper_last.pt ]; then
  python3 - <<'PYEOF' || true
import os, shutil, pathlib
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                          allow_patterns=["joint49v/*"],
                          local_dir="/content/_hf49v",
                          token=os.environ.get("HF_TOKEN"))
    src = pathlib.Path(p) / "joint49v"
    if (src / "mapper_last.pt").exists():
        shutil.copytree(src, "/content/joint49v", dirs_exist_ok=True)
        print("NOI LAI tu HF joint49v/")
    else:
        print("HF chua co joint49v/mapper_last.pt -> bat dau tu joint49s")
except Exception as e:
    print("khong lay duoc joint49v tu HF:", type(e).__name__, str(e)[:80])
PYEOF
fi
if [ -f /content/joint49v/mapper_last.pt ]; then
  INIT_M=/content/joint49v/mapper_last.pt
  [ -d /content/joint49v/lora_last ] && INIT_L=/content/joint49v/lora_last
  echo "NOI LAI tu: $INIT_M"
else
  echo "BAT DAU MOI tu joint49s"
fi

step "5/5 TRAIN joint49v (dung theo val, patience 3)"
# val-every 250 chu khong 500: Colab dang recycle moi ~1,5 gio, ma checkpoint
# chi duoc luu (va upload HF) o MOC VAL. Moc cach nhau 25 phut nghia la moi
# lan mat runtime co the mat gan het cong. 250 buoc ~ 12 phut, va val gio re
# hon nhieu vi da bo gsm8k (320 token/mau).
python3 -u e9_joint.py \
  --tgt-model Qwen/Qwen3.5-9B \
  --data-file /content/train_items.json \
  --pseudo-gold /content/pseudo_gold.json \
  --max-ctx 4096 --tbptt 128 --gold-cap 256 --gold-envelope 16384:256 \
  --steps ${STEPS:-8000} --val-every ${VALEVERY:-250} --val-n ${VALN:-150} \n  --ce-floor 0.05 \
  --patience ${PAT:-3} --accum ${ACCUM:-1} \n  --drop-kinds "${DROP:-gsm8k,suite_math}" \
  --no-offload --verify-meta 512 \
  --init-mapper "$INIT_M" \
  --init-lora   "$INIT_L" \
  --out /content/joint49v \
  --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix joint49v

step "XONG TAT CA"
echo "RUN_BIGEVAL_EXIT"
