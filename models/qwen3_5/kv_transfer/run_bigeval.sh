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

step "0/5 moi truong"
if python3 -c "import vllm" 2>/dev/null; then
  echo "vllm da co: $(python3 -c 'import vllm;print(vllm.__version__)')"
else
  bash "$ROOT/loading/setup_env.sh" 2>&1 | tail -6
fi
# CAI TUNG GOI MOT: `pip -q install A B` ma B hong build thi A CUNG khong duoc
# cai, va -q nuot loi (hoc phi da ghi trong docs/03-bug-va-cach-sua.md).
for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done
# setup_env.sh in ro "source /tmp/vllm_env.sh before running vllm": no dat
# LD_LIBRARY_PATH toi libcudart cu13. Thieu -> vLLM chet luc nap. `|| true`
# vi da co lan `run.sh serve` chet CAM khi file nay vang (set -e nuot loi).
[ -f /tmp/vllm_env.sh ] && . /tmp/vllm_env.sh || true
python3 -c "import vllm,torch,peft,bitsandbytes;print('vllm',vllm.__version__,'torch',torch.__version__)"

step "1/5 tap niem phong CU (chi de kiem ro ri)"
if [ -f /content/ext_bench_items.json ]; then echo "da co, bo qua"; else
  # --n-each 200, KHONG phai mac dinh 500: tap test da BAO CAO la 200/bo
  # (bbh 98/182 = 7 hang x 26 tac vu, musr 115/198 = 66 x 3, gsm8k 160/200)
  # va TEST_BBH_PER/TEST_MUSR_PER trong gen_data ghim cung con 200 do. Dung
  # 500 la tu bia ra mot tap niem phong TO HON tap that -> bao ro ri GIA.
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
  python3 -u eval_big.py self --tgt-model Qwen/Qwen3.5-9B --max-len 6144 || exit 1
fi

step "5/5 TRAIN joint49v (2500 buoc tu joint49s)"
python3 -u e9_joint.py \
  --tgt-model Qwen/Qwen3.5-9B \
  --data-file /content/train_items.json \
  --pseudo-gold /content/pseudo_gold.json \
  --max-ctx 4096 --tbptt 128 --gold-cap 256 --gold-envelope 16384:256 \
  --steps 8000 --val-every 500 --val-n 150 --ce-floor 0.05 --patience 3 \
  --no-offload --verify-meta 512 \
  --init-mapper /content/joint49s/mapper_best.pt \
  --init-lora   /content/joint49s/lora_best \
  --out /content/joint49v \
  --hf-repo gunnybd01/qwen35-kv-mapper-4b-27b --hf-prefix joint49v

step "XONG TAT CA"
echo "RUN_BIGEVAL_EXIT"
