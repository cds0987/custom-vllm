#!/usr/bin/env bash
# BUOC 3 -- RL/GRPO tren dinh dang CO CAU TRUC, 4 reward phan ra.
#
# Vao duoc buoc nay khi Buoc 2 (sft_struct.py) da dat cong >=90% parse --
# format hong thi moi reward = 0/-1 dong loat -> nhom K mau cung diem ->
# advantage = 0 -> GRPO khong co gradient (dung canh bao Unsloth).
#
# KHAC vong RL truoc (gsm_grpo_v1, reward chi 0/1 dap so cuoi):
#   - 4 reward doc lap: R_ent (PHAT NANG -8,0 khi gan sai so), R_rel, R_step
#     (LCS theo thu tu), R_ans (chia bac, hong format -1,0)
#   - BO HANN anchor-CE (--anchor-w 0): day la cho gay hoc vet
#   - cham theo NOI DUNG DA PARSE, khong theo chuoi chu
#
#   bash run_gsm_struct_rl.sh              # sanity 5 buoc
#   GO=1 bash run_gsm_struct_rl.sh         # chay that
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CK="${CK:-sft_struct_v3}"          # warm-start tu Buoc 2
OUT="${OUT:-gsm_struct_rl_v1}"
STEPS="${STEPS:-1000}"
K="${K:-3}"                         # K=4 tung OOM o vong truoc
GEN_LEN="${GEN_LEN:-320}"

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done

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
        print("KEO VE", dest)
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
[ -f "/content/$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK (chay Buoc 2 truoc), DUNG"; exit 1; }

if [ "${GO:-0}" = "1" ]; then
  ARGS="--steps $STEPS --sanity 0"
else
  ARGS="--sanity 5"
fi
python3 -u eba_grpo.py --task gsm8k_struct \
  --init-mapper "/content/$CK/mapper_best.pt" \
  --init-lora "/content/$CK/lora_best" \
  --init-lora-t "/content/$CK/lorat_best" \
  --gsm-data /content/train_items.json \
  --struct-gold /content/struct_gold_gsm.json \
  --k "$K" --gen-len "$GEN_LEN" --gold-cap 320 \
  --anchor-w 0 \
  --val-every 100 --val-n 32 --snapshot-every 200 \
  --out "/content/$OUT" --hf-prefix "$OUT" \
  $ARGS 2>&1 | tee "/content/logs/${OUT}.log"
STATUS=${PIPESTATUS[0]}
[ "$STATUS" = 0 ] && echo "RUN_GSM_STRUCT_RL_EXIT" || echo "RUN_GSM_STRUCT_RL_FAIL status=$STATUS"
