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
#   PROBE=1 bash run_gsm_struct_rl.sh      # do BSZ/K lon nhat khong OOM
#   GO=1 bash run_gsm_struct_rl.sh         # chay that
#
# GOP LO (2026-09-05): moi buoc xu ly BSZ mau x K nhanh trong MOT vong decode.
# Do duoc (probe_decode_speed.py): thoi gian moi buoc decode gan nhu KHONG DOI
# tu 2 den 16 hang (95,7 -> 100,1 ms) trong khi thong luong x7,7 -- decode o
# batch nho bi chan boi bang thong doc trong so, khong phai phep tinh. Rang
# buoc that la VRAM (dinh 20,59/22,5 GiB o bsz=1,k=2), nen PROBE truoc.
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CK="${CK:-sft_struct_v3}"          # warm-start tu Buoc 2
OUT="${OUT:-gsm_struct_rl_v1}"
STEPS="${STEPS:-1000}"
K="${K:-3}"                         # K=4 tung OOM o vong truoc (khi bsz=1)
BSZ="${BSZ:-4}"                     # so MAU/buoc (user chot 2026-09-05: 4x3)
TF_CHUNK="${TF_CHUNK:-2}"           # so hang/mieng o pha 2 (cho OOM lop GDN)
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

run_rl () {   # $1=bsz $2=k $3=args them $4=log
  python3 -u eba_grpo.py --task gsm8k_struct \
    --init-mapper "/content/$CK/mapper_best.pt" \
    --init-lora "/content/$CK/lora_best" \
    --init-lora-t "/content/$CK/lorat_best" \
    --gsm-data /content/train_items.json \
    --struct-gold /content/struct_gold_gsm.json \
    --bsz "$1" --k "$2" --tf-chunk "$TF_CHUNK" \
    --gen-len "$GEN_LEN" --gold-cap 320 \
    --gsm-limit 0 --anchor-w 0 \
    --val-every 100 --val-n 32 --snapshot-every 200 \
    --out "/content/$OUT" --hf-prefix "$OUT" \
    $3 2>&1 | tee "$4"
  return ${PIPESTATUS[0]}
}

if [ "${PROBE:-0}" = "1" ]; then
  # Thu tu do: cau hinh user chot truoc (4x3=12 hang), tut dan khi OOM.
  # Chay 3 buoc that (khong phai chi nap model) vi dinh VRAM roi vao pha
  # sampling + backward, khong phai luc nap.
  for cfg in "4 3" "4 2" "2 3" "2 2"; do
    set -- $cfg
    echo "=== PROBE bsz=$1 k=$2 (= $(($1 * $2)) hang decode) ==="
    if run_rl "$1" "$2" "--sanity 3" "/content/logs/probe_bsz_$1x$2.log" \
       && grep -q EBA_GRPO_SANITY_EXIT "/content/logs/probe_bsz_$1x$2.log"; then
      echo "PROBE_CHOT bsz=$1 k=$2"
      grep -E 'SANITY xong|lo train' "/content/logs/probe_bsz_$1x$2.log"
      exit 0
    fi
    echo "  -> HONG (OOM hoac loi), tut xuong cau hinh sau"
    sleep 5
  done
  echo "PROBE_KHONG_CAU_HINH_NAO_CHAY"; exit 1
fi

if [ "${GO:-0}" = "1" ]; then
  ARGS="--steps $STEPS --sanity 0"
else
  # sanity dai hon 5 buoc khi do bo nho: dinh VRAM phu thuoc DO DAI PROMPT
  # (41-201 token) va do dai sinh ra, 5 buoc de trung toan mau ngan -> bao
  # "vua" roi OOM giua chung khi chay that.
  ARGS="--sanity ${SANITY:-5}"
fi
run_rl "$BSZ" "$K" "$ARGS" "/content/logs/${OUT}.log"
STATUS=$?
[ "$STATUS" = 0 ] && echo "RUN_GSM_STRUCT_RL_EXIT" || echo "RUN_GSM_STRUCT_RL_FAIL status=$STATUS"
