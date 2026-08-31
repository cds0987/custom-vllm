#!/usr/bin/env bash
# joint49y — do NIEM PHONG (1.875 mau) + doi chung ctx-BO, theo CHOT CHAN da
# dat truoc khi train: suite_swe >60% VA ctx-BO van sap = thanh cong; diem
# len ma ctx-BO KHONG sap = an gian.
#
#   bash run_joint49y_seal.sh
#
# HAI LUOT, moi luot idempotent (bo qua neu da co ket qua):
#   1. mapped DAY DU tren TOAN BO 7 bo (bfcl/bbh/needle/musr/suite_*) — cot
#      chinh de bao cao.
#   2. mapped CHI suite_swe+musr, --no-ctx drop — doi chung. Gioi han 2 bo
#      nay vi day la noi 4B/9B gap nhau ve QUAN HE (nghi pham chinh), va
#      chay het 7 bo x 2 luot se ton gap doi thoi gian ma khong them thong tin.
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
CK="/content/joint49y"

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done

if [ ! -f "$CK/mapper_best.pt" ]; then
  echo "KHONG THAY $CK/mapper_best.pt -> chua train xong hoac chua tai ve"
  exit 1
fi
LORA=""
[ -d "$CK/lora_best" ]  && LORA="--lora $CK/lora_best"
[ -d "$CK/lorat_best" ] && LORA="$LORA --lora-t $CK/lorat_best"
echo "adapter: ${LORA:-khong co}"

echo "=== [1/2] mapped DAY DU (7 bo, niem phong) ==="
# KHONG kiem file ton tai o day: file duoc ghi TUNG PHAN (moi 25 mau),
# nen sau khi ket qua dau tien duoc luu la "if -f" luon dung -> bo qua
# nham phan con lai (hoc phi dot nay). eval_big.py TU resume theo tung
# mau (doc file cu, bo qua id da co) nen goi lai la an toan va DUNG.
python3 -u eval_big.py mapped \
--tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
--mapper "$CK/mapper_best.pt" $LORA \
--decode-batch "${DBATCH:-1}" --verify-batch "${VBATCH:-0}" \
--benches "${BENCHES:-bbh,bfcl,needle,musr,suite_mid,suite_rag,suite_swe}" \
--hf-prefix "evalbig_$(basename $CK)" || exit 1

echo "=== [2/2] mapped ctx-BO (chi suite_swe+musr — CHOT CHAN) ==="
python3 -u eval_big.py mapped \
--tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
--mapper "$CK/mapper_best.pt" $LORA \
--no-ctx drop \
--decode-batch "${DBATCH:-1}" --verify-batch "${VBATCH:-0}" \
--benches suite_swe,musr \
--hf-prefix "evalbig_$(basename $CK)_drop" || exit 1

EVALBIG_PREFIX="evalbig_$(basename $CK)" python3 -u eval_big.py agg \
  | tee /content/logs/evalbig_49y_agg.txt

python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/evalbig_49y_agg.txt", "/content/logs/49y_seal.log"]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f,
                            repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + os.path.basename(f))
            print("HF-UP", f)
        except Exception as e:
            print("HF-UP FAIL", f, type(e).__name__, str(e)[:100])
PYEOF
echo "RUN_49Y_SEAL_EXIT"
