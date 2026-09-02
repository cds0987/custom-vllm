#!/usr/bin/env bash
# joint49bb — do NIEM PHONG CHI 2 bo (suite_swe day du 123 mau + gsm8k 100
# mau, dung pham vi da thu hep — quy tac 14) + doi chung ctx-BO cho suite_swe
# + gop (agg). Co che giong het joint49z/joint49aa.
#
#   bash run_joint49bb_seal.sh
#
# gsm8k KHONG chay ctx-BO: context CHINH LA cau hoi nen bo ngu canh vo nghia
# (da ghi trong e9_joint.py, dung lai quy uoc cua joint49aa_seal).
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
CK="/content/joint49bb"

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

echo "=== [1/2] mapped DAY DU (suite_swe + gsm8k, niem phong) ==="
python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA \
  --decode-batch "${DBATCH:-8}" --verify-batch "${VBATCH:-25}" \
  --benches "${BENCHES:-suite_swe,gsm8k}" \
  --hf-prefix "evalbig_$(basename $CK)" || exit 1

echo "=== [2/2] mapped ctx-BO (chi suite_swe — gsm8k bo qua, xem ghi chu tren) ==="
python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA \
  --no-ctx drop \
  --decode-batch "${DBATCH:-8}" --verify-batch 0 \
  --benches suite_swe \
  --hf-prefix "evalbig_$(basename $CK)_drop" || exit 1

echo "=== GOP (agg): self vs mapped, giu duoc bao nhieu % ==="
EVALBIG_PREFIX="evalbig_$(basename $CK)" python3 -u eval_big.py agg \
  | tee /content/logs/evalbig_49bb_agg.txt

python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/evalbig_49bb_agg.txt", "/content/logs/49bb_seal.log"]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f,
                            repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + os.path.basename(f))
            print("HF-UP", f)
        except Exception as e:
            print("HF-UP FAIL", f, type(e).__name__, str(e)[:100])
PYEOF
echo "RUN_49BB_SEAL_EXIT"
