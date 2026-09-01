#!/usr/bin/env bash
# joint49aa — do NIEM PHONG (1.650 mau, 7 bo) + doi chung ctx-BO + gop (agg),
# giong het co che da dung cho joint49y (run_joint49y_seal.sh).
#
#   bash run_joint49aa_seal.sh
#
# Chot chan (nhu joint49y): suite_swe/musr sap ve gan 0 khi bo ngu canh =
# khong an gian. DBATCH=8 la muc da do 6,7x va cong kiem 25/25 khop.
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
CK="/content/joint49aa"

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

echo "=== [1/2] mapped DAY DU (8 bo KE CA gsm8k, niem phong) ==="
# joint49aa la lan DAU TIEN gsm8k duoc dua vao vong niem phong (truoc day
# GIAI DOAN 1 chi do cac ho da ghi nhan mapper chay, gsm8k de sang giai
# doan 2 — nay la giai doan 2). 100 mau gsm8k co san trong eval_big_items.json.
# KHONG kiem file ton tai o day (hoc phi cu voi joint49y): file ghi TUNG
# PHAN nen "if -f" luon dung -> bo qua nham phan con lai. eval_big.py TU
# resume theo tung mau, goi lai la an toan va DUNG.
python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA \
  --decode-batch "${DBATCH:-8}" --verify-batch "${VBATCH:-25}" \
  --benches "${BENCHES:-bbh,bfcl,needle,musr,suite_mid,suite_rag,suite_swe,gsm8k}" \
  --hf-prefix "evalbig_$(basename $CK)" || exit 1

echo "=== [2/2] mapped ctx-BO (suite_swe+musr — CHOT CHAN; gsm8k BO QUA vi"
echo "context CHINH LA cau hoi nen no_ctx vo nghia, nhu da ghi trong e9_joint.py)"
python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA \
  --no-ctx drop \
  --decode-batch "${DBATCH:-8}" --verify-batch 0 \
  --benches suite_swe,musr \
  --hf-prefix "evalbig_$(basename $CK)_drop" || exit 1

echo "=== GOP (agg): self vs mapped, giu duoc bao nhieu % ==="
EVALBIG_PREFIX="evalbig_$(basename $CK)" python3 -u eval_big.py agg \
  | tee /content/logs/evalbig_49aa_agg.txt

python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/evalbig_49aa_agg.txt", "/content/logs/49aa_seal.log"]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f,
                            repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + os.path.basename(f))
            print("HF-UP", f)
        except Exception as e:
            print("HF-UP FAIL", f, type(e).__name__, str(e)[:100])
PYEOF
echo "RUN_49AA_SEAL_EXIT"
