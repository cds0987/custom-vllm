#!/usr/bin/env bash
# Pha 6: MAPPED tren 1875 mau niem phong — cot con thieu ben canh cot self.
#
#   bash run_bigmapped.sh [/content/joint49s] [slice]
#
# ~4 gio: transformers TUAN TU (vLLM khong cho tiem cache do mapper dung).
# NOI LAI duoc: eval_big ghi + upload HF moi 25 mau, va bo qua mau da cham —
# runtime da bi thu hoi 5 lan trong ~10 tieng nen chay lien 4 gio chac chan dut.
set -u
cd "$(dirname "$0")"
CK="${1:-/content/joint49s}"
SL="${2:-}"
mkdir -p /content/logs

# moi truong (idempotent, khong can vLLM cho pha nay)
for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done

# bo 1875 mau: keo tu HF neu recycle da xoa
python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
d = pathlib.Path("/content/eval_big_items.json")
if not d.exists():
    p = hf_hub_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                        "evalbig/eval_big_items.json",
                        token=os.environ.get("HF_TOKEN"))
    d.write_bytes(pathlib.Path(p).read_bytes())
    print("keo bo mau tu HF:", d.stat().st_size, "byte")
PYEOF

# checkpoint: local hoac keo tu HF
if [ ! -f "$CK/mapper_best.pt" ]; then
  python3 - "$CK" <<'PYEOF'
import os, sys, shutil, pathlib
from huggingface_hub import snapshot_download
name = pathlib.Path(sys.argv[1]).name
p = snapshot_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                      allow_patterns=[f"{name}/*"], local_dir="/content/_hfck",
                      token=os.environ.get("HF_TOKEN"))
shutil.copytree(pathlib.Path(p) / name, sys.argv[1], dirs_exist_ok=True)
print("keo checkpoint:", sorted(x.name for x in pathlib.Path(sys.argv[1]).iterdir()))
PYEOF
fi
[ -f "$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK/mapper_best.pt"; exit 1; }
LORA=""
[ -d "$CK/lora_best" ] && LORA="--lora $CK/lora_best"

# GIAI DOAN 1 (user chot): chi do cac ho DA GHI NHAN mapper chay. gsm8k va
# suite_math de sang giai doan 2 — gsm8k ton 320 token/mau (dat nhat ca
# luot, ~45 phut cho 100 mau) ma mapper moi dat 9%; suite_math tran chi
# 1,6% nen khong phan biet duoc mapper tot hay xau. Bo mau tren dia GIU NGUYEN.
echo "=== mapped tren $CK ${SL:+(slice $SL)} ==="
python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA ${SL:+--slice $SL} \
  --decode-batch "${DBATCH:-1}" --verify-batch "${VBATCH:-0}" \
  --benches "${BENCHES:-bbh,bfcl,needle,musr,suite_mid,suite_rag,suite_swe}" \
  --hf-prefix "evalbig_$(basename $CK)" || exit 1

EVALBIG_PREFIX="evalbig_$(basename $CK)" python3 -u eval_big.py agg \
  | tee /content/logs/evalbig_agg.txt

python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/evalbig_agg.txt", "/content/logs/bigmapped.log"]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f,
                            repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + os.path.basename(f))
            print("HF-UP", f)
        except Exception as e:
            print("HF-UP FAIL", f, type(e).__name__, str(e)[:100])
PYEOF
echo "RUN_BIGMAPPED_EXIT"
