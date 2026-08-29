#!/usr/bin/env bash
# Pha 6: chay MAPPED tren 2000 mau niem phong. Tach rieng vi phai doi chon
# duoc checkpoint tot nhat (joint49v hay joint49s) tu ket qua val.
#
#   bash run_bigmapped.sh /content/joint49v          # thu muc chua mapper_best.pt + lora_best
#
# ~4 gio: transformers TUAN TU (vLLM khong cho tiem cache do mapper dung).
# Resume duoc: pha A bo qua file spill da co; --slice cat khuc neu can.
set -u
cd "$(dirname "$0")"
CK="${1:-/content/joint49v}"
SL="${2:-}"
[ -f "$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK/mapper_best.pt"; exit 1; }
LORA=""
[ -d "$CK/lora_best" ] && LORA="--lora $CK/lora_best"
echo "=== mapped tren $CK ${SL:+(slice $SL)} ==="
python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA ${SL:+--slice $SL} || exit 1
python3 -u eval_big.py agg | tee /content/logs/evalbig_agg.txt

# quy tac 6d: len HF NGAY, cung phien
python3 - "$CK" <<'EOF'
import os, sys, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
pref = "evalbig_" + pathlib.Path(sys.argv[1]).name
for f in ["/content/eval_big_items.json", "/content/logs/evalbig_self.json",
          "/content/logs/evalbig_mapped.json", "/content/logs/evalbig_agg.txt",
          "/content/logs/bigmapped.log"]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f, repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo=f"{pref}/{pathlib.Path(f).name}")
            print("HF-UP", f)
        except Exception as e:
            print("HF-UP FAIL", f, type(e).__name__, str(e)[:100])
EOF
echo "RUN_BIGMAPPED_EXIT"
