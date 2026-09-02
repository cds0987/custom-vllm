#!/usr/bin/env bash
# ORACLE ABLATION — cau hoi phan xu: GDN co THAT SU la nut that gsm8k khong,
# hay attn mapped moi la van de? Hoan doi truc tiep tung nua cache bang cache
# 9B THAT (chinh 9B tu prefill), khong train gi — gan mien phi ve GPU.
#
#   bash run_oracle_ablation.sh            # 30 mau mac dinh
#   ORACLE_N=20 bash run_oracle_ablation.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs

for pkg in peft bitsandbytes; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done

python3 -u oracle_ablation.py \
  --mapper /content/joint49bb/mapper_best.pt \
  --lora /content/joint49bb/lora_best \
  --lora-t /content/joint49bb/lorat_best \
  --n "${ORACLE_N:-30}" \
  --out /content/logs/oracle_ablation.json || exit 1

python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/oracle_ablation.json"]:
    p = pathlib.Path(f)
    if p.exists():
        try:
            api.upload_file(path_or_fileobj=f, repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + p.name)
            print("HF-UP", p.name)
        except Exception as e:
            print("HF-UP FAIL", p.name, type(e).__name__, str(e)[:100])
PYEOF
echo "RUN_ORACLE_EXIT"
