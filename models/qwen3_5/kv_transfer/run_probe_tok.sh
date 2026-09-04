#!/usr/bin/env bash
# PHEP KIEM 40 PHUT (ra soat 2026-09-04): kenh danh tinh token kieu EAGLE co
# mang thong tin that khong? Chay HAI nhanh mot-bien, cung seed/du lieu/buoc:
#   A) tok_rank=0  = mapper hien tai (doi chung)
#   B) tok_rank=64 = them kenh embedding token
# Doc: CE tren gold + do chinh xac gsm8k THAT (chuc nang) truoc/sau train.
#
#   bash run_probe_tok.sh
#   STEPS=300 RANK=64 bash run_probe_tok.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
STEPS="${STEPS:-300}"
RANK="${RANK:-64}"
CK="${CK:-gsm_grpo_v1c}"

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done

python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import hf_hub_download
REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
for name, dest in [("joint49_cot/train_items_gsm.json", "/content/train_items_gsm.json"),
                   ("joint49_cot/pseudo_gold_gsm2.json", "/content/pseudo_gold_gsm2.json")]:
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
                          allow_patterns=[f"{name}/*"],
                          local_dir=f"/content/_hf_{name}",
                          token=os.environ.get("HF_TOKEN"))
    src = pathlib.Path(p) / name
    if any(src.glob("mapper_*.pt")):
        shutil.copytree(src, f"/content/{name}", dirs_exist_ok=True)
        print("KEO VE", name)
except Exception as e:
    print("khong lay duoc", name, type(e).__name__, str(e)[:80])
PYEOF
[ -f "/content/$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK, DUNG"; exit 1; }

echo "===== [A] DOI CHUNG: tok_rank=0 (mapper hien tai) ====="
python3 -u probe_tok_channel.py --tok-rank 0 --tag base --steps "$STEPS" \
  --init-dir "/content/$CK" || exit 1

echo "===== [B] BIEN: tok_rank=$RANK (kenh danh tinh token) ====="
python3 -u probe_tok_channel.py --tok-rank "$RANK" --tag tok --steps "$STEPS" \
  --init-dir "/content/$CK" || exit 1

echo "===== SO SANH ====="
python3 - <<'PYEOF'
import json, pathlib
d = json.loads(pathlib.Path("/content/logs/probe_tok.json").read_text())
print(f"\n{'nhanh':22} {'CE truoc':>9} {'CE sau':>9} {'acc truoc':>10} {'acc sau':>9}")
for tag, ten in (("base", "tok_rank=0 (cu)"), ("tok", "tok_rank>0 (kenh)")):
    if tag not in d:
        print(f"{ten:22} {'--':>9} (chua chay)"); continue
    ev = d[tag]["eval"]
    a, b = ev[0], ev[-1]
    print(f"{ten:22} {a['ce']:9.4f} {b['ce']:9.4f} "
          f"{a['acc']*100:9.1f}% {b['acc']*100:8.1f}%")
if "base" in d and "tok" in d:
    bce, tce = d["base"]["eval"][-1]["ce"], d["tok"]["eval"][-1]["ce"]
    bac, tac = d["base"]["eval"][-1]["acc"], d["tok"]["eval"][-1]["acc"]
    print(f"\nCHENH (kenh - cu): CE {tce-bce:+.4f} (am = tot hon) | "
          f"acc {(tac-bac)*100:+.1f} diem")
    print("\nDOC: CE thap hon => cache mapped tai tao hanh vi 9B tot hon.")
    print("     NHUNG theo luat error-placement, chi CE KHONG du -- phai nhin")
    print("     acc (thi nghiem chuc nang). Neu CE tot ma acc khong nhuc nhich")
    print("     thi kenh CHUA chung minh duoc gi.")
PYEOF
echo "RUN_PROBE_TOK_EXIT"
