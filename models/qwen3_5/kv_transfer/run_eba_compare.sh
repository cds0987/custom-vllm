#!/usr/bin/env bash
# So best@50 vs last@300 cua eba_grpo_v1 tren bo EBA held-out (seed KHAC
# luc train), + kiem dinh McNemar -- dung dung pattern run_gsm_traintest.sh.
#   bash run_eba_compare.sh
#   N=300 bash run_eba_compare.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
for pkg in peft bitsandbytes; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done
D="${CKPT_DIR:-/content/eba_grpo_v1}"
N="${N:-200}"
[ -f "$D/mapper_best.pt" ] || { echo "KHONG THAY $D/mapper_best.pt"; exit 1; }

echo "=== [1/2] cham best (buoc 50) ==="
python3 -u eba_eval_one.py --ckpt-dir "$D" --tag best --n "$N" \
  --out /content/logs/eba_eval_best.json || exit 1

echo "=== [2/2] cham last (buoc 300) ==="
python3 -u eba_eval_one.py --ckpt-dir "$D" --tag last --n "$N" \
  --out /content/logs/eba_eval_last.json || exit 1

python3 - <<'PYEOF'
import json, math, pathlib
a = json.loads(pathlib.Path("/content/logs/eba_eval_best.json").read_text())
b = json.loads(pathlib.Path("/content/logs/eba_eval_last.json").read_text())
ids = sorted(set(a["results"]) & set(b["results"]))
assert len(ids) == a["n"] == b["n"], "mismatch item id giua 2 lan cham -- seed lech?"

def avg(res, k):
    return sum(res[i][k] for i in ids) / len(ids)

print(f"\n{'checkpoint':12} {'A_tb':>7} {'B_tb':>7} {'C_tb':>7} (n={len(ids)})")
for tag, res in (("best@50", a["results"]), ("last@300", b["results"])):
    print(f"{tag:12} {avg(res,'A'):7.3f} {avg(res,'B'):7.3f} {avg(res,'C'):7.3f}")

ac = [a["results"][i]["C"] for i in ids]
bc = [b["results"][i]["C"] for i in ids]
n_bc = sum(1 for x, y in zip(ac, bc) if x and not y)   # best dung, last sai
n_cb = sum(1 for x, y in zip(ac, bc) if not x and y)   # best sai, last dung
if n_bc + n_cb == 0:
    chi2, p = 0.0, 1.0
else:
    chi2 = (abs(n_bc - n_cb) - 1) ** 2 / (n_bc + n_cb)
    p = math.erfc(math.sqrt(chi2 / 2))
print(f"\nMcNemar (C): best-dung/last-sai={n_bc}  best-sai/last-dung={n_cb}  "
      f"chi2={chi2:.3f}  p={p:.4f}")
print("p<0.05 -> chenh lech co y nghia thong ke; p>=0.05 -> CHUA phan biet "
      "duoc voi nhieu (dung ket luan 'X tot hon Y' o muc nay)")

out = {"best": a, "last": b, "mcnemar_C": {"chi2": chi2, "p": p,
       "best_win": n_bc, "last_win": n_cb}}
pathlib.Path("/content/logs/eba_compare_final.json").write_text(json.dumps(out))
print("\nda ghi /content/logs/eba_compare_final.json")
PYEOF

python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/eba_eval_best.json", "/content/logs/eba_eval_last.json",
         "/content/logs/eba_compare_final.json"]:
    p = pathlib.Path(f)
    if p.exists():
        try:
            api.upload_file(path_or_fileobj=f, repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + p.name)
            print("HF-UP", p.name)
        except Exception as e:
            print("HF-UP FAIL", p.name, type(e).__name__, str(e)[:80])
PYEOF
echo "RUN_EBA_COMPARE_EXIT"
