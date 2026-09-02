#!/usr/bin/env bash
# PROBE TRICH XUAT SO — phan xu hai chan doan cho gsm8k 8%:
#   (a) thong tin CO trong cache, loi o khau SUY LUAN  -> --w-entity la thuoc
#   (b) thong tin KHONG toi duoc 9B                    -> phai mo --gdn-terms
#
# Cach lam: lay chinh cac bai gsm8k, BO phan tinh toan, chi bao model NHAC LAI
# mot con so co san trong de. Neu nhac lai dung ma van sai khi tinh -> (a).
# Neu nhac lai cung sai -> (b).
#
# Hai bien the de tach anh huong VI TRI trong ngu canh:
#   probe_dau  : nhac lai con so DAU TIEN  (nong, gan dau ngu canh)
#   probe_cuoi : nhac lai con so CUOI CUNG (sau, gan cau hoi)
# Neu dau dung / cuoi sai -> mat thong tin theo DO SAU, khong phai mat toan bo.
#
#   bash run_49bb_probe_so.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
CK="/content/joint49bb"
N="${PROBE_N:-40}"

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done
[ -f "$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK/mapper_best.pt"; exit 1; }

echo "=== [1/2] dung bo probe tu chinh cac bai gsm8k NIEM PHONG ==="
python3 - <<PYEOF
import json, re, pathlib, random
random.seed(11)
N = ${N}
items = json.loads(pathlib.Path("/content/eval_big_items.json").read_text())
gsm = [it for it in items if it.get("bench") == "gsm8k"]
random.shuffle(gsm)

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
out, skipped = [], 0
for it in gsm:
    p = it["prompt"]
    # tach rieng phan DE BAI (giua "Problem:" va "<think>")
    if "Problem:" not in p:
        skipped += 1; continue
    head, rest = p.split("Problem:", 1)
    body = rest.split("<think>")[0]
    nums = [m.group(0) for m in NUM.finditer(body)]
    nums = [n for n in nums if n.strip(",.")]
    if len(nums) < 2:
        skipped += 1; continue

    for tag, want in (("dau", nums[0]), ("cuoi", nums[-1])):
        hoi = ("FIRST" if tag == "dau" else "LAST")
        # GIU NGUYEN de bai (de cache 4B chua dung noi dung do), chi doi CAU HOI
        prompt = (head + "Problem:" + body.rstrip() +
                  f"\n\nDo NOT solve or calculate anything. Simply repeat the "
                  f"{hoi} number that appears in the problem above.\n"
                  f"Final Answer: ")
        out.append({"bench": "gsm8k", "kind": "gsm8k", "sub": "probe_" + tag,
                    "id": f"probe_{tag}/" + it["id"].split("/")[-1],
                    "prompt": prompt, "expect": want.strip(",.")})
    if len(out) >= 2 * N:
        break

pathlib.Path("/content/probe_so_items.json").write_text(json.dumps(out))
print(f"dung {len(out)} muc probe ({len(out)//2} bai x 2 bien the), bo qua {skipped}")
PYEOF

LORA=""
[ -d "$CK/lora_best" ]  && LORA="--lora $CK/lora_best"
[ -d "$CK/lorat_best" ] && LORA="$LORA --lora-t $CK/lorat_best"

echo "=== [2/2] chay probe qua duong ong mapped (batch=1, gen 24 token) ==="
# gen 24 token la du cho MOT con so — khong can 320 nhu gsm8k that.
EVALBIG_ITEMS=/content/probe_so_items.json EVALBIG_GEN_LEN="gsm8k:24" \
python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA \
  --decode-batch 1 --verify-batch 0 \
  --benches gsm8k \
  --hf-prefix "evalbig_49bb_probeso" || exit 1

echo "=== KET QUA PROBE ==="
python3 - <<'PYEOF'
import json, pathlib
items = {it["id"]: it for it in json.loads(pathlib.Path("/content/probe_so_items.json").read_text())}
res = json.loads(pathlib.Path("/content/logs/evalbig_49bb_probeso_mapped.json").read_text())
print(f"\n{'bien the':14} {'n':>4} {'nhac lai DUNG':>15}")
for tag in ("dau", "cuoi"):
    ids = [i for i in res if items.get(i, {}).get("sub") == "probe_" + tag]
    if not ids:
        continue
    h = sum(res[i]["hit"] for i in ids)
    print(f"probe_{tag:8} {len(ids):4} {100*h/len(ids):14.1f}%")
print("\nDOC:")
print("  ca hai CAO (>80%)  -> thong tin CO trong cache; loi o khau SUY LUAN")
print("                        -> bat --w-entity (da dung san, chua tung bat)")
print("  ca hai THAP        -> thong tin KHONG toi duoc 9B -> phai mo --gdn-terms")
print("  dau cao / cuoi thap-> mat theo DO SAU ngu canh (van la van de dung luong)")
print("\n=== 8 mau de doc tay ===")
for tag in ("dau", "cuoi"):
    ids = [i for i in res if items.get(i, {}).get("sub") == "probe_" + tag][:4]
    for i in ids:
        print(f"  [{tag}] can={items[i]['expect']!r:>10} | hit={res[i]['hit']} | ra={res[i]['txt'][:70]!r}")
PYEOF

python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/evalbig_49bb_probeso_mapped.json",
          "/content/probe_so_items.json", "/content/logs/49bb_probeso.log"]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f, repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + os.path.basename(f))
            print("HF-UP", f)
        except Exception as e:
            print("HF-UP FAIL", f, type(e).__name__, str(e)[:100])
PYEOF
echo "RUN_49BB_PROBESO_EXIT"
