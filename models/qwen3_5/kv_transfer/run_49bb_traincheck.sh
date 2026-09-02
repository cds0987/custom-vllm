#!/usr/bin/env bash
# joint49bb — CHAM TREN CHINH TAP TRAIN de phan biet hai kha nang:
#   (a) "hoc khong noi": diem tren TRAIN cung thap (~8% nhu niem phong)
#       -> gioi han nang luc that cua mapper, khong phai thieu du lieu.
#   (b) "hoc thuoc long" (qua khop): diem tren TRAIN cao (>50%) ma niem phong
#       chi 8% -> mapper thuoc bai chu khong tong quat hoa duoc.
#
# Cham CA HAI bo trong pham vi hien tai (gsm8k + suite_swe) de co doi chieu:
# suite_swe niem phong 77,2% — neu train cua no cung ~77% thi khong qua khop,
# cho thay khoang cach cua gsm8k la rieng biet chu khong phai benh chung.
#
#   bash run_49bb_traincheck.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
CK="/content/joint49bb"
N="${TRAIN_N:-60}"   # moi bo lay N mau tu TRAIN

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -2
done

[ -f "$CK/mapper_best.pt" ] || { echo "KHONG THAY $CK/mapper_best.pt"; exit 1; }

echo "=== [1/3] dung tap kiem tu TRAIN (dung dinh dang eval_big) ==="
python3 - <<PYEOF
import json, random, pathlib
random.seed(7)
N = ${N}
data = json.loads(pathlib.Path("/content/train_items_gsm.json").read_text())
train = data["train"]

out = []
for kind in ("gsm8k", "suite_swe"):
    xs = [it for it in train if it.get("kind") == kind]
    random.shuffle(xs)
    for it in xs[:N]:
        # eval_big can: bench, kind, sub, id, prompt, expect
        exp = it.get("expect", it.get("answer", it.get("gold", "")))
        if not exp:
            continue
        out.append({"bench": kind, "kind": kind, "sub": "train",
                    "id": "train/" + str(it.get("id", len(out))),
                    "prompt": it["prompt"], "expect": exp})
    print(kind, "lay duoc", sum(1 for o in out if o["bench"] == kind), "mau")

pathlib.Path("/content/traincheck_items.json").write_text(json.dumps(out))
print("TONG", len(out), "mau -> /content/traincheck_items.json")
PYEOF

LORA=""
[ -d "$CK/lora_best" ]  && LORA="--lora $CK/lora_best"
[ -d "$CK/lorat_best" ] && LORA="$LORA --lora-t $CK/lorat_best"

echo "=== [2/3] cham suite_swe TREN TRAIN (batch 8 — an toan, gen ngan) ==="
EVALBIG_ITEMS=/content/traincheck_items.json python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA \
  --decode-batch 8 --verify-batch 25 \
  --benches suite_swe \
  --hf-prefix "evalbig_49bb_train" || exit 1

echo "=== [3/3] cham gsm8k TREN TRAIN (batch=1 — bat buoc, gen 320 token) ==="
EVALBIG_ITEMS=/content/traincheck_items.json python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$CK/mapper_best.pt" $LORA \
  --decode-batch 1 --verify-batch 0 \
  --benches gsm8k \
  --hf-prefix "evalbig_49bb_train" || exit 1

echo "=== KET QUA: train vs niem phong ==="
python3 - <<'PYEOF'
import json, pathlib
items = {it["id"]: it for it in json.loads(pathlib.Path("/content/traincheck_items.json").read_text())}
tr = json.loads(pathlib.Path("/content/logs/evalbig_49bb_train_mapped.json").read_text())
NIEM_PHONG = {"suite_swe": 77.2, "gsm8k": 8.0}
print(f"\n{'bo':12} {'n(train)':>9} {'TRAIN':>8} {'NIEM PHONG':>12} {'chenh':>8}")
for bench in ("suite_swe", "gsm8k"):
    ids = [i for i in tr if items.get(i, {}).get("bench") == bench]
    if not ids:
        continue
    h = sum(tr[i]["hit"] for i in ids)
    pct = 100 * h / len(ids)
    sealed = NIEM_PHONG[bench]
    print(f"{bench:12} {len(ids):9} {pct:7.1f}% {sealed:11.1f}% {pct-sealed:+7.1f}")
print("\nDOC: train ~ niem phong  -> KHONG qua khop (gioi han nang luc that)")
print("     train >> niem phong -> QUA KHOP (thuoc bai, khong tong quat hoa)")
PYEOF

python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/evalbig_49bb_train_mapped.json", "/content/logs/49bb_traincheck.log"]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f,
                            repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + os.path.basename(f))
            print("HF-UP", f)
        except Exception as e:
            print("HF-UP FAIL", f, type(e).__name__, str(e)[:100])
PYEOF
echo "RUN_49BB_TRAINCHECK_EXIT"
