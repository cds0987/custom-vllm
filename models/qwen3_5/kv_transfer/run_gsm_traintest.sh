#!/usr/bin/env bash
# gsm8k: do TREN TAP TRAIN va TREN TAP NIEM PHONG cho MOT checkpoint, de biet
# mo hinh "hoc khong noi" hay "hoc thuoc long".
#   train ~ test  -> khong qua khop, gioi han nang luc that
#   train >> test -> qua khop
#
# joint49bb da co so: train 8,3% (60 mau) / niem phong 8,0% (100 mau).
#
#   CK=joint49cc bash run_gsm_traintest.sh
#   CK=joint49cc GSM_TRAIN_N=60 bash run_gsm_traintest.sh
#
# gsm8k BAT BUOC decode-batch 1 (gen 320 token/mau — cong kiem batch=8 da bat
# duoc lech that o mau big/gsm8k/221).
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
CK="${CK:-joint49cc}"
N="${GSM_TRAIN_N:-60}"
D="/content/$CK"

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done
[ -f "$D/mapper_best.pt" ] || { echo "KHONG THAY $D/mapper_best.pt"; exit 1; }

LORA=""
[ -d "$D/lora_best" ]  && LORA="--lora $D/lora_best"
[ -d "$D/lorat_best" ] && LORA="$LORA --lora-t $D/lorat_best"
echo "checkpoint: $CK | adapter: ${LORA:-khong co}"

echo "=== [1/3] dung tap kiem gsm8k tu TRAIN ($N mau) ==="
if [ -f /content/gsm_train_items.json ]; then
  echo "da co, bo qua"
else
  python3 - <<PYEOF
import json, random, pathlib
random.seed(7)   # CUNG seed voi run_49bb_traincheck.sh -> CUNG $N mau
N = ${N}
data = json.loads(pathlib.Path("/content/train_items_gsm.json").read_text())
xs = [it for it in data["train"] if it.get("kind") == "gsm8k"]
random.shuffle(xs)
out = [{"bench": "gsm8k", "kind": "gsm8k", "sub": "train",
        "id": "train/" + str(it.get("id", j)),
        "prompt": it["prompt"], "expect": it["expect"]}
       for j, it in enumerate(xs[:N]) if it.get("expect")]
pathlib.Path("/content/gsm_train_items.json").write_text(json.dumps(out))
print("dung", len(out), "mau gsm8k tu TRAIN")
PYEOF
fi

echo "=== [2/3] cham gsm8k tren TRAIN (batch=1) ==="
EVALBIG_ITEMS=/content/gsm_train_items.json python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$D/mapper_best.pt" $LORA \
  --decode-batch 1 --verify-batch 0 \
  --benches gsm8k \
  --hf-prefix "gsm_train_$CK" || exit 1

echo "=== [3/3] cham gsm8k tren NIEM PHONG (100 mau, batch=1) ==="
python3 -u eval_big.py mapped \
  --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
  --mapper "$D/mapper_best.pt" $LORA \
  --decode-batch 1 --verify-batch 0 \
  --benches gsm8k \
  --hf-prefix "gsm_seal_$CK" || exit 1

echo "=== KET QUA: $CK tren gsm8k ==="
python3 - "$CK" <<'PYEOF'
import json, pathlib, sys
ck = sys.argv[1]
L = pathlib.Path("/content/logs")
seal_items = {it["id"]: it for it in
              json.loads(pathlib.Path("/content/eval_big_items.json").read_text())}

def pct(path, only_gsm=False):
    p = L / path
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    if only_gsm:
        d = {k: v for k, v in d.items()
             if seal_items.get(k, {}).get("bench") == "gsm8k"}
    if not d:
        return None, None
    return sum(v["hit"] for v in d.values()), len(d)

print(f"\n{'tap':14} {'dung':>8} {'n':>5} {'ty le':>8}")
for ten, f, og in (("TRAIN", f"gsm_train_{ck}_mapped.json", False),
                   ("NIEM PHONG", f"gsm_seal_{ck}_mapped.json", True)):
    h, n = pct(f, og)
    if h is None:
        print(f"{ten:14} {'--':>8} {'--':>5} {'chua co':>8}"); continue
    print(f"{ten:14} {h:8} {n:5} {100*h/n:7.1f}%")
print("\nDOI CHIEU joint49bb: TRAIN 8,3% (5/60) | NIEM PHONG 8,0% (8/100)")
print("           9B tu tra loi (tran): 89,0%")
print("\nDOC: train ~ test -> khong qua khop, gioi han nang luc that")
print("     train >> test -> qua khop (thuoc bai)")
PYEOF

python3 - "$CK" <<'PYEOF'
import os, sys, pathlib
from huggingface_hub import HfApi
ck = sys.argv[1]
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in [f"/content/logs/gsm_train_{ck}_mapped.json",
          f"/content/logs/gsm_seal_{ck}_mapped.json",
          "/content/gsm_train_items.json",
          "/content/logs/gsm_traintest.log"]:
    p = pathlib.Path(f)
    if p.exists():
        try:
            api.upload_file(path_or_fileobj=f, repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + p.name)
            print("HF-UP", p.name)
        except Exception as e:
            print("HF-UP FAIL", p.name, type(e).__name__, str(e)[:80])
PYEOF
echo "RUN_GSM_TRAINTEST_EXIT"
