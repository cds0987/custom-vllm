#!/usr/bin/env bash
# BO suite_swe LON (mac dinh 600 mau) — do lai voi n lon de thu hep khoang tin
# cay, va so TRUC TIEP joint49cc (gdn-terms 4) voi joint49bb (gdn-terms 1)
# tren CUNG bo de, cung engine.
#
# Bo niem phong cu chi 123 mau suite_swe (sai so ~±4 diem). 600 mau -> ~±2.
# Seed 90210 KHAC seed 31337 cua bo niem phong va khac seed cua train ->
# kiem ro ri bang CHUOI PROMPT truoc khi chay (khong tin suy luan chi so).
#
#   bash run_swe_big.sh            # 600 mau, ca hai checkpoint
#   SWE_N=300 bash run_swe_big.sh  # nhanh hon
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
N="${SWE_N:-600}"

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done

echo "=== [1/4] dung bo suite_swe LON (seed 90210, khac bo niem phong) ==="
if [ -f /content/swe_big_items.json ]; then
  echo "da co, bo qua"
else
  python3 - <<PYEOF
import importlib.util, json, pathlib
from transformers import AutoTokenizer
H = pathlib.Path(".")
def _load(name):
    spec = importlib.util.spec_from_file_location(name, H / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sg = _load("suite_gen")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

N = ${N}
sg.build_suite(N, [1024, 2048, 4096], ["swe"], "/tmp/_swe_big.json", tok, seed=90210)
raw = json.load(open("/tmp/_swe_big.json"))
items = [{"bench": "suite_swe", "kind": "suite_swe", "sub": str(it.get("ctx")),
          "id": f"swebig/{j}", "prompt": it["prompt"], "expect": it["expect"]}
         for j, it in enumerate(raw)]

# ---- KIEM RO RI bang CHUOI PROMPT (khong tin chi so/seed) ----
cu = set()
for f in ("/content/train_items_gsm.json", "/content/eval_big_items.json"):
    p = pathlib.Path(f)
    if not p.exists():
        print("CANH BAO: khong thay", f, "-> kiem ro ri KHONG day du"); continue
    d = json.loads(p.read_text())
    xs = (d["train"] + d["val"] + d.get("test", [])) if isinstance(d, dict) else d
    cu |= {x["prompt"] for x in xs}
trung = [it["id"] for it in items if it["prompt"] in cu]
assert not trung, f"RO RI {len(trung)} mau trung du lieu cu: {trung[:5]}"
print(f"kiem ro ri: 0/{len(items)} mau trung {len(cu)} prompt da dung")

pathlib.Path("/content/swe_big_items.json").write_text(json.dumps(items))
print("dung", len(items), "mau -> /content/swe_big_items.json")
PYEOF
fi

run_ck () {   # $1 = ten checkpoint, $2 = them co (vd --no-ctx drop), $3 = hau to
  local CK="/content/$1"
  [ -f "$CK/mapper_best.pt" ] || { echo "BO QUA $1 (khong thay mapper_best.pt)"; return 0; }
  local LORA=""
  [ -d "$CK/lora_best" ]  && LORA="--lora $CK/lora_best"
  [ -d "$CK/lorat_best" ] && LORA="$LORA --lora-t $CK/lorat_best"
  echo "--- $1 ${2:-} ---"
  EVALBIG_ITEMS=/content/swe_big_items.json python3 -u eval_big.py mapped \
    --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
    --mapper "$CK/mapper_best.pt" $LORA ${2:-} \
    --decode-batch "${DBATCH:-8}" --verify-batch "${VBATCH:-25}" \
    --benches suite_swe \
    --hf-prefix "swebig_$1$3" || return 1
}

echo "=== [2/4] joint49cc (gdn-terms 4) tren bo LON ==="
run_ck joint49cc "" "" || exit 1

echo "=== [3/4] joint49bb (gdn-terms 1) tren CUNG bo — doi chung ==="
run_ck joint49bb "" "" || exit 1

echo "=== [4/4] joint49cc ctx-BO (chot chan an gian) ==="
VBATCH=0 run_ck joint49cc "--no-ctx drop" "_drop" || exit 1

echo "=== KET QUA ==="
python3 - <<'PYEOF'
import json, pathlib
n_all = len(json.loads(pathlib.Path("/content/swe_big_items.json").read_text()))
print(f"\n{'checkpoint':22} {'n':>5} {'suite_swe':>10}")
for name, f in (("joint49cc (terms 4)", "swebig_joint49cc_mapped.json"),
                ("joint49bb (terms 1)", "swebig_joint49bb_mapped.json"),
                ("joint49cc ctx-BO",    "swebig_joint49cc_drop_mapped.json")):
    p = pathlib.Path("/content/logs") / f
    if not p.exists():
        print(f"{name:22} {'--':>5} {'chua co':>10}"); continue
    d = json.loads(p.read_text())
    h = sum(v["hit"] for v in d.values())
    print(f"{name:22} {len(d):5} {100*h/max(len(d),1):9.1f}%")
print(f"\n(niem phong cu 123 mau: joint49bb 77,2% | joint49z 52,8%)")
PYEOF

python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/swe_big_items.json", "/content/logs/swe_big.log"]:
    if pathlib.Path(f).exists():
        try:
            api.upload_file(path_or_fileobj=f, repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + os.path.basename(f))
            print("HF-UP", f)
        except Exception as e:
            print("HF-UP FAIL", f, type(e).__name__, str(e)[:80])
PYEOF
echo "RUN_SWE_BIG_EXIT"
