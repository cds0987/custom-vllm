#!/usr/bin/env bash
# BUOC 4 -- do NIEM PHONG cho chien dich gsm8k co cau truc.
#
# Cham 3 checkpoint tren CUNG mot tap niem phong de tach duoc cong cua RL
# khoi cong cua SFT warm-start:
#   1. sft_struct_v3          = "RL 0 buoc"  <- MOC 0, truoc gio con thieu
#   2. gsm_struct_rl_v2/best  = moc val tot nhat
#   3. gsm_struct_rl_v2/last  = het 1 epoch (508 buoc)
# roi McNemar tung cap (mcnemar.py).
#
# TAP NIEM PHONG: gsm8k split `test` (openai/gsm8k main/test), 250 mau dau.
# Tach hoan toan khoi split `train` ma mapper hoc tren do. 100 mau dau TRUNG
# voi tap niem phong cu -> so cu (joint49bb 8,0% / gsm_grpo_v1c 10,0%) van doi
# chieu duoc. n=250 chon vi n=100 lan truoc cho p=0,149 va p=0,114 -- xu huong
# thang 3-4:1 nhung KHONG du bang chung.
#
# TOC DO: gom lo decode CUNG DO DAI CHINH XAC (--decode-same-len 1). Gom lo
# thuong tung lam sai gsm8k vi stack_students dem TRAI attention KV bang 0;
# cung do dai => khong co token dem => dong nhat batch 1. KHONG tin suong:
# --verify-batch 24 chay lai batch 1 doi chieu, lech mot mau la DUNG HAN.
#
#   bash run_struct_sealed.sh                 # chay het
#   CKS="sft_struct_v3:best" bash run_struct_sealed.sh   # chi mot cai
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
N_SEAL="${N_SEAL:-250}"
DB="${DB:-8}"                 # so hang gom lo khi decode
VB="${VB:-24}"                # so mau cong kiem batch1 vs batchB
# Hau to prefix. BAT BUOC doi khi doi che do decode: eval_big "noi lai" bang
# cach TAI KET QUA CU TU HF theo prefix -- doi DB ma giu nguyen prefix thi no
# lang le dung lai so cu (dung bay da dinh o joint49cc: resume keo ve ket qua
# SAI tu HF, phai xoa ca HF lan local moi ra so dung).
SUF="${SUF:-}"
# "thu_muc:tag" -- tag chon mapper_<tag>.pt / lora_<tag> / lorat_<tag>
CKS="${CKS:-sft_struct_v3:best gsm_struct_rl_v2:best gsm_struct_rl_v2:last}"

for pkg in peft bitsandbytes datasets; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done

echo "=== [0/4] keo du lieu tap train (de KIEM RO RI) ==="
# BAT BUOC co file nay. Truoc day kiem ro ri chi chay "neu file ton tai" ->
# sau mot lan runtime recycle (mat /content) no se AM THAM BO QUA va van in
# ket qua nhu binh thuong. Kiem ro ri bi bo qua trong im lang la kieu hong
# te nhat: bao cao van dep, so van sai.
python3 - <<'PYEOF'
import os, pathlib
from huggingface_hub import hf_hub_download
d = pathlib.Path("/content/train_items.json")
if d.exists():
    print("da co", d)
else:
    p = hf_hub_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                        "joint_v1/train_items.json",
                        token=os.environ.get("HF_TOKEN"))
    d.write_bytes(pathlib.Path(p).read_bytes())
    print("KEO VE", d)
PYEOF
[ -f /content/train_items.json ] || {
  echo "KHONG CO train_items.json -> khong kiem ro ri duoc -> DUNG"; exit 1; }

echo "=== [1/4] dung tap niem phong $N_SEAL mau (gsm8k split test) ==="
python3 - "$N_SEAL" <<'PYEOF'
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("ext_bench", "ext_bench.py")
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
n = int(sys.argv[1])
out = pathlib.Path("/content/gsm_sealed.json")
if out.exists() and len(json.loads(out.read_text())) == n:
    items = json.loads(out.read_text())
    print("da co", out, f"({len(items)} mau)")
else:
    items = eb._gsm8k_items(n)
    for it in items:
        it["kind"] = "gsm8k"
    out.write_text(json.dumps(items))
    print("da dung", len(items), "mau niem phong ->", out)
# KIEM RO RI chay MOI LAN, ke ca khi tap niem phong da co san (truoc day
# nhanh "da co" thoat som -> lan chay lai KHONG kiem ro ri lan nao).
# Khong mau niem phong nao duoc nam trong tap train cua mapper.
tr = pathlib.Path("/content/train_items.json")
assert tr.exists(), "thieu train_items.json -- KHONG duoc bo qua kiem ro ri"
d = json.loads(tr.read_text())
hoi = {x["prompt"] for sp in ("train", "val") for x in d.get(sp, [])}
assert hoi, "train_items.json rong -> kiem ro ri vo nghia"
ro = sum(1 for it in items if it["prompt"] in hoi)
print(f"kiem ro ri train/test: {ro}/{len(items)} (doi chieu {len(hoi)} prompt)"
      + ("  <-- CO RO RI, DUNG LAI" if ro else "  (sach)"))
assert ro == 0, "tap niem phong dinh mau da train"
PYEOF
[ -f /content/gsm_sealed.json ] || { echo "khong dung duoc tap niem phong"; exit 1; }

echo "=== [2/4] keo checkpoint tu HF neu thieu ==="
python3 - $CKS <<'PYEOF'
import os, pathlib, shutil, sys
from huggingface_hub import snapshot_download
for spec in sys.argv[1:]:
    name, tag = spec.split(":")
    d = pathlib.Path(f"/content/{name}")
    if (d / f"mapper_{tag}.pt").exists():
        print("da co", spec); continue
    try:
        p = snapshot_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                              allow_patterns=[f"{name}/*"],
                              local_dir=f"/content/_hf_{name}",
                              token=os.environ.get("HF_TOKEN"))
        src = pathlib.Path(p) / name
        if any(src.glob("mapper_*.pt")):
            shutil.copytree(src, d, dirs_exist_ok=True)
            print("KEO VE", spec)
    except Exception as e:
        print("khong lay duoc", spec, type(e).__name__, str(e)[:90])
PYEOF

echo "=== [3/4] cham tung checkpoint ==="
for spec in $CKS; do
  name="${spec%%:*}"; tag="${spec##*:}"
  D="/content/$name"
  [ -f "$D/mapper_$tag.pt" ] || { echo "BO QUA $spec (khong co mapper_$tag.pt)"; continue; }
  LORA=""
  [ -d "$D/lora_$tag" ]  && LORA="--lora $D/lora_$tag"
  [ -d "$D/lorat_$tag" ] && LORA="$LORA --lora-t $D/lorat_$tag"
  PFX="sealed_${name}_${tag}${SUF}"
  echo "--- $spec (prefix $PFX) ---"
  EVALBIG_ITEMS=/content/gsm_sealed.json python3 -u eval_big.py mapped \
    --tgt-model Qwen/Qwen3.5-9B --max-len 6144 \
    --mapper "$D/mapper_$tag.pt" $LORA \
    --decode-batch "$DB" --decode-same-len 1 --verify-batch "$VB" \
    --benches gsm8k --hf-prefix "$PFX" \
    2>&1 | tee "/content/logs/${PFX}.log"
  st=${PIPESTATUS[0]}
  [ "$st" = 0 ] || { echo "LOI o $spec (status $st) -- DUNG"; exit 1; }
done

echo "=== [4/4] doc tay 8 mau + McNemar tung cap ==="
SUF="$SUF" python3 - $CKS <<'PYEOF'
import glob, itertools, os, json, pathlib, importlib.util
spec = importlib.util.spec_from_file_location("mcnemar", "mcnemar.py")
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
import sys
cks = sys.argv[1:]

def tim(name, tag):
    # eval_big ghi ra /content/logs/<hf_prefix>_mapped.json
    suf = os.environ.get("SUF", "")
    p = pathlib.Path(f"/content/logs/sealed_{name}_{tag}{suf}_mapped.json")
    if p.exists():
        return str(p)
    g = sorted(glob.glob(f"/content/logs/sealed_{name}_{tag}{suf}*.json"))
    return g[-1] if g else None

files = {}
for s in cks:
    name, tag = s.split(":")
    f = tim(name, tag)
    if f:
        files[s] = f
    else:
        print("khong thay file ket qua cho", s)

# QUY TAC 15: doc tay >=8 mau KEM DIEM truoc khi tin con so tong. Dinh dang
# co cau truc ket thuc bang 'Final Answer: N' -- phai TU MAT thay grader bat
# dung con so do, khong phai bat so cuoi trong khoi STEPS.
if files:
    s0 = list(files)[0]
    d0 = json.loads(pathlib.Path(files[s0]).read_text())
    d0 = d0.get("items", d0)
    seal = {x["id"]: x for x in json.loads(
        pathlib.Path("/content/gsm_sealed.json").read_text())}
    print(f"\n===== DOC TAY 8 MAU ({s0}) =====")
    for i, (k, v) in enumerate(list(d0.items())[:8]):
        txt = (v["txt"] if isinstance(v, dict) else "")[-260:]
        print(f"\n--- {k} | diem={v['hit'] if isinstance(v,dict) else v} "
              f"| dap an dung={seal.get(k,{}).get('expect')}")
        print(repr(txt))

print("\n===== NHANH CHAM DIEM (bao nhieu diem den tu nhanh du phong?) =====")
for s, f in files.items():
    d = json.loads(pathlib.Path(f).read_text())
    d = d.get("items", d)
    dem = {}
    for v in d.values():
        if isinstance(v, dict) and v.get("hit"):
            dem[v.get("how", "?")] = dem.get(v.get("how", "?"), 0) + 1
    tong = sum(dem.values())
    print(f"{s}: {tong} diem | " + " ".join(f"{k}={n}" for k, n in sorted(dem.items())))
    if dem.get("so_cuoi", 0) > 0.25 * max(tong, 1):
        print("  CANH BAO: >25% diem den tu nhanh 'so_cuoi' (dau ra co the bi "
              "CAT truoc khi viet Final Answer) -- doc tay truoc khi tin.")
    if dem.get("?", 0):
        print("  (ket qua cu chua ghi 'how' -- chay lai moi co)")

print("\n===== McNEMAR TUNG CAP =====")
for a, b in itertools.combinations(files, 2):
    r = mc.mcnemar(mc.doc(files[a]), mc.doc(files[b]))
    print(f"\n{a}  vs  {b}")
    print(f"  {a}: {r['a_dung']}/{r['n']} = {100*r['a_dung']/max(r['n'],1):.1f}%")
    print(f"  {b}: {r['b_dung']}/{r['n']} = {100*r['b_dung']/max(r['n'],1):.1f}%")
    print(f"  lech {r['a_thang']}-{r['b_thang']} | chi2={r['chi2']} "
          f"p={r['p']} ({r['cach']})")
    print("  -> " + ("CO y nghia thong ke" if r["p"] < 0.05 else
                     "CHUA du bang chung, khong duoc goi la cai tien"))
PYEOF
echo "RUN_STRUCT_SEALED_EXIT"
