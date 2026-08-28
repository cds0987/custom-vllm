"""gen_pseudo -- sinh PSEUDO-GOLD bang chinh model dich tu doc.

User chot 2026-08-28: "cho no hoc ca buoc reasoning cua model large de cai
thien" + "ta ko can no out perform 9b, can map gan 9b nhat".

DIEU DO DINH NGHIA LUON DICH HOC: khong phai dap an cua nguoi viet, ma la
CHINH QUY DAO MA 9B TU DI. Dich trung voi thuoc do (retention = mapped/self),
nen mapper hoc dung cai dang duoc cham.

Hai benh khac nhau ma cung mot thuoc:
  bbh/musr  -- gold hien tai la DUNG 1 TOKEN ("B", "valid"). Do duoc: musr
               474/474 = 100% va bbh 634/2500 = 25,4% item co gold 1 token.
               Pseudo-gold bien 1 token -> 24-48 token giam sat day.
  gsm8k     -- gold DA la 256 token day du, nhung la loi giai NGUOI VIET.
               self 7/13 chung minh 9B giai duoc; chi la bang van phong khac.
               Bat mapper tai tao mot quy dao ma chinh model dich khong tu di
               la bai kho vo ich. Lay loi giai cua chinh 9B = on-policy.

QUY TAC AN TOAN: chi lay pseudo-gold khi no CHAM DIEM DUNG. Sai thi giu gold
tham chieu ngan -> khong mat mau nao, va khong bao gio day mapper tai tao
suy luan SAI.

Sinh THEO LO (batch): tuan tu ~1,05 trieu token = ~9 gio, khong chap nhan
duoc. Lo 8 dua ve ~1-1,5 gio.

Chay:
  python -u gen_pseudo.py --data /content/train_items.json \\
      --out /content/pseudo_gold.json --tgt-model Qwen/Qwen3.5-9B
"""

import argparse
import importlib.util
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")
gd = _load("gen_data")

# bam theo ext_bench.N_NEW: do dai sinh luc do that
N_NEW = {"gsm8k": 320, "bbh": 48, "musr": 24,
         "suite_rag": 24, "suite_mid": 24, "suite_math": 24, "suite_swe": 24}
# tran token cua pseudo-gold khi dua vao train (bao do VRAM cua cap 4->9
# cho phep gold 256 toi ctx 16384)
GOLD_CAP = {"gsm8k": 256, "bbh": 48, "musr": 24,
            "suite_rag": 24, "suite_mid": 24, "suite_math": 24, "suite_swe": 24}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/train_items.json")
    ap.add_argument("--out", default="/content/pseudo_gold.json")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--kinds", default="bbh,musr,gsm8k",
                    help="chi sinh cho cac ho nay (cac ho khac giu gold cu)")
    ap.add_argument("--limit", type=int, default=0, help="0 = tat ca")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="joint49")
    args = ap.parse_args()

    kinds = set(args.kinds.split(","))
    data = json.loads(Path(args.data).read_text())
    items = [it for it in data["train"] if it["kind"] in kinds]
    if args.limit:
        items = items[:args.limit]
    print(f"se sinh cho {len(items)} item: "
          f"{dict(Counter(i['kind'] for i in items))}", flush=True)

    tok, model = e5.load_4bit(args.tgt_model)
    tok.padding_side = "left"        # BAT BUOC de sinh theo lo cho dung
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"     # cau hoi nam O CUOI prompt

    # gom theo ho de moi lo co cung max_new_tokens
    by_kind = defaultdict(list)
    for it in items:
        by_kind[it["kind"]].append(it)

    out, t0, done = {}, time.time(), 0
    n_tot = len(items)
    for kind, group in by_kind.items():
        n_new = N_NEW.get(kind, 24)
        cap = GOLD_CAP.get(kind, 24)
        for s in range(0, len(group), args.batch):
            chunk = group[s:s + args.batch]
            enc = tok([c["prompt"] for c in chunk], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=args.max_len).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=n_new,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for c, row in zip(chunk, new):
                txt = tok.decode(row, skip_special_tokens=True)
                hit = gd.score_item(c, txt)
                if hit:
                    # cat theo tran gold; giu nguyen van ban 9B tu viet
                    ids = tok(txt, add_special_tokens=False)["input_ids"][:cap]
                    out[c["id"]] = {"kind": kind, "hit": 1,
                                    "gold": tok.decode(ids)}
            done += len(chunk)
            del enc, gen, new
            torch.cuda.empty_cache()
            if (s // args.batch) % 10 == 0:
                el = time.time() - t0
                print(f"{kind} {done}/{n_tot} | dung {len(out)} "
                      f"({100*len(out)/max(done,1):.0f}%) | {el/60:.0f} phut "
                      f"| con ~{el/max(done,1)*(n_tot-done)/60:.0f} phut",
                      flush=True)
        json.dump(out, open(args.out, "w"), ensure_ascii=False)
        print(f"== xong ho {kind}: {len(out)} pseudo-gold tich luy", flush=True)

    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    per = Counter(v["kind"] for v in out.values())
    tot = Counter(i["kind"] for i in items)
    print("\n=== KET QUA PSEUDO-GOLD ===")
    print(f"{'ho':10} {'tong':>6} {'9B tu lam dung':>15} {'ty le':>7}")
    for k in sorted(tot):
        print(f"{k:10} {tot[k]:6} {per[k]:15} {100*per[k]/tot[k]:6.1f}%")
    print(f"{'TONG':10} {n_tot:6} {len(out):15} "
          f"{100*len(out)/max(n_tot,1):6.1f}%")
    print("Item 9B lam SAI giu nguyen gold tham chieu ngan — khong mat mau, "
          "va khong day mapper tai tao suy luan sai.")

    if args.hf_repo and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=args.out,
                path_in_repo=f"{args.hf_prefix}/{Path(args.out).name}",
                repo_id=args.hf_repo)
            print(f"HF-UP {Path(args.out).name}")
        except Exception as ex:
            print(f"HF-UP FAIL: {type(ex).__name__}: {ex}")
    print("GEN_PSEUDO_EXIT", flush=True)


if __name__ == "__main__":
    main()
