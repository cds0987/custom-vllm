"""gen_pseudo_vllm -- sinh pseudo-gold bang vLLM OFFLINE (user 2026-08-28:
"sao ko tham khao unsloth fast language model hay vllm mode offline?").

VI SAO DOI SANG vLLM. Ban transformers (gen_pseudo.py) chay ~11,8 token/giay
(bnb-4bit, batch 1, eager) -- da do trong chinh du an o E2/E3. Sinh 1,05
trieu token = ~9 gio tuan tu, ~1-1,5h voi lo 8. vLLM offline lam viec nay o
~390 token/giay (so da do cua 9B champion tren L4) -> **~5 phut**.

Ve Unsloth: FastLanguageModel DA BI DONG o E8 -- khong dung duoc cho loss
tren cache (tra past=None khi training, tra processor da phuong thuc). Phan
suy luan nhanh cua no ban chat la boc vLLM, nen goi thang vLLM sach hon.

CHON TEACHER. Muc tieu user chot la "map gan 9B nhat", nen teacher PHAI khop
model dich dang duoc do lam tran self. Ban transformers dung bnb-4bit ->
o day dung quantization="bitsandbytes" tren CUNG stock checkpoint. Neu
duong bnb cua vLLM khong nap duoc thi ha ve bf16 (--quant bf16) va GHI RO,
khong im lang doi teacher.

RUI RO THAP nho BO LOC: chi giu dau ra CHAM DIEM DUNG. Khac biet nho giua
cac duong luong tu khong lam hong du lieu -- cung lam la giu it mau hon.

BAT BUOC: LLM() phai nam trong `if __name__ == "__main__":` (quy tac du an --
dat o module level thi chet vi spawn).

Chay:
  python -u gen_pseudo_vllm.py --data /content/train_items.json \\
      --out /content/pseudo_gold.json --model Qwen/Qwen3.5-9B --quant bnb
"""

import argparse
import importlib.util
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gd = _load("gen_data")

# (user duyet 2026-09-01) 4 ho QUAN HE mo tu 24 -> 200 token: 24 chi du cho
# dap an ngan + mao dau suy luan bi CAT CUT, khong phai CoT that. Phia train
# (e9_joint.py --gold-cap 256) da san sang nhan gold dai toi 256 -> ha tang
# khong thieu, chi ngan sach SINH du lieu dang bo hep. suite_math/bbh/gsm8k
# GIU NGUYEN (khong phai muc tieu dot nay).
N_NEW = {"gsm8k": 320, "bbh": 48, "musr": 200,
         "suite_rag": 200, "suite_mid": 200, "suite_math": 24, "suite_swe": 200}
GOLD_CAP = {"gsm8k": 256, "bbh": 48, "musr": 200,
            "suite_rag": 200, "suite_mid": 200, "suite_math": 24, "suite_swe": 200}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/train_items.json")
    ap.add_argument("--out", default="/content/pseudo_gold.json")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--quant", default="bnb", choices=["bnb", "bf16"],
                    help="bnb = khop y het teacher transformers dang dung lam "
                         "tran self; bf16 = du phong khi duong bnb hong")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--kinds", default="bbh,musr,gsm8k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "val", "ca"])
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="joint49")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kinds = set(args.kinds.split(","))
    data = json.loads(Path(args.data).read_text())
    pool = (data["train"] + data["val"] if args.split == "ca"
            else data[args.split])
    items = [it for it in pool if it["kind"] in kinds]
    if args.limit:
        items = items[:args.limit]
    print(f"se sinh cho {len(items)} item: "
          f"{dict(Counter(i['kind'] for i in items))}", flush=True)

    kw = dict(model=args.model, max_model_len=args.max_len,
              gpu_memory_utilization=args.util, enforce_eager=False)
    if args.quant == "bnb":
        kw["quantization"] = "bitsandbytes"
    t0 = time.time()
    llm = LLM(**kw)
    print(f"vLLM nap xong {time.time()-t0:.0f}s (quant={args.quant})",
          flush=True)

    # gom theo ho: moi ho mot max_tokens rieng (vLLM tu lo continuous batching)
    by_kind = defaultdict(list)
    for it in items:
        by_kind[it["kind"]].append(it)

    out, n_tok = {}, 0
    tok = llm.get_tokenizer()
    for kind, group in by_kind.items():
        sp = SamplingParams(temperature=0.0, max_tokens=N_NEW.get(kind, 24))
        t1 = time.time()
        res = llm.generate([g["prompt"] for g in group], sp)
        dt = time.time() - t1
        cap, hit_n = GOLD_CAP.get(kind, 24), 0
        for g, r in zip(group, res):
            txt = r.outputs[0].text
            n_tok += len(r.outputs[0].token_ids)
            if gd.score_item(g, txt):
                ids = tok(txt, add_special_tokens=False)["input_ids"][:cap]
                out[g["id"]] = {"kind": kind, "hit": 1, "gold": tok.decode(ids)}
                hit_n += 1
        print(f"{kind}: {len(group)} mau, {hit_n} dung "
              f"({100*hit_n/len(group):.1f}%), {dt:.0f}s "
              f"({n_tok/max(time.time()-t0,1):.0f} tok/s tich luy)", flush=True)
        json.dump(out, open(args.out, "w"), ensure_ascii=False)

    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    per, tot = Counter(v["kind"] for v in out.values()), Counter(i["kind"] for i in items)
    print("\n=== KET QUA PSEUDO-GOLD (vLLM) ===")
    print(f"{'ho':10} {'tong':>6} {'9B lam dung':>13} {'ty le':>7}")
    for k in sorted(tot):
        print(f"{k:10} {tot[k]:6} {per[k]:13} {100*per[k]/tot[k]:6.1f}%")
    print(f"{'TONG':10} {len(items):6} {len(out):13} "
          f"{100*len(out)/max(len(items),1):6.1f}%")
    print(f"tong thoi gian {(time.time()-t0)/60:.1f} phut, "
          f"{n_tok} token sinh")
    print("Item 9B lam SAI giu gold tham chieu — khong mat mau, va khong day "
          "mapper tai tao suy luan sai.")

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
    print("GEN_PSEUDO_VLLM_EXIT", flush=True)


if __name__ == "__main__":   # BAT BUOC: LLM() o module level chet vi spawn
    main()
