"""gen_data -- data TRAIN da dang cho mapper (user 2026-08-28: "huan luyen
tren toan bo dataset, tong quat toan bo").

BOI CANH. Kiem `build_data()` cua e6v3_ce.py sau khi user hoi "co train
mapper cho math/reasoning ko?": train ~1330 item = bfcl ~655 + needle 430
+ ifstruct 135 + pbtable 110. Math 0, reasoning nhieu buoc 0, QA van xuoi 0.
Va MOI target train deu NGAN-TRICH (GEN_LEN bfcl 24, needle 16 token) ->
mapper CHUA TUNG bi ep giu cache song qua vai tram token sinh, dung cai ma
bbh/gsm8k/musr doi hoi. Do la 2 lech phai vá.

HAI RANG BUOC CUNG cua module nay:

(1) KHONG DUNG mau da do. Ca 3 loader trong ext_bench.py deu lay bang
    `.head(n)` (tien to co dinh) nen phan DUOI la sach:
        gsm8k : test da dung 200 -> train lay tu SPLIT `train` (7473, tach han)
        bbh   : test lay 7 dau moi tac vu (182) -> train lay tu hang thu 7
        musr  : test lay 66 dau moi split (198) -> train lay tu hang thu 66
    `assert_no_leak()` doi chieu bang chuoi prompt de chac chan, khong tin
    vao suy luan chi so.

(2) CHAM DIEM BANG CHINH GRADER CUA BENCHMARK. Moi item mang ca truong
    e6v3 (`kind`/`prompt`/`gold`) LAN truong ext_bench (`bench`/`expect`),
    nen val luc train goi thang `ext_bench.score_text` -- val va bao cao
    cuoi khong the troi khoi nhau (hoc phi 3 lan suyt bao so sai vi harness).

Gold cua gsm8k la LOI GIAI DAY DU (~100-200 token) -- day chinh la nguon
tin hieu "giu cache song qua sinh dai" ma data cu khong co.

Chay:  python -u gen_data.py --out /content/train_items.json --n-gsm8k 3000
"""

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

_HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# so mau MOI BEN da dung lam test niem phong (ext_bench: bbh n=200 chia deu
# 26 tac vu -> 7/tac vu; musr n=200 chia 3 split -> 66/split)
TEST_BBH_PER = 200 // 26
TEST_MUSR_PER = 200 // 3

# do dai sinh khi val/eval -- BAM THEO ext_bench.N_NEW de val giong that
GEN_LEN_NEW = {"gsm8k": 320, "bbh": 48, "musr": 24}
# tran token gold khi teacher-force CE
GMAX_NEW = {"gsm8k": 256, "bbh": 24, "musr": 8}


def gsm8k_train(n):
    """SPLIT `train` (7473) -- tach hoan toan khoi `test` da do 200 mau.
    gold = loi giai day du -> ep mapper giu cache qua ~150-250 token sinh."""
    eb = _load("ext_bench")
    df = eb._parquet("openai/gsm8k", "main/train/0000.parquet")
    items = []
    for i, row in df.head(n).iterrows():
        ans = str(row["answer"])
        body, gold_num = ans.split("####")
        body = re.sub(r"<<[^>]*>>", "", body).strip()      # bo annotation calc
        gold_num = gold_num.strip().replace(",", "")
        items.append({
            "kind": "gsm8k", "bench": "gsm8k", "id": f"gsm8k_tr/{i}",
            "prompt": (f"Solve step by step, then give the final numeric "
                       f"answer after 'Final Answer: '.\n\n"
                       f"Problem: {row['question']}\n\n"
                       f"<think>\n\n</think>\n\nSolution: "),
            "gold": f"{body}\nFinal Answer: {gold_num}",
            "expect": gold_num})
    return items


def bbh_tail(n):
    """Bo qua TEST_BBH_PER hang dau moi tac vu (chinh la tap test)."""
    eb = _load("ext_bench")
    per = max(1, n // len(eb.BBH_TASKS))
    items = []
    for task in eb.BBH_TASKS:
        try:
            df = eb._parquet("lukaemon/bbh", f"{task}/test/0000.parquet")
        except Exception as e:
            print(f"bbh skip {task}: {type(e).__name__}")
            continue
        tail = df.iloc[TEST_BBH_PER:TEST_BBH_PER + per]
        for i, row in tail.iterrows():
            tgt = str(row["target"]).strip()
            items.append({
                "kind": "bbh", "bench": "bbh", "id": f"{task}/{i}",
                "prompt": (f"{row['input']}\n\nAnswer concisely.\n"
                           f"<think>\n\n</think>\n\nAnswer: "),
                "gold": tgt, "expect": tgt})
    return items[:n]


def musr_tail(n):
    """Bo qua TEST_MUSR_PER hang dau moi split (chinh la tap test)."""
    import ast
    from datasets import load_dataset
    per = max(1, n // 3)
    items = []
    for split in ("murder_mysteries", "object_placements", "team_allocation"):
        ds = load_dataset("TAUR-Lab/MuSR", split=split)
        for i, ex in enumerate(ds):
            if i < TEST_MUSR_PER:
                continue
            if i >= TEST_MUSR_PER + per:
                break
            try:
                choices = json.loads(ex["choices"])
            except json.JSONDecodeError:
                choices = ast.literal_eval(ex["choices"])   # python-repr, khong JSON
            letters = [chr(65 + j) for j in range(len(choices))]
            opts = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
            gold = letters[ex["answer_index"]]
            items.append({
                "kind": "musr", "bench": "musr", "id": f"{split}/{i}",
                "prompt": (f"{ex['narrative']}\n\n{ex['question']}\n{opts}\n\n"
                           f"Answer with the letter of the correct choice.\n"
                           f"<think>\n\n</think>\n\nAnswer: "),
                "gold": gold, "expect": gold})
    return items[:n]


def suite_items(n, tok, ctxs=(1024, 2048, 4096)):
    """4 ho de cua suite_gen.py (rag/mid/math/swe) -- DA DUNG tu 2026-08-26
    nhung CHUA TUNG dung de TRAIN, chi de test."""
    sg = _load("suite_gen")
    out_path = "/tmp/_suite_train.json"
    sg.build_suite(n, list(ctxs), ["rag", "mid", "math", "swe"],
                   out_path, tok, seed=777)
    items = []
    for it in json.load(open(out_path)):
        fam = it["family"]
        items.append({
            "kind": f"suite_{fam}", "bench": f"suite_{fam}",
            "id": f"{fam}/{len(items)}", "prompt": it["prompt"],
            "gold": it["expect"][0], "expect": it["expect"],
            "ctx": it.get("ctx")})
    return items


def score_item(it, txt):
    """Cham diem: bbh/gsm8k/musr uy quyen cho ext_bench.score_text (CUNG
    grader voi bao cao cuoi); suite_* dung sg.score."""
    if it["bench"] in ("bbh", "gsm8k", "musr"):
        return _load("ext_bench").score_text(it, txt)
    if it["bench"].startswith("suite_"):
        fam = it["bench"].split("_", 1)[1]
        return _load("suite_gen").score(
            {"family": fam, "expect": it["expect"]}, txt)
    raise ValueError(it["bench"])


def assert_no_leak(items, sealed_path):
    """Doi chieu BANG CHUOI PROMPT, khong tin suy luan chi so."""
    if not Path(sealed_path).exists():
        print(f"CANH BAO: khong thay {sealed_path} -- BO QUA kiem ro ri. "
              "Phai chay lai khi co file test niem phong.")
        return
    sealed = {x["prompt"] for x in json.load(open(sealed_path))}
    bad = [it["id"] for it in items if it["prompt"] in sealed]
    assert not bad, f"RO RI {len(bad)} item trung tap test: {bad[:5]}"
    print(f"kiem ro ri: 0/{len(items)} item trung {len(sealed)} mau niem phong")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/train_items.json")
    ap.add_argument("--sealed", default="/content/ext_bench_items.json")
    ap.add_argument("--n-gsm8k", type=int, default=3000)
    ap.add_argument("--n-bbh", type=int, default=2600)
    ap.add_argument("--n-musr", type=int, default=500)
    ap.add_argument("--n-suite", type=int, default=800)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()

    items = []
    for nm, fn in (("gsm8k", lambda: gsm8k_train(args.n_gsm8k)),
                   ("bbh", lambda: bbh_tail(args.n_bbh)),
                   ("musr", lambda: musr_tail(args.n_musr))):
        try:
            xs = fn()
            print(f"{nm}: {len(xs)} item", flush=True)
            items += xs
        except Exception as e:
            print(f"{nm} THAT BAI: {type(e).__name__}: {e}", flush=True)

    if args.n_suite:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.src_model)
            xs = suite_items(args.n_suite, tok)
            print(f"suite: {len(xs)} item", flush=True)
            items += xs
        except Exception as e:
            print(f"suite THAT BAI: {type(e).__name__}: {e}", flush=True)

    assert_no_leak(items, args.sealed)
    rng = random.Random(4242)
    rng.shuffle(items)
    n_val = max(20, int(len(items) * args.val_frac))
    data = {"val": items[:n_val], "train": items[n_val:]}
    from collections import Counter
    for k, v in data.items():
        print(f"{k}: {len(v)} — {dict(Counter(x['kind'] for x in v))}")
    json.dump(data, open(args.out, "w"), indent=1)
    print(f"da ghi {args.out}")
    print("GEN_DATA_EXIT")


if __name__ == "__main__":
    main()
