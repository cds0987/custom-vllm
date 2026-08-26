"""C2b — 3 cong dung dan cho cross-model KV qua LMCache (Phase C).

Kien truc 1-GPU tuan tu: [4B --produce] -> stop -> [9B baseline khong
connector] -> stop -> [9B --cross co connector, doc trang 4B tu lmcache
server dang song] -> so sanh.

Gates (thu tu cung, luat error-placement):
  1. logprob-parity: greedy 24 token + logprobs, cross vs self-baseline
  2. needle functional: ma 6 chu so trong context 1.5K/8K/30K
  3. TTFT: cold (baseline, server moi boot) vs cross-warm (trang tu 4B)

Prompt filler tong hop deterministic (khong can dataset/FineWeb — moi
prompt seed rieng, khong dung nhau de prefix-cache noi bo khong nhieu).
"""

import argparse
import json
import time
import urllib.request

API = "http://127.0.0.1:8000/v1"
PROMPTS_F = "/content/c2b_prompts.json"


def http(path, body=None, timeout=600):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_id():
    return http("/models")["data"][0]["id"]


def build_prompts():
    """6 needle prompt (2 moi co 1.5K/8K/30K tu-word ~ token) — filler
    danh so deterministic, needle giua, hoi cuoi."""
    import random
    prompts = []
    # (ctx_label_token, so_WORD) — moi "wNNNN" ~2.5 token (bai hoc 400: 30K
    # word ~75K token > mml 65536). Word count = token_target / 2.5.
    for ctx, nw, n in [(1500, 600, 2), (8000, 3200, 2), (30000, 12000, 2)]:
        for j in range(n):
            rng = random.Random(1000 * ctx + j)
            code = "".join(rng.choice("0123456789") for _ in range(6))
            name = f"PRJ{rng.randint(100, 999)}"
            words = [f"w{rng.randint(0, 9999)}" for _ in range(nw)]
            half = len(words) // 2
            txt = (" ".join(words[:half])
                   + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
                   + " ".join(words[half:])
                   + f"\nQuestion: What is the secret code for project {name}?"
                   + "\nAnswer: The secret code is ")
            prompts.append({"ctx": ctx, "code": code, "prompt": txt})
    with open(PROMPTS_F, "w") as fh:
        json.dump(prompts, fh)
    print(f"gen: {len(prompts)} prompts -> {PROMPTS_F}")


def build_prompts_n(n, ctx_t):
    """C2b-N (user 2026-08-26: '3/4 qua it samples, thu 2-300'): N prompt
    aligned mot do dai — thong ke that."""
    import random
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/content/models/frame4b")
    m = max(1, round((ctx_t - 5) / 1056))
    tstar = m * 1056 + 5
    prompts = []
    for j in range(n):
        rng = random.Random(7000 * ctx_t + j)
        code = "".join(rng.choice("0123456789") for _ in range(6))
        name = f"PRJ{rng.randint(100, 999)}"

        def mk(ws, pad):
            half = len(ws) // 2
            return (" ".join(ws[:half])
                    + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
                    + " ".join(ws[half:])
                    + (" " + " ".join(pad) if pad else "")
                    + f"\nQuestion: What is the secret code for project {name}?"
                    + "\nAnswer: The secret code is ")

        def nt(txt):
            return len(tok(txt)["input_ids"])
        bank = [f"w{rng.randint(0, 9999)}" for _ in range(ctx_t)]
        lo, hi = 0, len(bank)
        while lo < hi:
            mid = (lo + hi) // 2
            if nt(mk(bank[:mid], [])) < tstar:
                lo = mid + 1
            else:
                hi = mid
        k = max(2, lo - 1)
        pad = []
        txt = mk(bank[:k], pad)
        T = nt(txt)
        for _ in range(12):
            if T >= tstar - 3:
                break
            pad.append("a")
            txt = mk(bank[:k], pad)
            T = nt(txt)
        prompts.append({"ctx": ctx_t, "code": code, "prompt": txt,
                        "T": T, "rem": T % 1056})
        if j % 40 == 0:
            print(f"gen-n {j}/{n} T={T} rem={T % 1056}")
    with open(PROMPTS_F, "w") as fh:
        json.dump(prompts, fh)
    print(f"gen-n: {n} prompts ctx{ctx_t} -> {PROMPTS_F}")


def build_prompts_sem(n, ctx_t):
    """C2c — scope NGU NGHIA tren serving (protocol E2, filler wikitext THAT):
    fact tu nhien giau giua van ban thuc, cau hoi PARAPHRASE cuoi, cham diem
    KEYWORD (khong doi khop nguyen van chuoi so — tranh dung mep vuc bien mong
    cua exact-retrieval). Van align T ≡ ~5 (mod 1056) de lmcache hit."""
    import os
    import random
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok_src = "/content/models/frame4b"
    if not os.path.isdir(tok_src):
        tok_src = "Qwen/Qwen3.5-4B"
    tok = AutoTokenizer.from_pretrained(tok_src)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    corpus = " ".join(t.strip() for t in ds["text"] if t.strip()).split()
    subs = ["Doctor Vance", "Professor Ito", "Captain Reyes", "Curator Lam",
            "Engineer Bok", "Archivist Pena", "Colonel Draye", "Sister Maud"]
    objs = ["telescope", "manuscript", "compass", "violin", "microscope",
            "chalice", "ledger", "sextant", "tapestry", "astrolabe"]
    cols = ["silver", "crimson", "ivory", "emerald", "bronze", "cobalt"]
    plcs = ["attic", "basement", "greenhouse", "chapel", "lighthouse",
            "archive", "cellar", "observatory", "stables", "pantry"]
    m = max(1, round((ctx_t - 5) / 1056))
    tstar = m * 1056 + 5
    prompts = []
    stride = max(1, (len(corpus) - ctx_t - 600) // max(n, 1))
    for j in range(n):
        rng = random.Random(9100 * ctx_t + j)
        s, o = rng.choice(subs), rng.choice(objs)
        c, pl = rng.choice(cols), rng.choice(plcs)
        fact = (f"For many years {s} kept the {c} {o} hidden away in the "
                f"{pl}, and almost nobody ever knew about it.")
        if j % 2 == 0:
            q = f"Where did {s} hide the {o}?"
            ans, kw = "It was hidden in the ", pl
        else:
            q = f"What color was the {o} that {s} hid?"
            ans, kw = "The color of it was ", c
        start = (j * stride + rng.randint(0, 500)) % max(1, len(corpus) - ctx_t)
        bank = corpus[start:start + ctx_t]

        def mk(ws, pad):
            half = len(ws) // 2
            return (" ".join(ws[:half]) + "\n" + fact + "\n"
                    + " ".join(ws[half:])
                    + (" " + " ".join(pad) if pad else "")
                    + f"\nQuestion: {q}\nAnswer: {ans}")

        def nt(txt):
            return len(tok(txt)["input_ids"])
        lo, hi = 0, len(bank)
        while lo < hi:
            mid = (lo + hi) // 2
            if nt(mk(bank[:mid], [])) < tstar:
                lo = mid + 1
            else:
                hi = mid
        k = max(2, lo - 1)
        pad = []
        txt = mk(bank[:k], pad)
        T = nt(txt)
        for _ in range(12):
            if T >= tstar - 3:
                break
            pad.append("a")
            txt = mk(bank[:k], pad)
            T = nt(txt)
        prompts.append({"ctx": ctx_t, "kw": kw, "prompt": txt,
                        "T": T, "rem": T % 1056})
        if j % 10 == 0:
            print(f"gen-sem {j}/{n} T={T} rem={T % 1056} kw={kw}")
    with open(PROMPTS_F, "w") as fh:
        json.dump(prompts, fh)
    print(f"gen-sem: {n} prompts ctx{ctx_t} -> {PROMPTS_F}")


def agg(kinds=("baseline", "cross")):
    """Gom tat ca c2b_<kind>*.json -> ty le."""
    import glob
    for kind in kinds:
        hits = tot = 0
        lats = []
        for f in sorted(glob.glob(f"/content/logs/c2b_{kind}*.json")):
            d = json.load(open(f))
            runs = d["runs"] if isinstance(d, dict) else d
            for r in runs:
                tot += 1
                hits += r["hit"]
                lats.append(r["lat"])
        if tot:
            lats.sort()
            print(f"AGG {kind}: {hits}/{tot} = {hits/tot:.1%} | "
                  f"lat p50 {lats[len(lats)//2]:.2f}s")
    print("AGG_DONE")


def build_prompts_aligned():
    """C2b-4: prompt co T ≡ ~5 (mod 1056) — phan du re-prefill cua 9B chi
    con vai token (dung lieu WARM_P transformers da chung minh vo hai).
    Phan xu gia thuyet suffix-re-prefill."""
    import random
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/content/models/frame4b")
    prompts = []
    for ctx_t, n in [(8000, 2), (30000, 2)]:
        for j in range(n):
            rng = random.Random(3000 * ctx_t + j)
            code = "".join(rng.choice("0123456789") for _ in range(6))
            name = f"PRJ{rng.randint(100, 999)}"

            def mk(ws, pad):
                half = len(ws) // 2
                return (" ".join(ws[:half])
                        + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
                        + " ".join(ws[half:])
                        + (" " + " ".join(pad) if pad else "")
                        + f"\nQuestion: What is the secret code for project {name}?"
                        + "\nAnswer: The secret code is ")

            def nt(txt):
                return len(tok(txt)["input_ids"])
            # C2b-5 v2: BINARY SEARCH so tu theo tokenizer that (don dieu)
            # — vong pad so hoc cu overshoot roi quan vong mod 1056 (T no
            # len 49K/71K, rem ket 410). Tstar = m*1056 + 5 gan ctx_t.
            m = max(1, round((ctx_t - 5) / 1056))
            tstar = m * 1056 + 5
            bank = [f"w{rng.randint(0, 9999)}" for _ in range(ctx_t)]
            lo, hi = 0, len(bank)
            while lo < hi:
                mid = (lo + hi) // 2
                if nt(mk(bank[:mid], [])) < tstar:
                    lo = mid + 1
                else:
                    hi = mid
            k = max(2, lo - 1)
            pad = []
            txt = mk(bank[:k], pad)
            T = nt(txt)
            for _ in range(12):   # tinh chinh tung token don " a"
                if T >= tstar - 3:
                    break
                pad.append("a")
                txt = mk(bank[:k], pad)
                T = nt(txt)
            prompts.append({"ctx": ctx_t, "code": code, "prompt": txt,
                            "T": T, "rem": T % 1056})
            print(f"aligned ctx{ctx_t} j{j}: T={T} rem={T % 1056}")
    with open(PROMPTS_F, "w") as fh:
        json.dump(prompts, fh)
    print(f"gen-aligned: {len(prompts)} prompts -> {PROMPTS_F}")


def ttft_stream(mid, prompt):
    """Do TTFT bang stream SSE — timestamp chunk token dau tien."""
    body = json.dumps({"model": mid, "prompt": prompt, "max_tokens": 8,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(API + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b"text" in line:
                return time.time() - t0
    return -1.0


def _sliced(sl):
    prompts = json.load(open(PROMPTS_F))
    if sl:
        a, b = (int(x) for x in sl.split(":"))
        return list(enumerate(prompts))[a:b], f"_{a}_{b}"
    return list(enumerate(prompts)), ""


def run_pass(tag, sl=""):
    mid = model_id()
    pairs, suffix = _sliced(sl)
    out = []
    for i, p in pairs:
        t0 = time.time()
        r = http("/completions", {"model": mid, "prompt": p["prompt"],
                                  "max_tokens": 24, "temperature": 0,
                                  "logprobs": 1})
        dt = time.time() - t0
        ch = r["choices"][0]
        toks = ch.get("logprobs", {}).get("tokens", [])
        lps = ch.get("logprobs", {}).get("token_logprobs", [])
        txt = ch["text"]
        if "kw" in p:                     # C2c sem: cham keyword
            hit = int(p["kw"].lower() in txt.lower())
        else:                             # needle: cham chuoi so
            hit = int(p["code"] in "".join(c for c in txt if c.isdigit()))
        out.append({"i": i, "ctx": p["ctx"], "hit": hit, "lat": round(dt, 3),
                    "text": txt[:80], "tokens": toks, "logprobs": lps})
        print(f"{tag} {i} ctx{p['ctx']} hit={hit} lat={dt:.2f}s "
              f"text={txt[:40]!r}")
    res = {"runs": out, "ttft_30k_repeat": ttft_stream(mid, pairs[-1][1]["prompt"])}
    path = f"/content/logs/c2b_{tag}{suffix}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{tag} saved -> {path}")


def produce(sl=""):
    mid = model_id()
    pairs, _ = _sliced(sl)
    for i, p in pairs:
        t0 = time.time()
        http("/completions", {"model": mid, "prompt": p["prompt"],
                              "max_tokens": 1, "temperature": 0})
        print(f"produce {i} ctx{p['ctx']} {time.time()-t0:.2f}s")
    print("PRODUCE_DONE")


def compare():
    a = json.load(open("/content/logs/c2b_baseline.json"))
    b = json.load(open("/content/logs/c2b_cross.json"))
    n_tok = n_match = 0
    dlp = []
    for ra, rb in zip(a["runs"], b["runs"]):
        for t1, t2, l1, l2 in zip(ra["tokens"], rb["tokens"],
                                  ra["logprobs"], rb["logprobs"]):
            n_tok += 1
            if t1 == t2:
                n_match += 1
                if l1 is not None and l2 is not None:
                    dlp.append(abs(l1 - l2))
    needle_a = sum(r["hit"] for r in a["runs"])
    needle_b = sum(r["hit"] for r in b["runs"])
    lat_a = {r["ctx"]: r["lat"] for r in a["runs"]}
    lat_b = {r["ctx"]: r["lat"] for r in b["runs"]}
    rep = {
        "gate1_token_match": f"{n_match}/{n_tok}",
        "gate1_mean_abs_dlogprob": round(sum(dlp) / max(len(dlp), 1), 4),
        "gate2_needle_self": f"{needle_a}/{len(a['runs'])}",
        "gate2_needle_cross": f"{needle_b}/{len(b['runs'])}",
        "gate3_lat_self_first": lat_a, "gate3_lat_cross_first": lat_b,
        "ttft30k_self_repeat": a.get("ttft_30k_repeat"),
        "ttft30k_cross_repeat": b.get("ttft_30k_repeat"),
    }
    print("===== C2B KET QUA =====")
    print(json.dumps(rep, indent=1))
    with open("/content/logs/c2b_report.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    print("C2B_COMPARE_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "gen-aligned", "gen-n", "gen-sem",
                                     "produce", "baseline", "cross", "compare",
                                     "agg", "sembase", "semcross", "agg-sem"])
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--slice", default="")
    args = ap.parse_args()
    if args.mode == "gen":
        build_prompts()
    elif args.mode == "gen-aligned":
        build_prompts_aligned()
    elif args.mode == "gen-n":
        build_prompts_n(args.n, args.ctx)
    elif args.mode == "gen-sem":
        build_prompts_sem(args.n, args.ctx)
    elif args.mode == "produce":
        produce(args.slice)
    elif args.mode == "compare":
        compare()
    elif args.mode == "agg":
        agg()
    elif args.mode == "agg-sem":
        agg(("sembase", "semcross"))
    else:
        run_pass(args.mode, args.slice)
