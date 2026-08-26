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
            # C2b-5: target TOKEN that (calib 2 vong) + pad NGAU NHIEN
            # khong lap (nhieu C2b-4: pad chu ky x0..x96 tu du degeneration)
            nw = ctx_t // 3
            words = [f"w{rng.randint(0, 9999)}" for _ in range(nw)]
            for _ in range(3):
                T = nt(mk(words, []))
                if abs(T - ctx_t) < ctx_t * 0.04:
                    break
                nw = max(20, int(nw * ctx_t / T))
                words = [f"w{rng.randint(0, 9999)}" for _ in range(nw)]
            pad = []
            for _ in range(40):
                txt = mk(words, pad)
                T = nt(txt)
                need = (5 - T) % 1056
                if need <= 3 or need >= 1053:
                    break
                pad += [f"z{rng.randint(0, 9999)}" for _ in range(max(1, need // 3))]
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


def run_pass(tag):
    mid = model_id()
    prompts = json.load(open(PROMPTS_F))
    out = []
    for i, p in enumerate(prompts):
        t0 = time.time()
        r = http("/completions", {"model": mid, "prompt": p["prompt"],
                                  "max_tokens": 24, "temperature": 0,
                                  "logprobs": 1})
        dt = time.time() - t0
        ch = r["choices"][0]
        toks = ch.get("logprobs", {}).get("tokens", [])
        lps = ch.get("logprobs", {}).get("token_logprobs", [])
        txt = ch["text"]
        hit = int(p["code"] in "".join(c for c in txt if c.isdigit()))
        out.append({"i": i, "ctx": p["ctx"], "hit": hit, "lat": round(dt, 3),
                    "text": txt[:80], "tokens": toks, "logprobs": lps})
        print(f"{tag} {i} ctx{p['ctx']} hit={hit} lat={dt:.2f}s "
              f"text={txt[:40]!r}")
    # TTFT rieng: prompt 30K dau tien (da prefill o vong tren -> voi
    # prefix-cache bat thi lat nay la warm; do them mot prompt 30K MOI
    # o che do cold-cho-9B (chua tung qua 9B nhung DA qua 4B o --produce)
    res = {"runs": out, "ttft_30k_repeat": ttft_stream(mid, prompts[-1]["prompt"])}
    path = f"/content/logs/c2b_{tag}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{tag} saved -> {path}")


def produce():
    mid = model_id()
    prompts = json.load(open(PROMPTS_F))
    for i, p in enumerate(prompts):
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
    ap.add_argument("mode", choices=["gen", "gen-aligned", "produce",
                                     "baseline", "cross", "compare"])
    args = ap.parse_args()
    if args.mode == "gen":
        build_prompts()
    elif args.mode == "gen-aligned":
        build_prompts_aligned()
    elif args.mode == "produce":
        produce()
    elif args.mode == "compare":
        compare()
    else:
        run_pass(args.mode)
