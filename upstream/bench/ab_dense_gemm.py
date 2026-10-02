"""A/B client for VLLM_GGUF_DENSE_GEMM_MIN_ROWS against a live vLLM server.

Self-contained on purpose (stdlib + requests): the numbers go into an upstream
PR, so a maintainer must be able to rerun this without the rest of our repo.

  prefill: one fresh random prompt per request (no prefix-cache hit possible),
           max_tokens=1 -> wall time is prefill. tok/s = prompt_tokens / wall.
  decode : closed loop, N workers, short prompt, fixed-length output
           (min_tokens = max_tokens, ignore_eos). tok/s = all tokens / wall.
  sample : greedy completions kept verbatim, to compare the two arms by eye.

    python ab_dense_gemm.py --model <served name> --label thr0 --out r.json
"""
import argparse
import json
import random
import string
import threading
import time

import requests

SAMPLE_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "List three prime numbers greater than 10:",
    "Water boils at 100 degrees Celsius, which in Fahrenheit is",
]


def rand_prompt(n_words: int, rng: random.Random) -> str:
    words = ("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
             for _ in range(n_words))
    return f"{rng.randint(0, 10**9)} " + " ".join(words)


def post(url, model, prompt, max_tokens, fixed_len=False, timeout=600):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0}
    if fixed_len:
        body |= {"min_tokens": max_tokens, "ignore_eos": True}
    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=timeout)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    return dt, j["usage"], j["choices"][0]["text"]


def bench_prefill(url, model, target_tokens, repeats, words_per_token, rng):
    post(url, model, rand_prompt(int(target_tokens * words_per_token), rng), 1)
    rows = []
    for _ in range(repeats):
        dt, usage, _ = post(
            url, model, rand_prompt(int(target_tokens * words_per_token), rng), 1)
        rows.append({"prompt_tokens": usage["prompt_tokens"], "sec": dt,
                     "tok_s": usage["prompt_tokens"] / dt})
    rates = sorted(r["tok_s"] for r in rows)
    return {"target_tokens": target_tokens, "median_tok_s": rates[len(rates) // 2],
            "min_tok_s": rates[0], "max_tok_s": rates[-1], "runs": rows}


def bench_decode(url, model, conc, per_worker, max_tokens):
    done, errors, lock = [], [], threading.Lock()

    def worker(i):
        for k in range(per_worker):
            try:
                _, usage, text = post(
                    url, model, f"Request {i}-{k}. Write a long story about a river.",
                    max_tokens, fixed_len=True)
                with lock:
                    done.append((usage["completion_tokens"], text))
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(repr(e))

    post(url, model, "warmup", 8)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    toks = sum(n for n, _ in done)
    # the failure that has bitten us before: one token repeated forever
    degenerate = sum(1 for _, s in done if len(set(s.split())) <= 2)
    return {"conc": conc, "requests": len(done), "errors": len(errors),
            "degenerate": degenerate, "tokens": toks, "sec": wall,
            "tok_s": toks / wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefill-tokens", type=int, nargs="+", default=[2048, 8192, 12000])
    ap.add_argument("--prefill-repeats", type=int, default=5)
    ap.add_argument("--words-per-token", type=float, default=0.32)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--per-worker", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    rng = random.Random(1234)
    res = {"label": a.label, "model": a.model, "prefill": [], "decode": [], "samples": []}
    for p in SAMPLE_PROMPTS:
        res["samples"].append({"prompt": p, "text": post(a.url, a.model, p, 48)[2]})
    for n in a.prefill_tokens:
        r = bench_prefill(a.url, a.model, n, a.prefill_repeats, a.words_per_token, rng)
        print(f"[{a.label}] prefill ~{r['runs'][0]['prompt_tokens']} tok: "
              f"{r['median_tok_s']:.0f} tok/s (min {r['min_tok_s']:.0f}, max {r['max_tok_s']:.0f})",
              flush=True)
        res["prefill"].append(r)
    for c in a.conc:
        r = bench_decode(a.url, a.model, c, a.per_worker, a.max_tokens)
        print(f"[{a.label}] decode conc={c}: {r['tok_s']:.1f} tok/s "
              f"({r['requests']} ok, {r['errors']} err, {r['degenerate']} degenerate)", flush=True)
        res["decode"].append(r)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[{a.label}] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
