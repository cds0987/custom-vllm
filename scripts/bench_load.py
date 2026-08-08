"""Concurrency sweep against an OpenAI-compatible vLLM endpoint.

Reports what actually matters for serving: sustained output throughput and the
latency spread as concurrency rises. Run it after the server is warm — the
first request on a cold server pays for Triton autotune, which would otherwise
dominate the numbers.

Usage:
    VLLM_MODEL=repo:QUANT python scripts/bench_load.py [concurrency ...]
"""

import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ["VLLM_MODEL"]
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "64"))
PROMPT = os.environ.get("BENCH_PROMPT", "Liệt kê 10 thành phố lớn của Việt Nam.")


def one(i: int) -> tuple[float, int]:
    t0 = time.time()
    r = requests.post(
        URL,
        json={
            "model": MODEL,
            # vary the prompt so prefix caching can't serve everything from cache
            "messages": [{"role": "user", "content": f"{PROMPT} (#{i})"}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        },
        timeout=1800,
    )
    r.raise_for_status()
    return time.time() - t0, r.json()["usage"]["completion_tokens"]


def warmup() -> None:
    print("warmup (pays the Triton autotune cost) ...", flush=True)
    t0 = time.time()
    one(-1)
    print(f"  done in {time.time() - t0:.1f}s\n", flush=True)


def bench(concurrency: int) -> None:
    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(one, range(concurrency)))
    wall = time.time() - t0
    tokens = sum(n for _, n in results)
    lat = sorted(l for l, _ in results)
    print(
        f"conc={concurrency:3d}  wall={wall:7.1f}s  out={tokens:5d} tok  "
        f"throughput={tokens / wall:7.2f} tok/s  "
        f"lat p50={statistics.median(lat):6.1f}s  p95={lat[int(0.95 * (len(lat) - 1))]:6.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    levels = [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8, 16, 32]
    print(f"model={MODEL}  max_tokens={MAX_TOKENS}\n")
    warmup()
    for c in levels:
        bench(c)
