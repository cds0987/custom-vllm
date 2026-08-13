"""Measure pure prefill throughput at specific prompt lengths, cache-proof.

Every other throughput tool in this repo (bench_load.py, bench_serving.py,
bench_skills.py) either fixes prompt content across repeats or cycles a
bounded pool of prompts -- both hit vLLM's automatic prefix caching after
enough requests (see STATUS.md "Bay do luong #5"), which quietly turns a
prefill measurement into a cache-hit measurement. This script exists because
none of them can isolate prefill at an exact token length while guaranteeing
every single request is a genuine cache miss.

Method:
  - Each prompt is freshly generated random lowercase "words" plus a random
    numeric nonce prefix, so no two requests -- across repeats, across
    lengths, across server restarts -- share a token prefix. Cache hits are
    structurally impossible, independent of whether prefix caching is on or
    off on the server (though --no-enable-prefix-caching is still
    recommended for clean A/B comparisons, per STATUS.md convention).
  - `max_tokens` is kept tiny (default 4) so wall-clock is dominated by
    prefill; the residual decode cost of a few tokens is noise at 4k+ prompt
    lengths.
  - A throwaway request (max_tokens=1) is fired before the timed repeats to
    pay for any first-call overhead (CUDA graph capture / Triton autotune on
    a fresh shape) without polluting the measurement.

FACTOR controls the words-per-token ratio used to hit a target token count.
Random lowercase strings tokenize far less efficiently than natural text
(no BPE merges to exploit), so naive "n_tokens words" overshoots by ~2.3x on
this tokenizer family -- 0.32 was calibrated empirically against Qwen3.5's
tokenizer to land within ~1% of the target. Override with env PB_FACTOR if
using a different tokenizer or seeing systematic drift.

Usage:
    VLLM_MODEL=/content/champion python scripts/prefill_bench.py
    VLLM_MODEL=/content/champion VLLM_URL=http://localhost:8000/v1/completions \\
        PB_FACTOR=0.32 python scripts/prefill_bench.py
"""

import os
import time
import random
import string
import requests

URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/completions")
MODEL = os.environ["VLLM_MODEL"]
FACTOR = float(os.environ.get("PB_FACTOR", "0.32"))


def rand_prompt(n_tokens: int) -> str:
    """Generate a prompt that is guaranteed to be a fresh cache miss."""
    words = []
    rng = random.Random(time.time_ns())
    for _ in range(int(n_tokens * FACTOR)):
        words.append("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8))))
    return "unique-nonce-%d " % rng.randint(0, 10**9) + " ".join(words)


def measure(n_tokens: int, max_tokens: int = 4) -> tuple[int, int, float]:
    prompt = rand_prompt(n_tokens)
    t0 = time.time()
    r = requests.post(
        URL,
        json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0},
        timeout=600,
    )
    if r.status_code != 200:
        print("ERR", r.status_code, r.text[:300])
    r.raise_for_status()
    dt = time.time() - t0
    usage = r.json()["usage"]
    return usage["prompt_tokens"], usage["completion_tokens"], dt


if __name__ == "__main__":
    for target in [4000, 16000, 30000]:
        measure(target, max_tokens=1)  # pay first-call overhead, discard
        results = [measure(target, max_tokens=4) for _ in range(3)]
        avg_ptoks = sum(p for p, _, _ in results) / len(results)
        avg_dt = sum(d for _, _, d in results) / len(results)
        print(
            f"target={target} actual_prompt_tokens~{avg_ptoks:.0f} "
            f"avg_wall={avg_dt:.3f}s prefill_tok_s~{avg_ptoks / avg_dt:.1f}"
        )
