"""Open-loop, rate-based serving benchmark for an OpenAI-compatible vLLM endpoint.

Supersedes nothing — it complements bench_swebench.py's `concurrent` mode,
which is a *closed-loop* sweep: N workers each fire a request, wait for the
reply, then fire the next one. Closed-loop is blind to saturation, because
the moment the server slows down, the offered load automatically shrinks
(workers are stuck waiting instead of sending). A server that's falling over
can look "fine" under closed-loop load simply because nobody's arrival rate
went up to reveal it.

This script instead fixes the *arrival* rate (queries/sec) and fires
requests on that schedule regardless of whether earlier ones have finished
— exactly what a production traffic source does. Interarrival gaps are
drawn from an exponential distribution (a Poisson arrival process), not a
fixed metronome: real traffic is bursty, and a uniform interval
systematically under-counts queueing/tail latency because it never produces
the tight clusters of near-simultaneous arrivals that actually cause a
server to fall behind. Sweeping the rate and watching achieved QPS,
concurrency, and TTFT diverge from what was offered is how you find the
knee in the latency curve — the load level beyond which the server can no
longer keep up.

Dataset: zai-org/LongAlign-10k (long real documents, ~60k chars/row) so
prefill is never trivial. Prompts are truncated to a token budget (default
~12000, leaving headroom under a 16K context) and cached locally so repeated
runs compare apples to apples instead of drifting with the HF dataset.

Usage:
    VLLM_MODEL=repo:QUANT python scripts/bench_serving.py run --rates 3 5 7 9 12 30
    VLLM_MODEL=repo:QUANT python scripts/bench_serving.py run --rates 10 --label q4_k_m
    python scripts/bench_serving.py compare out/bench_serving_summary_*.json

Every response is checked for the failure mode that has actually bitten this
project: a fast server streaming back one token repeated forever. A
fast-but-broken config must never look like a win, so degenerate output is
tracked and surfaced right next to the throughput numbers.
"""

import argparse
import glob
import itertools
import json
import os
import random
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache"
OUT_DIR = REPO_ROOT / "out"

# Mutated by main() from CLI args / env; module-level like the other bench
# scripts so the request helpers below don't need these threaded through.
MODEL = os.environ.get("VLLM_MODEL", "")
URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/") + "/v1/chat/completions"
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "120"))  # per-chunk stall ceiling, not total request time


# --------------------------------------------------------------------------
# Dataset: zai-org/LongAlign-10k, truncated + cached
# --------------------------------------------------------------------------

def _extract_first_user_message(messages: list) -> str:
    """LongAlign `messages` is a list of {"role", "content"} dicts. Content is
    normally a plain string, but be defensive about list-of-blocks content
    (some HF chat datasets use that shape) rather than crashing the load."""
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
    return ""


def _truncate_chars(text: str, token_budget: int) -> str:
    # ~4 chars/token is a crude but standard estimate; good enough to keep
    # prompts in the target band when no real tokenizer is available.
    chars_per_token = 4
    limit = token_budget * chars_per_token
    return text[:limit]


def _make_char_truncator(token_budget: int):
    return lambda text: _truncate_chars(text, token_budget)


def _make_tokenizer_truncator(tokenizer_name: str, token_budget: int):
    """Only called when --tokenizer is passed; falls back to char truncation
    (with a warning) if transformers isn't installed rather than hard-failing
    — this benchmark needs to run in GPU-less, vllm-less environments too."""
    try:
        from transformers import AutoTokenizer  # imported lazily: heavy, optional
    except ImportError:
        print(
            f"WARNING: --tokenizer {tokenizer_name} given but `transformers` isn't "
            "installed; falling back to char-based truncation.",
            file=sys.stderr,
        )
        return _make_char_truncator(token_budget)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    def truncate(text: str) -> str:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= token_budget:
            return text
        return tok.decode(ids[:token_budget], skip_special_tokens=True)

    return truncate


def load_prompts(num_prompts: int, token_budget: int, tokenizer_name: str | None) -> list:
    """Load LongAlign-10k (train split), build one prompt per row from the
    first user message, truncate to `token_budget`, and cache the *result*
    (not the raw dataset) to a local JSONL.

    Caching the truncated text — keyed by budget + tokenizer, so a different
    config can't silently reuse a stale cache — means reproducibility beats
    freshness here: two runs of the same config always see byte-identical
    prompts, which is the whole point of a comparison benchmark.
    """
    tok_tag = tokenizer_name.replace("/", "_") if tokenizer_name else "chars4"
    cache_file = CACHE_DIR / f"longalign_prompts_budget{token_budget}_{tok_tag}_n{num_prompts}.jsonl"

    if not cache_file.exists():
        from datasets import load_dataset  # imported lazily: only needed on first run

        print(f"Downloading zai-org/LongAlign-10k (train split) -> {cache_file} ...", file=sys.stderr)
        ds = load_dataset("zai-org/LongAlign-10k", split="train")

        truncate = (
            _make_tokenizer_truncator(tokenizer_name, token_budget)
            if tokenizer_name
            else _make_char_truncator(token_budget)
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        n_written = 0
        with cache_file.open("w", encoding="utf-8") as f:
            for row in ds:
                raw = _extract_first_user_message(row.get("messages") or [])
                if not raw:
                    continue
                prompt = truncate(raw)
                f.write(json.dumps({"prompt": prompt, "orig_length": row.get("length")}) + "\n")
                n_written += 1
                if n_written >= num_prompts:
                    break

    rows = []
    with cache_file.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["prompt"])
    return rows


# --------------------------------------------------------------------------
# SSE streaming client (same wire parsing as bench_swebench.py)
# --------------------------------------------------------------------------

def stream_chat_completion(prompt: str, max_tokens: int):
    """POST /v1/chat/completions with stream=True and yield (t_recv, delta_text, usage).

    Parses SSE line-by-line (not on blank-line event separators) because
    OpenAI-style chat streams emit one `data:` line per event and some
    proxies don't reliably send the trailing blank line. Buffers raw bytes
    across chunk boundaries so a `data: ...` line split across two TCP reads
    is never mis-parsed as two lines.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    buf = b""
    with requests.post(
        URL, json=payload, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    ) as r:
        r.raise_for_status()
        # chunk_size=1, not None: for a chunked-transfer response, requests/
        # urllib3 will coalesce and only yield once the *entire* body has
        # arrived if you let it pick the read size, silently destroying the
        # per-token timing (TTFT/ITL) this whole benchmark depends on. This
        # detail is load-bearing — without it every latency number is wrong.
        for chunk in r.iter_content(chunk_size=1):
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                t_recv = time.monotonic()
                usage = obj.get("usage")
                delta_text = ""
                choices = obj.get("choices") or []
                if choices:
                    delta_text = choices[0].get("delta", {}).get("content") or ""
                yield t_recv, delta_text, usage


def percentile(values: list, p: float):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def is_degenerate(text: str) -> bool:
    """Flag empty output or output collapsed onto one repeated token — the
    exact failure mode that has bitten this project: a broken GGUF/quant
    patch that streams tokens fast but produces garbage. A speed number
    without this check would make that look like a win."""
    text = text.strip()
    words = text.split()
    if len(words) < 5:
        return len(text) == 0
    most_common = max(Counter(words).values())
    return most_common / len(words) > 0.6


# --------------------------------------------------------------------------
# Admission control: bounds in-flight requests so a saturated server can't
# exhaust the client (memory, sockets, threads).
# --------------------------------------------------------------------------

class InFlightGate:
    def __init__(self, cap: int):
        self.cap = cap
        self._lock = threading.Lock()
        self._count = 0

    def try_acquire(self) -> bool:
        with self._lock:
            if self._count >= self.cap:
                return False
            self._count += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._count -= 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


# --------------------------------------------------------------------------
# Per-request runner (fired on its own daemon thread by the dispatcher)
# --------------------------------------------------------------------------

def run_one_request(idx: int, prompt: str, max_tokens: int, t_arrival_mono: float,
                     wall_offset: float, gate: InFlightGate, results: list, results_lock: threading.Lock) -> None:
    """Runs in its own thread; appends its record to `results` when done.

    `wall_offset` converts monotonic timestamps (correct for measuring
    durations) to epoch/wall-clock ones (what a JSONL time series needs to
    be replottable and comparable against other runs) via a single fixed
    offset captured once at the start of the rate level, so the conversion
    can't be skewed by a mid-run system clock adjustment.
    """
    record = {
        "idx": idx,
        "ok": False,
        "dropped": False,
        "error": None,
        "t_arrival": t_arrival_mono + wall_offset,
    }
    token_times = []
    text_parts = []
    last_usage = None
    try:
        for t_recv, delta_text, usage in stream_chat_completion(prompt, max_tokens):
            if usage:
                last_usage = usage
            if delta_text:
                token_times.append(t_recv)
                text_parts.append(delta_text)
        t_end = time.monotonic()
        text = "".join(text_parts)

        completion_tokens = last_usage.get("completion_tokens") if last_usage else None
        if completion_tokens is None:
            completion_tokens = len(token_times)  # proxy: one SSE content-delta ~= one token

        itls = [b - a for a, b in zip(token_times, token_times[1:])]

        record.update(
            ok=True,
            t_arrival=t_arrival_mono + wall_offset,
            t_first_token=(token_times[0] + wall_offset) if token_times else None,
            t_end=t_end + wall_offset,
            ttft_s=(token_times[0] - t_arrival_mono) if token_times else None,
            e2e_s=t_end - t_arrival_mono,
            queue_plus_prefill_s=(token_times[0] - t_arrival_mono) if token_times else None,
            prompt_tokens=last_usage.get("prompt_tokens") if last_usage else None,
            completion_tokens=completion_tokens,
            itls=itls,
            degenerate=is_degenerate(text),
            text_preview=text[:200],
        )
    except requests.exceptions.RequestException as e:
        record["t_end"] = time.monotonic() + wall_offset
        record["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # never let one bad request crash the whole sweep
        record["t_end"] = time.monotonic() + wall_offset
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        gate.release()
        with results_lock:
            results.append(record)


# --------------------------------------------------------------------------
# Concurrency / saturation analysis, derived purely from timestamps
# --------------------------------------------------------------------------

def concurrency_series(records: list):
    """Sweep-line over [t_arrival, t_end] intervals of every dispatched
    (non-dropped) request -> time-weighted mean concurrency and max
    concurrency. This is what "in-flight" means: it needs no live counter,
    just the timestamps every record already carries."""
    events = []
    for r in records:
        if r.get("dropped"):
            continue
        t0 = r.get("t_arrival")
        t1 = r.get("t_end")
        if t0 is None or t1 is None:
            continue
        events.append((t0, 1))
        events.append((t1, -1))
    if not events:
        return 0.0, 0
    events.sort()
    cur = 0
    max_c = 0
    area = 0.0
    span = 0.0
    last_t = None
    for t, delta in events:
        if last_t is not None and t > last_t:
            area += cur * (t - last_t)
            span += t - last_t
        cur += delta
        max_c = max(max_c, cur)
        last_t = t
    mean_c = area / span if span > 0 else float(max_c)
    return mean_c, max_c


def queueing_grows(records: list) -> bool:
    """Split the run into thirds by arrival order and check whether mean
    in-flight concurrency is monotonically increasing (with a 10% margin per
    step, to avoid flagging on noise). A server that's keeping up shows flat
    or oscillating concurrency; one that's falling behind shows a queue that
    only ever grows."""
    ok = [r for r in records if r.get("ok") or (not r.get("dropped") and r.get("t_end") is not None)]
    ok = [r for r in ok if r.get("t_arrival") is not None]
    if len(ok) < 9:
        return False
    ok.sort(key=lambda r: r["t_arrival"])
    n = len(ok)
    thirds = [ok[: n // 3], ok[n // 3 : 2 * n // 3], ok[2 * n // 3 :]]
    means = []
    for chunk in thirds:
        m, _ = concurrency_series(chunk)
        means.append(m)
    return means[0] > 0 and means[1] > means[0] * 1.1 and means[2] > means[1] * 1.1


# --------------------------------------------------------------------------
# Open-loop dispatcher for one rate level
# --------------------------------------------------------------------------

def run_rate_level(rate_qps: float, prompt_cycle, max_tokens: int, duration_s: float,
                    count: int, max_inflight: int, drain_timeout: float, seed: int) -> tuple:
    """Fires requests on a Poisson process at `rate_qps`, one daemon thread
    per accepted request, gated by an in-flight cap. Returns (records, summary).

    Arrival scheduling happens on the calling thread and never waits on a
    request's completion — that's the entire point of open-loop: a slow
    server must not throttle how fast we *offer* load, only how fast it can
    absorb it.
    """
    rng = random.Random(seed)
    gate = InFlightGate(max_inflight)
    results: list = []
    results_lock = threading.Lock()
    n_dropped = 0
    n_dispatched = 0

    mono_start = time.monotonic()
    wall_offset = time.time() - mono_start

    # Build the arrival schedule up front as cumulative exponential gaps —
    # a Poisson process, not a fixed metronome. Bursty clustering is exactly
    # what a uniform interval would hide, and it's what breaks servers.
    arrivals = []
    t = 0.0
    if count:
        for _ in range(count):
            t += rng.expovariate(rate_qps)
            arrivals.append(t)
    else:
        while t < duration_s:
            t += rng.expovariate(rate_qps)
            if t < duration_s:
                arrivals.append(t)

    hard_deadline = mono_start + max(duration_s, arrivals[-1] if arrivals else 0.0) + drain_timeout

    for offset in arrivals:
        target = mono_start + offset
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
        t_arrival_mono = time.monotonic()

        if not gate.try_acquire():
            n_dropped += 1
            with results_lock:
                results.append(
                    {
                        "idx": n_dispatched + n_dropped,
                        "ok": False,
                        "dropped": True,
                        "error": None,
                        "t_arrival": t_arrival_mono + wall_offset,
                        "t_end": t_arrival_mono + wall_offset,
                    }
                )
            continue

        prompt = next(prompt_cycle)
        th = threading.Thread(
            target=run_one_request,
            args=(n_dispatched, prompt, max_tokens, t_arrival_mono, wall_offset, gate, results, results_lock),
            daemon=True,  # daemon: a stuck/slow request must never block process exit (hard-stop requirement)
        )
        n_dispatched += 1
        th.start()

        if time.monotonic() > hard_deadline:
            break  # hard stop: don't let a saturated server make the sweep run forever

    wall_run_end = time.monotonic()

    # Drain: wait for in-flight requests to finish, but only up to
    # drain_timeout past the point arrivals stopped — another hard stop.
    drain_deadline = wall_run_end + drain_timeout
    while gate.count > 0 and time.monotonic() < drain_deadline:
        time.sleep(0.05)

    n_abandoned = gate.count  # still in flight when we gave up waiting
    with results_lock:
        records = list(results)
    wall_elapsed = time.monotonic() - mono_start

    summary = summarize_rate_level(rate_qps, records, wall_elapsed, n_dropped, n_abandoned, max_inflight)
    return records, summary


def summarize_rate_level(rate_qps: float, records: list, wall_elapsed: float,
                          n_dropped: int, n_abandoned: int, max_inflight: int) -> dict:
    ok = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok") and not r.get("dropped")]

    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    e2es = [r["e2e_s"] for r in ok if r.get("e2e_s") is not None]
    all_itls = [v for r in ok for v in r.get("itls", [])]

    total_completion_tokens = sum(r.get("completion_tokens") or 0 for r in ok)
    total_prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok)
    achieved_qps = len(ok) / wall_elapsed if wall_elapsed > 0 else 0.0

    mean_conc, max_conc = concurrency_series(records)
    grows = queueing_grows(records)

    # Saturation signal #1: achieved QPS meaningfully below offered QPS.
    # 85% is a deliberately loose threshold — a couple of dropped/slow
    # requests shouldn't false-positive a healthy run.
    throughput_gap = achieved_qps < rate_qps * 0.85
    saturated = throughput_gap or grows

    n_degen = sum(1 for r in ok if r.get("degenerate"))

    return {
        "offered_qps": rate_qps,
        "achieved_qps": achieved_qps,
        "wall_elapsed_s": wall_elapsed,
        "n_dispatched": len(ok) + len(failed),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "n_dropped": n_dropped,
        "n_abandoned": n_abandoned,
        "max_inflight_cap": max_inflight,
        "ttft_p50": percentile(ttfts, 0.5),
        "ttft_p95": percentile(ttfts, 0.95),
        "ttft_p99": percentile(ttfts, 0.99),
        "ttft_max": max(ttfts) if ttfts else None,
        "itl_p50": percentile(all_itls, 0.5),
        "itl_p95": percentile(all_itls, 0.95),
        "e2e_p50": percentile(e2es, 0.5),
        "e2e_p95": percentile(e2es, 0.95),
        "e2e_p99": percentile(e2es, 0.99),
        "output_tok_s": total_completion_tokens / wall_elapsed if wall_elapsed > 0 else 0.0,
        "prefill_tok_s": total_prompt_tokens / wall_elapsed if wall_elapsed > 0 else 0.0,
        "mean_inflight": mean_conc,
        "max_inflight_observed": max_conc,
        "queueing_grows": grows,
        "throughput_gap": throughput_gap,
        "saturated": saturated,
        "n_degenerate": n_degen,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _fmt(v, ndigits=2):
    return "-" if v is None else f"{v:.{ndigits}f}"


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def print_summary_table(summaries: list) -> None:
    header = (
        f"{'QPS off':>7} {'QPS ach':>7} {'ttft p50':>8} {'ttft p95':>8} {'ttft p99':>8} "
        f"{'itl p50':>7} {'itl p95':>7} {'e2e p50':>7} {'e2e p95':>7} {'e2e p99':>7} "
        f"{'out tok/s':>9} {'pfill tok/s':>11} {'conc avg':>8} {'conc max':>8} "
        f"{'ok':>5} {'drop':>4} {'err':>3} {'degen':>5} {'SAT':>3}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['offered_qps']:>7.1f} {s['achieved_qps']:>7.2f} "
            f"{_fmt(s['ttft_p50']):>8} {_fmt(s['ttft_p95']):>8} {_fmt(s['ttft_p99']):>8} "
            f"{_fmt(s['itl_p50'], 3):>7} {_fmt(s['itl_p95'], 3):>7} "
            f"{_fmt(s['e2e_p50']):>7} {_fmt(s['e2e_p95']):>7} {_fmt(s['e2e_p99']):>7} "
            f"{s['output_tok_s']:>9.1f} {s['prefill_tok_s']:>11.1f} "
            f"{s['mean_inflight']:>8.1f} {s['max_inflight_observed']:>8d} "
            f"{s['n_ok']:>5} {s['n_dropped']:>4} {s['n_failed']:>3} {s['n_degenerate']:>5} "
            f"{'YES' if s['saturated'] else '.':>3}"
        )
    print()
    for s in summaries:
        if s["saturated"]:
            reasons = []
            if s["throughput_gap"]:
                reasons.append(f"achieved QPS {s['achieved_qps']:.2f} << offered {s['offered_qps']:.1f}")
            if s["queueing_grows"]:
                reasons.append("in-flight concurrency grew monotonically through the run")
            print(f"SATURATED at offered={s['offered_qps']:.1f} QPS: {'; '.join(reasons)}")


# --------------------------------------------------------------------------
# Mode: run
# --------------------------------------------------------------------------

def warmup(prompt: str, max_tokens: int) -> None:
    # First request on a cold vLLM+Triton server pays a large autotune cost
    # that would otherwise dominate every other number in the run.
    print("warmup (pays cold-start/autotune cost, excluded from stats) ...", flush=True)
    gate = InFlightGate(1)
    gate.try_acquire()
    results = []
    run_one_request(-1, prompt, max_tokens, time.monotonic(), time.time() - time.monotonic(), gate, results, threading.Lock())
    rec = results[0]
    if rec["ok"]:
        print(f"  done in {rec['e2e_s']:.1f}s\n", flush=True)
    else:
        print(f"  WARMUP FAILED: {rec['error']}\n", flush=True)


def mode_run(args) -> None:
    prompts = load_prompts(args.num_prompts, args.prompt_token_budget, args.tokenizer)
    if not prompts:
        print("ERROR: no prompts loaded", file=sys.stderr)
        sys.exit(1)
    print(f"loaded {len(prompts)} prompts (token_budget={args.prompt_token_budget})\n")

    model_tag = MODEL.replace("/", "_").replace(":", "_")
    label = args.label or "default"
    ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"=== open-loop serving benchmark: model={MODEL} label={label} rates={args.rates} ===\n")
    warmup(prompts[0], args.max_tokens)

    summaries = []
    for rate in args.rates:
        prompt_cycle = itertools.cycle(prompts)
        print(f"--- offered rate: {rate} QPS (duration={args.duration}s, max_inflight={args.max_inflight}) ---", flush=True)
        records, summary = run_rate_level(
            rate_qps=rate,
            prompt_cycle=prompt_cycle,
            max_tokens=args.max_tokens,
            duration_s=args.duration,
            count=args.count,
            max_inflight=args.max_inflight,
            drain_timeout=args.drain_timeout,
            seed=args.seed + int(rate * 1000),
        )
        summaries.append(summary)

        rate_path = OUT_DIR / f"bench_serving_{label}_{model_tag}_rate{rate}_{ts}.jsonl"
        write_jsonl(rate_path, records)
        print(
            f"  achieved={summary['achieved_qps']:.2f} QPS  ok={summary['n_ok']} "
            f"dropped={summary['n_dropped']} failed={summary['n_failed']} "
            f"ttft_p50={_fmt(summary['ttft_p50'])} e2e_p50={_fmt(summary['e2e_p50'])} "
            f"{'[SATURATED]' if summary['saturated'] else ''}",
            flush=True,
        )
        print(f"  raw records -> {rate_path}\n", flush=True)

    print("\n=== summary ===")
    print_summary_table(summaries)

    summary_path = OUT_DIR / f"bench_serving_summary_{label}_{model_tag}_{ts}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"label": label, "model": MODEL, "timestamp": ts, "rates": summaries}, f, indent=2, default=str)
    print(f"\nsummary -> {summary_path}")


# --------------------------------------------------------------------------
# Mode: compare
# --------------------------------------------------------------------------

def mode_compare(paths: list) -> None:
    expanded = []
    for p in paths:
        expanded.extend(sorted(glob.glob(p)) or [p])

    runs = []
    for p in expanded:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))

    if not runs:
        print("ERROR: no summary files to compare", file=sys.stderr)
        sys.exit(1)

    print(f"comparing {len(runs)} run(s): " + ", ".join(f"{r['label']}({r['model']})" for r in runs) + "\n")

    all_rates = sorted({round(s["offered_qps"], 3) for r in runs for s in r["rates"]})
    header = (
        f"{'QPS off':>7} {'label':<16} {'QPS ach':>7} {'ttft p50':>8} {'ttft p95':>8} "
        f"{'e2e p50':>7} {'e2e p95':>7} {'out tok/s':>9} {'pfill tok/s':>11} {'degen':>5} {'SAT':>3}"
    )
    print(header)
    print("-" * len(header))
    for rate in all_rates:
        for r in runs:
            match = next((s for s in r["rates"] if round(s["offered_qps"], 3) == rate), None)
            if match is None:
                continue
            print(
                f"{rate:>7.1f} {r['label']:<16.16} {match['achieved_qps']:>7.2f} "
                f"{_fmt(match['ttft_p50']):>8} {_fmt(match['ttft_p95']):>8} "
                f"{_fmt(match['e2e_p50']):>7} {_fmt(match['e2e_p95']):>7} "
                f"{match['output_tok_s']:>9.1f} {match['prefill_tok_s']:>11.1f} "
                f"{match['n_degenerate']:>5} {'YES' if match['saturated'] else '.':>3}"
            )
        print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Open-loop, rate-based serving benchmark (Poisson arrivals) for an OpenAI-compatible vLLM endpoint"
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    run_ap = sub.add_parser("run", help="run the QPS sweep against a live server")
    run_ap.add_argument("--rates", type=float, nargs="+", default=[3, 5, 7, 9, 12, 30], help="offered QPS levels to sweep")
    run_ap.add_argument("--duration", type=float, default=60.0, help="seconds of arrivals per rate level (ignored if --count set)")
    run_ap.add_argument("--count", type=int, default=0, help="fixed number of requests per rate level instead of a duration")
    run_ap.add_argument("--max-inflight", type=int, default=128, help="cap on concurrent in-flight requests; excess are dropped and counted")
    run_ap.add_argument("--drain-timeout", type=float, default=60.0, help="max extra seconds to wait for in-flight requests after arrivals stop (hard stop)")
    run_ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("BENCH_MAX_TOKENS", "256")))
    run_ap.add_argument("--num-prompts", type=int, default=500, help="distinct prompts to cache/cycle through")
    run_ap.add_argument("--prompt-token-budget", type=int, default=12000, help="truncate prompts to this many tokens (target context is 16K)")
    run_ap.add_argument("--tokenizer", default=None, help="tokenizer name for exact token-budget truncation (needs `transformers`); default: char-count estimate")
    run_ap.add_argument("--label", default=None, help="tag this run (e.g. a GGUF quant name) so results land in separate, comparable files")
    run_ap.add_argument("--model", default=None, help="overrides env VLLM_MODEL")
    run_ap.add_argument("--url", default=None, help="overrides env VLLM_URL (default http://localhost:8000)")
    run_ap.add_argument("--timeout", type=float, default=None, help="per-chunk read timeout seconds (env BENCH_TIMEOUT)")
    run_ap.add_argument("--seed", type=int, default=42, help="base RNG seed for Poisson arrival schedules (reproducible sweeps)")

    cmp_ap = sub.add_parser("compare", help="side-by-side comparison of several summary JSON files (e.g. different quants)")
    cmp_ap.add_argument("summaries", nargs="+", help="summary JSON files or globs, e.g. out/bench_serving_summary_*.json")

    args = ap.parse_args()

    if args.mode == "compare":
        mode_compare(args.summaries)
        return

    global MODEL, URL, READ_TIMEOUT
    if args.model:
        MODEL = args.model
    if not MODEL:
        print("ERROR: set VLLM_MODEL or pass --model", file=sys.stderr)
        sys.exit(1)
    if args.url:
        URL = args.url.rstrip("/") + "/v1/chat/completions"
    if args.timeout:
        READ_TIMEOUT = args.timeout

    mode_run(args)


if __name__ == "__main__":
    main()
