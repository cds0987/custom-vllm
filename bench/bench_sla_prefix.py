"""
Open-loop Poisson SLA sweep against a shared-prefix production scenario
(long cached context + short unique per-request suffix). Generalizes the
Colab-local taskF2_sla_sweep.py / taskF2_warmup.py scripts written for
TASK F2/F2b into a repo-tracked, argparse-driven tool so it survives Colab
runtime recycles (everything under /content is ephemeral).

Reuses bench_serving.py's InFlightGate / run_one_request / summarize_rate_level
building blocks, but deliberately does NOT use bench_serving's own
run_rate_level or its LongAlign dataset / anti-cache front-tagging: this
scenario needs the opposite -- one FIXED shared prefix that must stay
prefix-cache-hot across the whole run, with the per-request unique tag
placed AFTER the prefix (in the suffix) instead of at the front, so vLLM's
automatic prefix caching still matches the leading token blocks.

Usage:
    python scripts/bench_sla_prefix.py \\
        --model RedHatAI/Qwen3.5-9B-quantized.w4a16 \\
        --prefix-file /tmp/fixed_prefix.txt \\
        --rates 0.1 0.2 0.3 \\
        --duration 180 \\
        --max-tokens 400

The prefix file is any plain-text file long enough to be worth caching
(e.g. a synthetic ~30K-token document); this script does not generate or
ship one, since prefix content/length is scenario-specific and often
built from whatever long-context corpus a given task calls for (see
STATUS.md / TASK F for how the project's own 30K/60K prefixes were made).

Warm-up: always sends one full-prefix request (small max_tokens) before
the sweep starts, so the first rate level isn't paying a cold-cache
penalty that has nothing to do with the rate being measured.

Optional /metrics scraping: pass --metrics-rate to poll the server's
/metrics endpoint every --metrics-interval seconds during that one rate
level only (num_requests_running/waiting, kv_cache_usage_perc, cumulative
prompt/generation token counters) -- useful for telling prefill/decode
compute contention apart from KV-capacity or admission-queue backlog
(see TASK F2b's contention-vs-capacity read in STATUS.md for how these
signals were interpreted).
"""

import argparse
import itertools
import json
import re
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_serving as bs  # noqa: E402

FILLER_UNIT = "Chi tiet rieng cho phien nay ve tai lieu da cung cap phia tren, muc "

METRICS_KEYS = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]


def make_suffix(n: int, repeats: int) -> str:
    body = (FILLER_UNIT + str(n) + ". ") * repeats
    return f"[req {n}] " + body + "\n\nDua tren tai lieu o tren, hay phan tich chi tiet va tra loi day du."


def prompt_for(n: int, prefix: str, repeats: int) -> str:
    return prefix + "\n\n" + make_suffix(n, repeats)


def warmup(prefix: str, model: str, url: str, timeout: float = 120.0) -> None:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prefix + "\n\nWarm-up."}],
        "max_tokens": 8,
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    usage = r.json().get("usage")
    print(f"warm-up took {time.time() - t0:.2f}s, usage={usage}", flush=True)


def scrape_metrics(metrics_url: str, interval: float, stop_event: threading.Event, samples: list) -> None:
    while not stop_event.is_set():
        try:
            txt = requests.get(metrics_url, timeout=5).text
            sample = {"t": time.time()}
            for key in METRICS_KEYS:
                m = re.search(re.escape(key) + r"(?:\{[^}]*\})?\s+([0-9eE+\-.]+)", txt)
                sample[key] = float(m.group(1)) if m else None
            samples.append(sample)
        except Exception as e:  # noqa: BLE001 - best-effort background scraper
            samples.append({"t": time.time(), "error": str(e)})
        stop_event.wait(interval)


def run_rate_level_shared_prefix(
    prefix: str,
    rate_qps: float,
    duration_s: float,
    max_tokens: int,
    suffix_repeats: int,
    max_inflight: int,
    seed: int,
    metrics_url: str | None = None,
    metrics_interval: float = 10.0,
):
    import random

    rng = random.Random(seed)
    gate = bs.InFlightGate(max_inflight)
    results: list = []
    results_lock = threading.Lock()
    n_dropped = 0
    n_dispatched = 0
    counter = itertools.count()
    metrics_samples: list = []
    stop_metrics = threading.Event()

    mt = None
    if metrics_url:
        mt = threading.Thread(
            target=scrape_metrics, args=(metrics_url, metrics_interval, stop_metrics, metrics_samples), daemon=True
        )
        mt.start()

    mono_start = time.monotonic()
    wall_offset = time.time() - mono_start

    arrivals = []
    t = 0.0
    while t < duration_s:
        t += rng.expovariate(rate_qps)
        if t < duration_s:
            arrivals.append(t)

    drain_timeout = 60.0
    hard_deadline = mono_start + duration_s + drain_timeout

    def run_one(idx: int, prompt: str, t_arrival_mono: float) -> None:
        bs.run_one_request(idx, prompt, max_tokens, t_arrival_mono, wall_offset, gate, results, results_lock)

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

        n = next(counter)
        prompt = prompt_for(n, prefix, suffix_repeats)
        th = threading.Thread(target=run_one, args=(n_dispatched, prompt, t_arrival_mono), daemon=True)
        n_dispatched += 1
        th.start()

        if time.monotonic() > hard_deadline:
            break

    drain_deadline = time.monotonic() + drain_timeout
    while gate.count > 0 and time.monotonic() < drain_deadline:
        time.sleep(0.05)

    if mt is not None:
        stop_metrics.set()
        mt.join(timeout=2)

    n_abandoned = gate.count
    with results_lock:
        records = list(results)
    wall_elapsed = time.monotonic() - mono_start

    summary = bs.summarize_rate_level(rate_qps, records, wall_elapsed, n_dropped, n_abandoned, max_inflight)
    return records, summary, metrics_samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name as registered with the server (--served-model-name)")
    ap.add_argument("--prefix-file", required=True, help="plain-text file holding the fixed shared prefix")
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.2, 0.3], help="QPS levels to sweep")
    ap.add_argument("--duration", type=float, default=180.0, help="seconds per rate level")
    ap.add_argument("--max-tokens", type=int, default=400, help="output tokens per request")
    ap.add_argument("--suffix-repeats", type=int, default=90, help="FILLER_UNIT repeat count (controls suffix length, ~2-2.5K tok at 90)")
    ap.add_argument("--max-inflight", type=int, default=64, help="client-side concurrent-request admission cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metrics-rate", type=float, default=None, help="if set, scrape /metrics during this one rate level only")
    ap.add_argument("--metrics-interval", type=float, default=10.0)
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--output-json", default=None, help="optional path to dump the full per-rate results")
    args = ap.parse_args()

    bs.MODEL = args.model
    bs.URL = args.url

    prefix = Path(args.prefix_file).read_text()

    if not args.skip_warmup:
        print("warmup (pays prefix-cache cold cost + Triton autotune) ...", flush=True)
        warmup(prefix, args.model, args.url)

    all_summaries = {}
    all_records = {}
    all_metrics = {}

    for rate in args.rates:
        print(f"\n=== rate={rate} QPS, duration={args.duration}s ===", flush=True)
        metrics_url = args.metrics_url if rate == args.metrics_rate else None
        records, summary, metrics_samples = run_rate_level_shared_prefix(
            prefix,
            rate,
            args.duration,
            args.max_tokens,
            args.suffix_repeats,
            args.max_inflight,
            args.seed,
            metrics_url=metrics_url,
            metrics_interval=args.metrics_interval,
        )
        over3s = sum(1 for r in records if r.get("ok") and (r.get("ttft_s") or 0) > 3.0)
        ttft_p50 = summary["ttft_p50"] or -1
        ttft_p95 = summary["ttft_p95"] or -1
        e2e_p95 = summary["e2e_p95"] or -1
        print(
            f"achieved={summary['achieved_qps']:.2f} ok={summary['n_ok']} "
            f"dropped={summary['n_dropped']} failed={summary['n_failed']}",
            flush=True,
        )
        print(f"ttft_p50={ttft_p50:.2f}s ttft_p95={ttft_p95:.2f}s e2e_p95={e2e_p95:.2f}s", flush=True)
        print(f"requests with TTFT>3s: {over3s}/{summary['n_ok']}", flush=True)
        print(f"saturated={summary.get('saturated')}", flush=True)
        if metrics_samples:
            print(f"--- /metrics samples during rate={rate} ---", flush=True)
            for s in metrics_samples:
                print(json.dumps(s), flush=True)

        all_summaries[rate] = summary
        all_records[rate] = records
        all_metrics[rate] = metrics_samples

    print("\n=== SUMMARY TABLE ===")
    print(f"{'rate':>6} {'achieved':>8} {'ttft_p50':>8} {'ttft_p95':>8} {'e2e_p95':>8} {'ok':>5} {'drop':>5}")
    for rate, s in all_summaries.items():
        print(
            f"{rate:>6.2f} {s['achieved_qps']:>8.2f} {s['ttft_p50'] or -1:>8.2f} "
            f"{s['ttft_p95'] or -1:>8.2f} {s['e2e_p95'] or -1:>8.2f} {s['n_ok']:>5} {s['n_dropped']:>5}"
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "rates": args.rates,
                    "duration": args.duration,
                    "max_tokens": args.max_tokens,
                    "suffix_repeats": args.suffix_repeats,
                    "summaries": all_summaries,
                    "records": all_records,
                    "metrics": all_metrics,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.output_json}")

    print("\nDONE_BENCH_SLA_PREFIX")


if __name__ == "__main__":
    main()
