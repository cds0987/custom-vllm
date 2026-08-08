"""SWE-bench-driven serving benchmark for an OpenAI-compatible vLLM endpoint.

Supersedes bench_load.py: that script fired short, non-streaming requests,
which hides two things that matter for a real coding-agent workload —
prefill cost on long issue-text prompts (TTFT) and whether concurrent
streams are served fairly or whether some starve while others race ahead.
This script uses real SWE-bench problem statements (500-3000 tokens) so
prefill isn't trivial, and streams via SSE so TTFT/decode can be separated.

Two modes:
  single      - sequential requests, one at a time: how fast does a single
                stream go (TTFT, decode tok/s, e2e latency), p50/p95 over N
                SWE-bench instances.
  concurrent  - fire C requests at once over a sweep of concurrency levels;
                report fleet throughput plus a fairness view (spread of
                per-request decode rates, Jain index, completion-time gap).

Usage:
    VLLM_MODEL=repo:QUANT python scripts/bench_swebench.py single --n 20
    VLLM_MODEL=repo:QUANT python scripts/bench_swebench.py concurrent --concurrency 1 2 4 8 16 32

Every response is also checked for the failure mode that has actually bitten
this project: a fast server streaming back a single token repeated forever.
That's tracked as "degenerate" alongside the speed numbers so a broken quant
doesn't look like a win just because it's fast.
"""

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = REPO_ROOT / ".cache" / "swebench_dev.jsonl"
OUT_DIR = REPO_ROOT / "out"
DATASET_FIELDS = ["repo", "instance_id", "base_commit", "patch", "problem_statement", "hints_text"]

# Mutated by main() from CLI args / env; module-level like bench_load.py so
# the request helpers below don't need these threaded through every call.
MODEL = os.environ.get("VLLM_MODEL", "")
URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/") + "/v1/chat/completions"
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "300"))  # per-chunk stall ceiling, not total request time


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def load_swebench(limit: int) -> list:
    """Load SWE-bench/SWE-bench (dev split), caching to a local JSONL.

    Caching to disk (not just relying on the HF `datasets` cache) means
    reruns work even if `datasets` itself isn't installed, and keeps the
    benchmark reproducible offline once the first download has happened.
    """
    if not CACHE_FILE.exists():
        from datasets import load_dataset  # imported lazily: only needed on first run

        print(f"Downloading SWE-bench/SWE-bench (dev split) -> {CACHE_FILE} ...", file=sys.stderr)
        ds = load_dataset("SWE-bench/SWE-bench", split="dev")
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_FILE.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps({k: row[k] for k in DATASET_FIELDS}) + "\n")

    rows = []
    with CACHE_FILE.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_prompt(row: dict) -> str:
    """Turn a SWE-bench row into a realistic ~500-3000 token agent prompt.

    Uses the real problem_statement/hints_text verbatim rather than a
    synthetic summary, so prefill length and content distribution resemble
    an actual coding-agent turn instead of a toy prompt that hides prefill
    cost entirely.
    """
    parts = [
        f"You are a software engineer working in the repository `{row['repo']}` "
        f"at commit {row['base_commit']}.",
        "A user has filed the following issue. Read it carefully and propose a fix.",
        "",
        "## Issue",
        row.get("problem_statement") or "",
    ]
    if row.get("hints_text"):
        parts += ["", "## Maintainer hints", row["hints_text"]]
    text = "\n".join(parts)

    # ~4 chars/token is a crude estimate; only used to keep prompts in the
    # target band. Reported token counts always come from the server's
    # `usage` field, never from this estimate.
    est_tokens = len(text) // 4
    if est_tokens > 3000:
        text = text[: 3000 * 4]
    elif est_tokens < 500 and text:
        # short issues alone rarely reach a meaningful prefill length; repeat
        # the (real, non-filler) framing text rather than padding with junk
        reps = (500 * 4) // len(text) + 1
        text = (text + "\n\n") * reps
        text = text[: 3000 * 4]

    return text + "\n\nRespond with a concise root-cause explanation and a proposed patch approach."


# --------------------------------------------------------------------------
# SSE streaming client
# --------------------------------------------------------------------------

def stream_chat_completion(messages: list, max_tokens: int):
    """POST /v1/chat/completions with stream=True and yield (t_recv, delta_text, usage).

    Parses SSE line-by-line rather than waiting for blank-line event
    separators: OpenAI-style chat streams emit exactly one `data:` line per
    event, and some proxies don't reliably send the trailing blank line, so
    line-based parsing is both simpler and less likely to drop the final
    event. Buffers raw bytes across chunk boundaries so a `data: ...` line
    split across two TCP reads is never mis-parsed as two lines.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
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
        # per-token timing (TTFT/ITL) this whole benchmark depends on.
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


def jain_index(values: list):
    """Jain's fairness index over per-stream decode rates: 1.0 = perfectly
    even, 1/n = maximally unfair (one stream gets everything)."""
    if not values:
        return None
    n = len(values)
    s1 = sum(values)
    s2 = sum(v * v for v in values)
    if s2 == 0:
        return 1.0
    return (s1 * s1) / (n * s2)


def is_degenerate(text: str) -> bool:
    """Flag empty output or output collapsed onto one repeated token.

    This is the exact failure mode that has bitten this project: a broken
    GGUF/quant patch that streams tokens fast but produces garbage. A speed
    number without this check would make that look like a win.
    """
    text = text.strip()
    words = text.split()
    if len(words) < 5:
        return len(text) == 0
    most_common = max(Counter(words).values())
    return most_common / len(words) > 0.6


# --------------------------------------------------------------------------
# Per-request runner (shared by both modes)
# --------------------------------------------------------------------------

def run_one_request(idx, messages, max_tokens: int) -> dict:
    record = {"idx": idx, "ok": False, "error": None}
    t0 = time.monotonic()
    token_times = []
    text_parts = []
    last_usage = None
    try:
        for t_recv, delta_text, usage in stream_chat_completion(messages, max_tokens):
            if usage:
                last_usage = usage
            if delta_text:
                token_times.append(t_recv)
                text_parts.append(delta_text)
        t_end = time.monotonic()
        text = "".join(text_parts)

        decode_tokens = None
        if last_usage:
            decode_tokens = last_usage.get("completion_tokens")
        if decode_tokens is None:
            decode_tokens = len(token_times)  # proxy: one SSE content-delta ~= one token

        record.update(
            ok=True,
            t0=t0,
            t_end=t_end,
            e2e_s=t_end - t0,
            ttft_s=(token_times[0] - t0) if token_times else None,
            prompt_tokens=last_usage.get("prompt_tokens") if last_usage else None,
            decode_tokens=decode_tokens,
            text_len_chars=len(text),
            text_preview=text[:200],
            degenerate=is_degenerate(text),
        )

        # decode tok/s *excluding* TTFT: tokens after the first, over the
        # span from first token to end. Needs >=2 token events to mean anything.
        if len(token_times) >= 2:
            decode_span = t_end - token_times[0]
            record["decode_tok_s"] = (decode_tokens - 1) / decode_span if decode_span > 0 else None
        else:
            record["decode_tok_s"] = None

        itls = [b - a for a, b in zip(token_times, token_times[1:])]
        record["itl_p50"] = statistics.median(itls) if itls else None
        record["itl_p95"] = percentile(itls, 0.95)

    except requests.exceptions.RequestException as e:
        record["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # never let one bad request crash the whole sweep
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"raw records -> {path}")


def warmup(rows: list, max_tokens: int) -> None:
    # First request on a cold vLLM+Triton server pays a large autotune cost
    # that would otherwise dominate every other number in the run.
    print("warmup (pays cold-start/autotune cost, excluded from stats) ...", flush=True)
    rec = run_one_request(-1, [{"role": "user", "content": build_prompt(rows[0])}], max_tokens)
    if rec["ok"]:
        print(f"  done in {rec['e2e_s']:.1f}s\n", flush=True)
    else:
        print(f"  WARMUP FAILED: {rec['error']}\n", flush=True)


# --------------------------------------------------------------------------
# Mode 1: single-request throughput
# --------------------------------------------------------------------------

def mode_single(rows: list, max_tokens: int, out_path: Path) -> None:
    print(f"=== single-request throughput: model={MODEL} n={len(rows)} max_tokens={max_tokens} ===\n")
    warmup(rows, max_tokens)

    records = []
    header = f"{'idx':>3} {'instance_id':<28} {'prompt_tok':>10} {'ttft_s':>8} {'decode_tok':>10} {'dec_tok/s':>9} {'e2e_s':>7} {'ok':>3} {'degen':>5}"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows):
        rec = run_one_request(i, [{"role": "user", "content": build_prompt(row)}], max_tokens)
        rec["instance_id"] = row["instance_id"]
        records.append(rec)
        if rec["ok"]:
            print(
                f"{i:>3} {row['instance_id']:<28.28} {str(rec['prompt_tokens']):>10} "
                f"{rec['ttft_s']:>8.3f} {str(rec['decode_tokens']):>10} "
                f"{('%.1f' % rec['decode_tok_s']) if rec['decode_tok_s'] is not None else '-':>9} "
                f"{rec['e2e_s']:>7.2f} {'Y':>3} {('Y' if rec['degenerate'] else '.'):>5}",
                flush=True,
            )
        else:
            print(f"{i:>3} {row['instance_id']:<28.28} FAILED: {rec['error']}", flush=True)

    write_jsonl(out_path, records)

    ok = [r for r in records if r["ok"]]
    print(f"\n{len(ok)}/{len(records)} requests succeeded")
    if ok:
        ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
        rates = [r["decode_tok_s"] for r in ok if r["decode_tok_s"] is not None]
        e2es = [r["e2e_s"] for r in ok]
        n_degen = sum(1 for r in ok if r["degenerate"])
        print(
            f"ttft_s        p50={percentile(ttfts, 0.5):.3f}  p95={percentile(ttfts, 0.95):.3f}"
            if ttfts else "ttft_s        (no samples)"
        )
        print(
            f"decode_tok/s  p50={percentile(rates, 0.5):.1f}  p95={percentile(rates, 0.95):.1f}"
            if rates else "decode_tok/s  (no samples)"
        )
        print(f"e2e_s         p50={percentile(e2es, 0.5):.2f}  p95={percentile(e2es, 0.95):.2f}")
        print(f"degenerate    {n_degen}/{len(ok)}")


# --------------------------------------------------------------------------
# Mode 2: concurrent serving fairness
# --------------------------------------------------------------------------

def summarize_concurrency(c: int, recs: list, t_launch: float) -> dict:
    ok = [r for r in recs if r["ok"]]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    itl50s = [r["itl_p50"] for r in ok if r["itl_p50"] is not None]
    itl95s = [r["itl_p95"] for r in ok if r["itl_p95"] is not None]
    rates = [r["decode_tok_s"] for r in ok if r["decode_tok_s"] is not None]
    total_tokens = sum(r["decode_tokens"] for r in ok)
    t_ends = [r["t_end"] for r in ok]
    fleet_span = (max(t_ends) - t_launch) if t_ends else 0.0
    completion_gap = (max(t_ends) - min(t_ends)) if len(t_ends) > 1 else 0.0

    return {
        "concurrency": c,
        "n_ok": len(ok),
        "n_failed": len(recs) - len(ok),
        "throughput_tok_s": (total_tokens / fleet_span) if fleet_span > 0 else 0.0,
        "ttft_p50": percentile(ttfts, 0.5),
        "ttft_p95": percentile(ttfts, 0.95),
        "ttft_max": max(ttfts) if ttfts else None,
        "itl_p50": percentile(itl50s, 0.5),
        "itl_p95": percentile(itl95s, 0.95),
        "rate_min": min(rates) if rates else None,
        "rate_p50": percentile(rates, 0.5),
        "rate_max": max(rates) if rates else None,
        "jain_fairness": jain_index(rates),
        "completion_gap_s": completion_gap,
    }


def mode_concurrent(rows: list, levels: list, max_tokens: int, out_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"=== concurrent serving fairness: model={MODEL} max_tokens={max_tokens} levels={levels} ===\n")
    warmup(rows, max_tokens)

    row_cycle = itertools.cycle(rows)  # sweep can need more requests than dataset rows; cycling beats erroring out
    all_records = []
    summaries = []
    for c in levels:
        batch = [next(row_cycle) for _ in range(c)]
        t_launch = time.monotonic()
        recs = [None] * c
        with ThreadPoolExecutor(max_workers=c) as pool:
            futures = {
                pool.submit(
                    run_one_request, i, [{"role": "user", "content": build_prompt(batch[i])}], max_tokens
                ): i
                for i in range(c)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                recs[i] = fut.result()
                recs[i]["concurrency"] = c
                recs[i]["instance_id"] = batch[i]["instance_id"]
        all_records.extend(recs)
        summaries.append(summarize_concurrency(c, recs, t_launch))
        s = summaries[-1]
        print(
            f"conc={c:3d}  ok={s['n_ok']}/{c}  throughput={s['throughput_tok_s']:7.1f} tok/s  "
            f"ttft p50={_fmt(s['ttft_p50'])} p95={_fmt(s['ttft_p95'])}  "
            f"itl p50={_fmt(s['itl_p50'])} p95={_fmt(s['itl_p95'])}  "
            f"gap={_fmt(s['completion_gap_s'])}s",
            flush=True,
        )

    write_jsonl(out_path, all_records)

    print("\n--- fairness summary ---")
    header = (
        f"{'conc':>4} {'ok':>5} {'thpt tok/s':>10} {'ttft p50':>9} {'ttft p95':>9} {'ttft max':>9} "
        f"{'itl p50':>8} {'itl p95':>8} {'rate min':>8} {'rate p50':>8} {'rate max':>8} {'jain':>6} {'gap s':>7}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['concurrency']:>4} {s['n_ok']:>2}/{s['n_ok'] + s['n_failed']:<2} "
            f"{s['throughput_tok_s']:>10.1f} "
            f"{_fmt(s['ttft_p50']):>9} {_fmt(s['ttft_p95']):>9} {_fmt(s['ttft_max']):>9} "
            f"{_fmt(s['itl_p50']):>8} {_fmt(s['itl_p95']):>8} "
            f"{_fmt(s['rate_min']):>8} {_fmt(s['rate_p50']):>8} {_fmt(s['rate_max']):>8} "
            f"{_fmt(s['jain_fairness'], 3):>6} {_fmt(s['completion_gap_s']):>7}"
        )

    n_degen = sum(1 for r in all_records if r.get("ok") and r.get("degenerate"))
    n_ok = sum(1 for r in all_records if r.get("ok"))
    print(f"\ndegenerate responses: {n_degen}/{n_ok}")


def _fmt(v, ndigits=2):
    return "-" if v is None else f"{v:.{ndigits}f}"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="SWE-bench-driven serving benchmark: single-stream throughput and concurrent fairness"
    )
    ap.add_argument("mode", choices=["single", "concurrent"])
    ap.add_argument(
        "--n",
        type=int,
        default=int(os.environ.get("SWEBENCH_LIMIT", "20")),
        help="number of SWE-bench instances to load/use (env SWEBENCH_LIMIT)",
    )
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("BENCH_MAX_TOKENS", "256")))
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32], help="mode=concurrent only")
    ap.add_argument("--model", default=None, help="overrides env VLLM_MODEL")
    ap.add_argument("--url", default=None, help="overrides env VLLM_URL (default http://localhost:8000)")
    ap.add_argument("--timeout", type=float, default=None, help="per-chunk read timeout seconds (env BENCH_TIMEOUT)")
    ap.add_argument("--out", default=None, help="JSONL output path (default out/bench_swebench_<mode>_<ts>.jsonl)")
    args = ap.parse_args()

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

    rows = load_swebench(args.n)
    if not rows:
        print("ERROR: no SWE-bench rows loaded", file=sys.stderr)
        sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else OUT_DIR / f"bench_swebench_{args.mode}_{ts}.jsonl"

    if args.mode == "single":
        mode_single(rows, args.max_tokens, out_path)
    else:
        mode_concurrent(rows, args.concurrency, args.max_tokens, out_path)


if __name__ == "__main__":
    main()
