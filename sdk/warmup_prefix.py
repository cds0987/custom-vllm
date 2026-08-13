#!/usr/bin/env python3
"""
Prefix-cache warmup tool for an OpenAI-compatible vLLM server.

WHY THIS EXISTS
----------------
Every measurement in this repo agrees on one thing: the FIRST request that
touches a long shared prefix pays the full, uncached prefill cost, and
every request after it (as long as the prefix stays resident in the
prefix-cache KV blocks) is dramatically cheaper. Measured on this fork
(see STATUS.md):

    prefix ~30K tokens : cold prefill  10.5s   (warm-up run TTFT)
    prefix ~120K tokens: cold prefill  62.9s
    warm TTFT (cache hit), any of the above: ~0.2 - 1.0s

That cold cost is paid again on every server restart -- prefix-cache state
lives in GPU KV-cache memory, not on disk, so a fresh `vllm serve` process
(new deploy, autoscaled replica, crash recovery) starts with an empty
cache. If the first real user request is the one that pays the 10-63s
cold-prefill tax, that user sees it as latency/timeout, and in a blue-green
or rolling-restart deploy the "health check passed" signal is misleading:
the process is up and answering, but the very next request could still be
tens of seconds slow.

This script removes that by deliberately paying the cold-prefill cost
itself, before the server is allowed to take real traffic:

  1. Poll the server (health endpoint, falling back to /v1/models) with
     backoff until it responds -- vLLM's HTTP server can come up seconds
     before the engine has finished loading/compiling, so a warmup script
     that gives up on the first connection-refused is useless. This is
     the main point of the tool: be patient, not clever.
  2. Fire exactly ONE chat/completions request whose system message is
     the full prefix content, with max_tokens small (default 1) --
     purpose is to force the engine to prefill and cache the prefix's KV
     blocks, not to generate text.
  3. Report the cold wall-clock time and the prompt_tokens the server
     reports back (usage), which is the actual number of tokens that just
     got cached.
  4. Optionally (--verify) fire a second, byte-identical request and
     confirm it comes back much faster (< 30% of the first request's time
     by default) -- this is the only real proof the prefix cache is doing
     its job. If it doesn't speed up, that is a strong signal the server
     was started without --enable-prefix-caching (or hashing/eviction is
     preventing a hit), and the script says so loudly and exits non-zero.

USAGE IN A BLUE-GREEN / ROLLING DEPLOY
---------------------------------------
    1. Start the new server:      vllm serve ... --enable-prefix-caching &
    2. Warm it up before it takes traffic:
           python scripts/warmup_prefix.py \\
               --base-url http://localhost:8000 \\
               --prefix-file skills_pack/system_prefix.txt \\
               --verify
       Exit code 0 means: server responded, prefix is cached, and (if
       --verify was passed) the second request measurably hit the cache.
       Non-zero means: server never came up within --timeout, the warmup
       request itself failed, or --verify failed to show a speedup --
       treat all of these as "do not cut traffic over yet".
    3. Only after step 2 exits 0, flip the load balancer / DNS / service
       mesh weight to the new server.
    4. Tear down the old server.

This composes naturally as an entrypoint/systemd ExecStartPost step or a
CI/CD deploy-script stage; nothing here assumes a human is watching.
"""

import argparse
import json
import re
import sys
import time

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
CONNECT_TIMEOUT = 10.0


# --------------------------------------------------------------------------
# Prefix-cache metrics scraping (same approach as scripts/bench_skills.py --
# reused here on purpose, see that module's docstring/TASK H note about the
# vllm 0.26 "_total" suffix rename silently zeroing out bare-name scrapes).
# --------------------------------------------------------------------------

_METRIC_LINE_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(?P<value>[-\d.eE+]+)\s*$')

PREFIX_CACHE_METRIC_NAMES = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
)

_METRIC_NAME_ALIASES = {
    name + "_total": name for name in PREFIX_CACHE_METRIC_NAMES
}


def scrape_prefix_cache_metrics(base_url: str) -> dict:
    """Sum every series for each tracked metric name. Returns {} on any
    failure (endpoint missing/unreachable/disabled) -- metrics are a nice-to
    -have diagnostic here, never a hard requirement for warmup to succeed."""
    totals = {name: 0.0 for name in PREFIX_CACHE_METRIC_NAMES}
    try:
        r = requests.get(base_url.rstrip("/") + "/metrics", timeout=10)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return {}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        name = _METRIC_NAME_ALIASES.get(name, name)
        if name in totals:
            totals[name] += float(m.group("value"))
    return totals


# --------------------------------------------------------------------------
# Wait-for-server
# --------------------------------------------------------------------------

def wait_for_server(base_url: str, timeout: float, retries: int = None) -> bool:
    """Poll /health, falling back to /v1/models, with exponential backoff
    (capped at 5s between attempts) until the server answers or `timeout`
    seconds have elapsed. Returns True as soon as either endpoint responds
    with a 2xx status; False if the deadline (or `retries`, if given) is hit
    first.

    This is deliberately patient: right after `vllm serve` is launched, the
    HTTP listener can be up (connection accepted) well before the engine has
    finished loading weights / compiling, during which health checks may
    connection-refuse, time out, or 503 -- all of those are treated as
    "not ready yet", not as fatal errors.
    """
    base_url = base_url.rstrip("/")
    deadline = time.monotonic() + timeout
    attempt = 0
    backoff = 0.5
    while True:
        attempt += 1
        for path in ("/health", "/v1/models"):
            try:
                r = requests.get(base_url + path, timeout=CONNECT_TIMEOUT)
                if 200 <= r.status_code < 300:
                    print(f"server ready ({path} -> {r.status_code}) after {attempt} attempt(s)", flush=True)
                    return True
            except requests.exceptions.RequestException:
                pass
        if retries is not None and attempt >= retries:
            print(f"server did not become ready after {attempt} attempt(s) (retries exhausted)", file=sys.stderr)
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"server did not become ready within {timeout}s", file=sys.stderr)
            return False
        sleep_s = min(backoff, remaining, 5.0)
        print(f"server not ready yet (attempt {attempt}), retrying in {sleep_s:.1f}s ...", flush=True)
        time.sleep(sleep_s)
        backoff = min(backoff * 2, 5.0)


# --------------------------------------------------------------------------
# Warmup request
# --------------------------------------------------------------------------

def load_prefix(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ValueError(f"prefix file {path} is empty")
    return text


def build_payload(prefix: str, model: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "."},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }


def send_warmup_request(base_url: str, prefix: str, model: str, max_tokens: int, read_timeout: float) -> dict:
    """Fire one non-streaming chat/completions request containing `prefix`
    as the system message. Returns {ok, elapsed_s, prompt_tokens, error}."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = build_payload(prefix, model, max_tokens)
    t0 = time.monotonic()
    try:
        r = requests.post(url, json=payload, timeout=(CONNECT_TIMEOUT, read_timeout))
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        obj = r.json()
        usage = obj.get("usage") or {}
        return {
            "ok": True,
            "elapsed_s": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "error": None,
        }
    except requests.exceptions.RequestException as e:
        return {"ok": False, "elapsed_s": time.monotonic() - t0, "prompt_tokens": None, "error": f"{type(e).__name__}: {e}"}
    except (json.JSONDecodeError, ValueError) as e:
        return {"ok": False, "elapsed_s": time.monotonic() - t0, "prompt_tokens": None, "error": f"bad response body: {type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Warm the prefix cache of a running vLLM OpenAI-compatible server before "
        "it takes real traffic (cold prefix prefill measured at 10.5s@30K / 62.9s@120K tokens "
        "on this fork; warm TTFT is 0.2-1.0s). See module docstring for the blue-green deploy flow."
    )
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"server base URL (default {DEFAULT_BASE_URL})")
    ap.add_argument("--prefix-file", required=True, help="path to a text file whose content becomes the system-message prefix")
    ap.add_argument("--model", default="", help="model name for the request payload (many single-model vLLM servers ignore this field)")
    ap.add_argument("--max-tokens", type=int, default=1, help="max_tokens for the warmup request (default 1: only cache the prefix, don't generate)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the server to become ready before giving up (default 600)")
    ap.add_argument("--retries", type=int, default=None, help="cap the number of readiness-poll attempts (default: unbounded until --timeout)")
    ap.add_argument("--read-timeout", type=float, default=None, help="per-request read timeout in seconds (default: same as --timeout)")
    ap.add_argument("--verify", action="store_true", help="send a second identical request and confirm it is measurably faster (proves the cache hit)")
    ap.add_argument("--verify-threshold", type=float, default=0.3, help="PASS if 2nd request time < this fraction of the 1st (default 0.3 = must be <30%% of cold time)")
    args = ap.parse_args(argv)

    read_timeout = args.read_timeout if args.read_timeout is not None else args.timeout

    try:
        prefix = load_prefix(args.prefix_file)
    except (OSError, ValueError) as e:
        print(f"ERROR: could not load prefix file: {e}", file=sys.stderr)
        return 2

    print(f"waiting for server at {args.base_url} (timeout={args.timeout}s) ...", flush=True)
    if not wait_for_server(args.base_url, args.timeout, args.retries):
        return 3

    print(f"sending cold warmup request (prefix file: {args.prefix_file}, {len(prefix)} chars) ...", flush=True)
    first = send_warmup_request(args.base_url, prefix, args.model, args.max_tokens, read_timeout)
    if not first["ok"]:
        print(f"ERROR: warmup request failed: {first['error']}", file=sys.stderr)
        return 4

    print(f"cold prefill time: {first['elapsed_s']:.3f}s")
    print(f"prompt_tokens (server-reported): {first['prompt_tokens']}")
    if first["prompt_tokens"]:
        print(f"estimated cached: ~{first['prompt_tokens']} tokens")

    metrics = scrape_prefix_cache_metrics(args.base_url)
    if metrics:
        queries = metrics.get("vllm:prefix_cache_queries")
        hits = metrics.get("vllm:prefix_cache_hits")
        print(f"/metrics: prefix_cache_queries={queries} prefix_cache_hits={hits}")

    if not args.verify:
        print("warmup complete (--verify not requested).")
        return 0

    print("sending verify request (identical prefix, expecting a cache hit) ...", flush=True)
    second = send_warmup_request(args.base_url, prefix, args.model, args.max_tokens, read_timeout)
    if not second["ok"]:
        print(f"ERROR: verify request failed: {second['error']}", file=sys.stderr)
        return 5

    print(f"verify request time: {second['elapsed_s']:.3f}s")
    ratio = (second["elapsed_s"] / first["elapsed_s"]) if first["elapsed_s"] > 0 else 1.0
    if ratio < args.verify_threshold:
        print(f"VERIFY PASS: 2nd request took {ratio:.1%} of the 1st (< {args.verify_threshold:.0%} threshold) -- prefix cache is working.")
        return 0
    else:
        print(
            f"VERIFY WARNING: 2nd request took {ratio:.1%} of the 1st (>= {args.verify_threshold:.0%} threshold) -- "
            "no clear speedup. Check the server was started with --enable-prefix-caching, "
            "and that nothing between requests (load balancer routing to a different replica, "
            "cache eviction under memory pressure) broke the hit.",
            file=sys.stderr,
        )
        return 6


if __name__ == "__main__":
    sys.exit(main())
