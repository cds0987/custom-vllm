"""Shared-prefix ("skills pack") benchmark for an OpenAI-compatible vLLM endpoint.

Scenario: a large, fixed system prompt (a "skills pack" of instructions,
tens of thousands of tokens) is prepended to many different short questions.
This is the prefix-caching sweet spot — every request after the first should
be able to reuse the KV blocks computed for the shared system-prompt prefix,
so the thing worth measuring is not just raw TTFT/decode speed but *whether
the server's prefix cache is actually doing its job* (queries vs hits going
up the way they should) and how TTFT changes with concurrency once many
requests are contending for the same cached blocks.

This script is generic and dataset-free by design: it takes a `--prefix-file`
and a `--questions-file` as arguments and contains no benchmark content of its
own. The actual skills-pack system prompt and question set are produced
separately under `skills_pack/` (gitignored — never committed, see
STATUS.md TASK F/H) and are simply pointed to at run time:

    python scripts/bench_skills.py \\
        --prefix-file skills_pack/system_prefix.txt \\
        --questions-file skills_pack/selected_questions.jsonl \\
        --concurrency 1,8,16,32 --mode concurrent

SYNTHETIC MODE (for throughput measurement without private data):
    python scripts/bench_skills.py \\
        --synthetic-prefix-tokens 10000 \\
        --synthetic-questions 100 \\
        --concurrency 1,8,16,32 --mode concurrent

When using --synthetic-prefix-tokens and/or --synthetic-questions, the script
generates deterministic synthetic text (seeded by --seed, default 0) in place
of reading from files. This is useful for measuring *throughput* when no real
dataset is available — throughput depends only on token counts, not semantic
content. However, synthetic mode should NEVER be used to measure or report
*quality* metrics. Throughput numbers from synthetic and real-data modes are
not directly comparable.

Questions-file schema (JSONL, one object per line): {"id", "question",
"choices"?, "domain"?}. `choices` is an optional list/dict of MCQ options; if
present the client formats them into the user turn and appends an
"Đáp án: X" instruction so the model outputs a single-letter answer.

Output is JSONL with one record per request (including the *full* model
output text) plus trailing summary record(s) per concurrency level — full
text is kept so a later, separate pass can grade answer quality offline
without re-running inference. Nothing from the dataset (question text,
choices, prefix content) is ever printed to stdout; only ids and numbers are.

Prefix-cache instrumentation: this fork's Prometheus exporter
(vllm/v1/metrics/loggers.py, confirmed by grep against the vllm fork checked
out at d:\\Training\\AI_Module\\vllm\\vllm\\vllm) registers two Counters that
matter here:

    vllm:prefix_cache_queries{model_name=...,engine=...}
    vllm:prefix_cache_hits{model_name=...,engine=...}

(there's also vllm:external_prefix_cache_queries/hits for KV-connector/
external caches, scraped too in case that path is enabled). Both are
monotonic counters covering the whole process, so this script snapshots
/metrics immediately before and after each concurrency level and reports the
*delta* — that isolates the hit rate attributable to that level's requests
instead of the cumulative-since-boot rate.

Warm-up: one explicit request (prefix + first question) is fired before any
timed level, its TTFT reported separately as the cold prefix-compute cost,
and it is excluded from every aggregate — the whole point of the shared
prefix is that everyone *after* the first request should pay less than this.
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL = os.environ.get("VLLM_MODEL", "")
URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
CHAT_URL = URL + "/v1/chat/completions"
METRICS_URL = URL + "/metrics"
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "300"))


# --------------------------------------------------------------------------
# Synthetic data generation (for throughput measurement without real datasets)
# --------------------------------------------------------------------------

# Fixed vocabulary for deterministic synthetic generation
_VOCAB = [
    "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "Python", "code", "benchmark", "throughput", "token", "processing",
    "machine", "learning", "model", "inference", "server", "request",
    "system", "prompt", "question", "answer", "completion", "generation",
    "efficient", "fast", "reliable", "scalable", "robust", "optimal",
    "compute", "memory", "latency", "concurrent", "stream", "pipeline",
    "cache", "prefix", "data", "algorithm", "optimization", "performance",
]


def generate_synthetic_prefix(n_tokens: int, seed: int = 0) -> str:
    """Generate deterministic synthetic text approximately n_tokens in length.
    Uses ~4 chars per token (matching bench_skills.py's estimation convention)."""
    rng = random.Random(seed)
    n_words = (n_tokens * 4) // 5  # ~5 chars/word on average, so ~4 chars/token -> ~5/4 tokens/word
    words = [rng.choice(_VOCAB) for _ in range(n_words)]
    return " ".join(words)


def generate_synthetic_questions(n_questions: int, min_tokens: int = 50, max_tokens: int = 150, seed: int = 0) -> list:
    """Generate n_questions distinct synthetic questions, each approximately
    min_tokens to max_tokens tokens in length. Returns list of dicts with
    id and question fields."""
    rng = random.Random(seed)
    questions = []
    for i in range(n_questions):
        n_tokens = rng.randint(min_tokens, max_tokens)
        n_words = (n_tokens * 4) // 5
        # Shuffle vocabulary differently for each question to avoid exact duplication
        vocab_shuffled = _VOCAB.copy()
        rng.shuffle(vocab_shuffled)
        # Pick words, cycling through shuffled vocab to ensure variety
        words = []
        for j in range(n_words):
            words.append(vocab_shuffled[j % len(vocab_shuffled)])
        question_text = " ".join(words)
        questions.append({
            "id": f"synthetic_{i:06d}",
            "question": question_text,
        })
    return questions


# --------------------------------------------------------------------------
# Input loading (dataset-free: caller supplies the files)
# --------------------------------------------------------------------------

def load_prefix(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"prefix file {path} is empty")
    return text


def load_questions(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "question" not in obj:
                raise ValueError(f"question row missing id/question: keys={list(obj.keys())}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"questions file {path} has no rows")
    return rows


def format_choices(choices) -> str:
    """Choices may be a list (A, B, C, ... in order) or a dict {"A": "...", ...}."""
    if isinstance(choices, dict):
        items = sorted(choices.items())
    else:
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        items = list(zip(letters, choices))
    return "\n".join(f"{letter}. {text}" for letter, text in items)


def build_user_content(question_row: dict) -> str:
    question = question_row["question"]
    choices = question_row.get("choices")
    if choices:
        return f"{question}\n\n{format_choices(choices)}\n\nĐáp án: "
    return question


def build_messages(prefix: str, question_row: dict) -> list:
    return [
        {"role": "system", "content": prefix},
        {"role": "user", "content": build_user_content(question_row)},
    ]


# --------------------------------------------------------------------------
# SSE streaming client — same wire parsing as bench_serving.py/bench_swebench.py
# --------------------------------------------------------------------------

def stream_chat_completion(messages: list, max_tokens: int):
    """POST /v1/chat/completions with stream=True and yield (t_recv, delta_text, usage).

    chunk_size=1 (not None) is load-bearing: letting requests/urllib3 pick the
    read size coalesces a chunked-transfer response and only yields once the
    *entire* body has arrived, which silently destroys per-token timing
    (TTFT/ITL) — copied verbatim from bench_serving.py's stream_chat_completion.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    buf = b""
    with requests.post(
        CHAT_URL, json=payload, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    ) as r:
        r.raise_for_status()
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
                finish_reason = None
                choices = obj.get("choices") or []
                if choices:
                    delta_text = choices[0].get("delta", {}).get("content") or ""
                    finish_reason = choices[0].get("finish_reason")
                yield t_recv, delta_text, usage, finish_reason


def percentile(values: list, p: float):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# --------------------------------------------------------------------------
# Prefix-cache metrics scraping
# --------------------------------------------------------------------------

# Matches lines like: vllm:prefix_cache_queries{engine="0",model_name="..."} 123.0
# Deliberately tolerant of label sets (order/presence varies by vllm version)
# so this doesn't break on a metrics-format point release.
_METRIC_LINE_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(?P<value>[-\d.eE+]+)\s*$')

PREFIX_CACHE_METRIC_NAMES = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
)

# vllm 0.26.0 renamed the counters with a _total suffix (Prometheus counter
# convention); TASK H found the bare names silently scrape to zero there.
# Each _total series is folded into its bare-name bucket so the rest of the
# script keeps a single canonical key per metric.
_METRIC_NAME_ALIASES = {
    name + "_total": name for name in PREFIX_CACHE_METRIC_NAMES
}


def scrape_prefix_cache_metrics() -> dict:
    """Sum every series for each tracked metric name (collapses across
    engine/model_name labels — fine for a single-engine benchmark server).
    Returns {} on any failure (metrics endpoint missing/unreachable) rather
    than crashing the run — cache instrumentation is nice-to-have, not a
    hard requirement for getting speed numbers."""
    totals = {name: 0.0 for name in PREFIX_CACHE_METRIC_NAMES}
    try:
        r = requests.get(METRICS_URL, timeout=10)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"WARNING: could not scrape {METRICS_URL}: {e}", file=sys.stderr)
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


def diff_prefix_cache_metrics(before: dict, after: dict) -> dict:
    if not before or not after:
        return {}
    return {name: after.get(name, 0.0) - before.get(name, 0.0) for name in PREFIX_CACHE_METRIC_NAMES}


# --------------------------------------------------------------------------
# Per-request runner
# --------------------------------------------------------------------------

def run_one_request(qid, messages: list, max_tokens: int) -> dict:
    record = {"id": qid, "ok": False, "error": None}
    t_start = time.monotonic()
    token_times = []
    text_parts = []
    last_usage = None
    finish_reason = None
    try:
        for t_recv, delta_text, usage, fr in stream_chat_completion(messages, max_tokens):
            if usage:
                last_usage = usage
            if fr:
                finish_reason = fr
            if delta_text:
                token_times.append(t_recv)
                text_parts.append(delta_text)
        t_end = time.monotonic()
        text = "".join(text_parts)
        completion_tokens = last_usage.get("completion_tokens") if last_usage else None
        if completion_tokens is None:
            completion_tokens = len(token_times)
        e2e_s = t_end - t_start
        ttft_s = (token_times[0] - t_start) if token_times else None
        decode_s = (t_end - token_times[0]) if token_times else None
        decode_tok_s = (completion_tokens / decode_s) if decode_s and decode_s > 0 else None

        record.update(
            ok=True,
            ttft_s=ttft_s,
            e2e_s=e2e_s,
            decode_tok_s=decode_tok_s,
            prompt_tokens=last_usage.get("prompt_tokens") if last_usage else None,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            output_text=text,
        )
    except requests.exceptions.RequestException as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["e2e_s"] = time.monotonic() - t_start
    except Exception as e:  # never let one bad request crash the whole sweep
        record["error"] = f"{type(e).__name__}: {e}"
        record["e2e_s"] = time.monotonic() - t_start
    return record


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def summarize_level(concurrency: int, mode: str, records: list, wall_elapsed: float,
                     cache_delta: dict) -> dict:
    ok = [r for r in records if r.get("ok")]
    errors = [r for r in records if not r.get("ok")]

    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    decode_rates = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s") is not None]
    total_completion_tokens = sum(r.get("completion_tokens") or 0 for r in ok)

    queries_delta = cache_delta.get("vllm:prefix_cache_queries")
    hits_delta = cache_delta.get("vllm:prefix_cache_hits")
    hit_rate = (hits_delta / queries_delta) if queries_delta else None

    return {
        "type": "summary",
        "concurrency": concurrency,
        "mode": mode,
        "n_requests": len(records),
        "n_ok": len(ok),
        "n_errors": len(errors),
        "wall_elapsed_s": wall_elapsed,
        "ttft_p50": percentile(ttfts, 0.5),
        "ttft_p95": percentile(ttfts, 0.95),
        "mean_decode_tok_s": statistics.mean(decode_rates) if decode_rates else None,
        "total_throughput_tok_s": total_completion_tokens / wall_elapsed if wall_elapsed > 0 else 0.0,
        "prefix_cache_queries_delta": queries_delta,
        "prefix_cache_hits_delta": hits_delta,
        "prefix_cache_hit_rate": hit_rate,
        "prefix_cache_metrics_raw_delta": cache_delta,
    }


# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------

def run_level(questions: list, prefix: str, max_tokens: int, concurrency: int, mode: str) -> tuple:
    def task(q):
        return run_one_request(q["id"], build_messages(prefix, q), max_tokens)

    before = scrape_prefix_cache_metrics()
    t0 = time.monotonic()

    if mode == "sequential":
        records = [task(q) for q in questions]
    else:  # concurrent
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            records = list(pool.map(task, questions))

    wall_elapsed = time.monotonic() - t0
    after = scrape_prefix_cache_metrics()
    cache_delta = diff_prefix_cache_metrics(before, after)

    summary = summarize_level(concurrency, mode, records, wall_elapsed, cache_delta)
    return records, summary


def run_warmup(questions: list, prefix: str, max_tokens: int) -> dict:
    print("warmup (cold prefix-compute cost, excluded from aggregates) ...", flush=True)
    rec = run_one_request(questions[0]["id"], build_messages(prefix, questions[0]), max_tokens)
    rec["warmup"] = True
    if rec["ok"]:
        print(f"  warmup TTFT={rec['ttft_s']:.3f}s prompt_tokens={rec.get('prompt_tokens')}\n", flush=True)
    else:
        print(f"  WARMUP FAILED: {rec['error']}\n", flush=True)
    return rec


# --------------------------------------------------------------------------
# Dry run (no server contact): validate inputs + one request payload
# --------------------------------------------------------------------------

def dry_run(prefix: str, questions: list, max_tokens: int, synthetic_mode: dict = None) -> None:
    q0 = questions[0]
    messages = build_messages(prefix, q0)
    # ~4 chars/token: crude estimate, matches bench_serving.py's convention
    # for environments without a real tokenizer available.
    prefix_tok_est = len(prefix) // 4
    user_tok_est = len(messages[1]["content"]) // 4
    print("=== dry run (no server contact) ===")
    if synthetic_mode:
        print(f"SYNTHETIC MODE: prefix ~{synthetic_mode['prefix_tokens']} tokens (est.), "
              f"{synthetic_mode['n_questions']} questions, seed={synthetic_mode['seed']}")
    print(f"prefix: {len(prefix)} chars, ~{prefix_tok_est} tokens (est.)")
    print(f"questions loaded: {len(questions)}")
    print(f"first question id: {q0['id']}")
    print(f"first user-turn: {len(messages[1]['content'])} chars, ~{user_tok_est} tokens (est.)")
    print(f"message roles: {[m['role'] for m in messages]}")
    print(f"max_tokens={max_tokens} temperature=0 stream=True")
    print("payload constructed OK, exiting (--dry-run).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_concurrency(spec: str) -> list:
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Shared-prefix (skills-pack) benchmark for an OpenAI-compatible vLLM endpoint. "
        "Dataset-free: content comes entirely from --prefix-file/--questions-file or synthetic mode."
    )
    ap.add_argument("--prefix-file", default=None, help="path to the system-prompt text file (mutually exclusive with --synthetic-prefix-tokens)")
    ap.add_argument("--questions-file", default=None, help="path to a JSONL file with id/question/choices/domain fields (mutually exclusive with --synthetic-questions)")
    ap.add_argument("--synthetic-prefix-tokens", type=int, default=None, help="generate synthetic prefix of ~N tokens instead of reading --prefix-file")
    ap.add_argument("--synthetic-questions", type=int, default=None, help="generate N synthetic questions instead of reading --questions-file")
    ap.add_argument("--synthetic-question-tokens", default="50-150", help="token range per synthetic question (min-max, default 50-150)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for synthetic data generation (default 0, ensures deterministic output)")
    ap.add_argument("--model", default=None, help="overrides env VLLM_MODEL")
    ap.add_argument("--url", default=None, help="overrides env VLLM_URL (default http://localhost:8000)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=parse_concurrency, default=[1, 8, 16, 32], help="comma-separated concurrency levels, e.g. 1,8,16,32")
    ap.add_argument("--mode", choices=["sequential", "concurrent"], default="concurrent")
    ap.add_argument("--output", default=None, help="output JSONL path (default: out_skills/bench_skills_<ts>.jsonl)")
    ap.add_argument("--timeout", type=float, default=None, help="per-chunk read timeout seconds (env BENCH_TIMEOUT)")
    ap.add_argument("--dry-run", action="store_true", help="load/generate data, build the first request payload, print token estimates, exit — no server contact")
    args = ap.parse_args()

    # Validate synthetic vs file mode
    has_prefix_file = args.prefix_file is not None
    has_questions_file = args.questions_file is not None
    has_synthetic_prefix = args.synthetic_prefix_tokens is not None
    has_synthetic_questions = args.synthetic_questions is not None

    if (has_prefix_file or has_questions_file) and (has_synthetic_prefix or has_synthetic_questions):
        print("ERROR: cannot mix file-based (--prefix-file/--questions-file) and synthetic (--synthetic-*) modes",
              file=sys.stderr)
        sys.exit(1)

    # Check that either both files or both synthetic flags are provided
    if has_prefix_file != has_questions_file:
        print("ERROR: must provide both --prefix-file and --questions-file together (or use synthetic mode)",
              file=sys.stderr)
        sys.exit(1)

    if has_synthetic_prefix != has_synthetic_questions:
        print("ERROR: must provide both --synthetic-prefix-tokens and --synthetic-questions together (or use file mode)",
              file=sys.stderr)
        sys.exit(1)

    if not has_prefix_file and not has_synthetic_prefix:
        print("ERROR: must specify either (--prefix-file and --questions-file) or (--synthetic-prefix-tokens and --synthetic-questions)",
              file=sys.stderr)
        sys.exit(1)

    # Parse question token range
    if args.synthetic_question_tokens:
        parts = args.synthetic_question_tokens.split("-")
        if len(parts) != 2:
            print(f"ERROR: --synthetic-question-tokens must be MIN-MAX (got {args.synthetic_question_tokens})",
                  file=sys.stderr)
            sys.exit(1)
        try:
            min_q_tokens = int(parts[0])
            max_q_tokens = int(parts[1])
        except ValueError:
            print(f"ERROR: --synthetic-question-tokens must be numeric MIN-MAX",
                  file=sys.stderr)
            sys.exit(1)
    else:
        min_q_tokens = 50
        max_q_tokens = 150

    # Load or generate prefix and questions
    synthetic_mode = None
    if args.synthetic_prefix_tokens is not None or args.synthetic_questions is not None:
        synthetic_mode = {}
        if args.synthetic_prefix_tokens is not None:
            prefix = generate_synthetic_prefix(args.synthetic_prefix_tokens, seed=args.seed)
            synthetic_mode['prefix_tokens'] = args.synthetic_prefix_tokens
        else:
            # Need to load prefix file
            prefix = load_prefix(args.prefix_file)

        if args.synthetic_questions is not None:
            questions = generate_synthetic_questions(args.synthetic_questions, min_q_tokens, max_q_tokens, seed=args.seed)
            synthetic_mode['n_questions'] = args.synthetic_questions
        else:
            # Need to load questions file
            questions = load_questions(args.questions_file)

        synthetic_mode['seed'] = args.seed
    else:
        prefix = load_prefix(args.prefix_file)
        questions = load_questions(args.questions_file)

    if args.dry_run:
        dry_run(prefix, questions, args.max_tokens, synthetic_mode)
        return

    global MODEL, URL, CHAT_URL, METRICS_URL, READ_TIMEOUT
    if args.model:
        MODEL = args.model
    if not MODEL:
        print("ERROR: set VLLM_MODEL or pass --model", file=sys.stderr)
        sys.exit(1)
    if args.url:
        URL = args.url.rstrip("/")
        CHAT_URL = URL + "/v1/chat/completions"
        METRICS_URL = URL + "/metrics"
    if args.timeout:
        READ_TIMEOUT = args.timeout

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else REPO_ROOT / "out_skills" / f"bench_skills_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if synthetic_mode:
        prefix_tok_est = synthetic_mode.get('prefix_tokens', len(prefix) // 4)
        n_q = synthetic_mode.get('n_questions', len(questions))
        seed = synthetic_mode.get('seed', 0)
        print(f"=== skills-prefix benchmark (SYNTHETIC MODE) ===")
        print(f"prefix ~{prefix_tok_est} tokens, {n_q} questions, seed={seed}")
        print(f"model={MODEL} mode={args.mode} concurrency={args.concurrency}")
        print(f"WARNING: synthetic mode measures throughput only, NOT quality. "
              "Do not use these results to evaluate model capabilities.\n")
    else:
        print(f"=== skills-prefix benchmark: model={MODEL} questions={len(questions)} mode={args.mode} concurrency={args.concurrency} ===\n")

    all_summaries = []
    with out_path.open("w", encoding="utf-8") as f:
        # Write benchmark metadata at the start
        if synthetic_mode:
            metadata = {
                "type": "metadata",
                "mode": "synthetic",
                "prefix_tokens": synthetic_mode.get('prefix_tokens'),
                "n_questions": synthetic_mode.get('n_questions'),
                "seed": synthetic_mode.get('seed', 0),
            }
            f.write(json.dumps(metadata, default=str) + "\n")

        warmup_rec = run_warmup(questions, prefix, args.max_tokens)
        f.write(json.dumps(warmup_rec, default=str) + "\n")

        for c in args.concurrency:
            print(f"--- concurrency={c} mode={args.mode} ---", flush=True)
            records, summary = run_level(questions, prefix, args.max_tokens, c, args.mode)
            for r in records:
                r["concurrency_level"] = c
                f.write(json.dumps(r, default=str) + "\n")
            f.write(json.dumps(summary, default=str) + "\n")
            f.flush()
            all_summaries.append(summary)

            print(
                f"  ok={summary['n_ok']}/{summary['n_requests']} "
                f"ttft_p50={summary['ttft_p50']} ttft_p95={summary['ttft_p95']} "
                f"mean_decode_tok_s={summary['mean_decode_tok_s']} "
                f"throughput={summary['total_throughput_tok_s']:.1f} tok/s "
                f"cache_hit_rate={summary['prefix_cache_hit_rate']}",
                flush=True,
            )

    print(f"\nresults -> {out_path}")
    print("(ids only above; no dataset content printed to stdout)")


if __name__ == "__main__":
    main()
