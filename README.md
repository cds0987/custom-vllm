# custom_vllm

## Serving benchmark (scripts/bench_swebench.py)

SWE-bench-driven benchmark for an OpenAI-compatible vLLM endpoint (streaming
`/v1/chat/completions`). Supersedes `scripts/bench_load.py`. Prompts are built
from real SWE-bench `problem_statement`/`hints_text` text (dev split, cached
locally to `.cache/swebench_dev.jsonl` after the first run) so prefill cost
is representative — short synthetic prompts hide TTFT entirely.

Env vars (all optional, all overridable by flags):

| var | default | meaning |
|---|---|---|
| `VLLM_MODEL` | *(required)* | model name sent in the request body |
| `VLLM_URL` | `http://localhost:8000` | base URL of the server |
| `SWEBENCH_LIMIT` | `20` | number of SWE-bench instances to load/use |
| `BENCH_MAX_TOKENS` | `256` | `max_tokens` per request |
| `BENCH_TIMEOUT` | `300` | per-chunk stream read timeout (seconds) |

Both modes: do one warmup request first (excluded from stats, since a cold
vLLM+Triton server pays a large autotune cost that would otherwise dominate),
print a compact aligned table, and write raw per-request JSON records to
`out/bench_swebench_<mode>_<timestamp>.jsonl` (or `--out <path>`) for
run-to-run comparison. Every response is checked for degenerate output (empty,
or collapsed onto one repeated token) since a fast-but-broken config should
never look like a win.

### Mode 1: single-request throughput

Sequential, one request at a time — "how fast does *my* answer stream."
Reports prompt tokens, TTFT, decode tok/s (excluding TTFT), and end-to-end
latency per request, with p50/p95 over N instances.

```bash
VLLM_MODEL=repo:QUANT python scripts/bench_swebench.py single --n 20 --max-tokens 256
```

### Mode 2: concurrent serving fairness

Fires C requests at once over a sweep of concurrency levels. Reports fleet
throughput, TTFT/ITL percentiles, and a fairness view: min/p50/max per-request
decode rate, Jain's fairness index (1.0 = perfectly even), and the max gap
between concurrent streams' completion times — i.e. whether tokens are spread
evenly or some requests starve while others race ahead.

```bash
VLLM_MODEL=repo:QUANT python scripts/bench_swebench.py concurrent --concurrency 1 2 4 8 16 32
```

Out of scope: real SWE-bench patch evaluation (needs repo checkouts + test
runs) — this only measures serving speed and a lightweight non-degeneracy
signal, not solve rate.
