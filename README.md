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

## Open-loop rate benchmark (scripts/bench_serving.py)

`bench_swebench.py concurrent` is a **closed-loop** sweep: N workers each
send a request and wait for it before sending the next one. That hides
saturation — the moment the server slows down, the offered load
automatically shrinks (workers are just stuck waiting), so a server that's
falling over can still look fine.

`bench_serving.py` is **open-loop**: it fixes an arrival rate in
queries/second and fires requests on that schedule regardless of whether
earlier ones have finished — what a production traffic source actually
does. Interarrival gaps are drawn from an exponential distribution (a
Poisson process), not a fixed metronome, since real traffic is bursty and a
uniform interval understates tail latency. Requests are dispatched from
daemon threads so a slow/stuck request never throttles the arrival
schedule; an `--max-inflight` cap bounds client-side resource usage, and
anything rejected by that cap is counted as **dropped**, never silently
discarded from the stats.

Prompts come from `zai-org/LongAlign-10k` (long real documents, ~60k
chars/row) truncated to `--prompt-token-budget` (default 12000, leaving
headroom under a 16K context) and cached to `.cache/` so repeated runs are
byte-identical and comparable. Truncation is by a char/token estimate
unless `--tokenizer <name>` is given and `transformers` is installed, in
which case it tokenizes with the real served-model tokenizer.

```bash
VLLM_MODEL=repo:QUANT python scripts/bench_serving.py run \
  --rates 3 5 7 9 12 30 --duration 60 --label q4_k_m

# compare several configs (e.g. GGUF quants) side by side
python scripts/bench_serving.py compare out/bench_serving_summary_*.json
```

### Reading the saturation signal

Each rate level reports offered vs. **achieved** QPS, TTFT/ITL/e2e-latency
percentiles, output and prefill token throughput (reported separately —
with 12k-token prompts prefill dominates and lumping them together hides
that), and in-flight concurrency (mean/max, derived from each request's
arrival/completion timestamps). A rate level is flagged `SATURATED` when
either is true:

- **achieved QPS falls meaningfully below offered QPS** — the server can no
  longer drain requests as fast as they arrive, so they queue up faster
  than they complete;
- **in-flight concurrency grows monotonically** across the run (checked by
  comparing mean concurrency across the first/middle/last third) — a
  queue that only ever gets longer, even if throughput hasn't visibly
  dropped yet.

The rate just below the first `SATURATED` level is the server's real
sustainable throughput. Every response is also checked for the repeated-
single-token degeneracy bug — a fast-but-broken quant should never look
like a win just because its QPS number is high.

Raw per-request records (with absolute arrival/first-token/completion
timestamps, for replotting) go to
`out/bench_serving_<label>_<model>_rate<qps>_<ts>.jsonl`; the per-rate
summary table goes to `out/bench_serving_summary_<label>_<model>_<ts>.json`.
