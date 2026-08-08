"""Quality gate for quantized models: perplexity of SWE-bench gold patches.

Rationale: quantization damage (RTN clipping, K-quant block error, botched
tensor mapping — this repo has hit all three) rarely shows up as outright
garbage output; it shows up as the model becoming *less confident* in
sequences it should nail. A gold patch from SWE-bench is a known-good
continuation of a real issue: a healthy model assigns it high likelihood.
Feed the same (problem_statement, patch) pairs through the served model and
measure negative log-likelihood (NLL) of the patch tokens under the prompt —
a damaged quant shows up as higher NLL / higher perplexity on text it has no
business finding surprising. This needs no human grading, no held-out labels,
and no generation sampling (hence no seed/temperature noise) — just one
forward pass per instance — so it runs in minutes against a live server and
is cheap enough to run on every quant/config change as a gate.

Gate thresholds (see `compare`): perplexity ratio candidate/baseline
  < 1.10           PASS  (noise-level drift)
  1.10 - 1.25       WARN  (real but survivable degradation, needs a look)
  >= 1.25           FAIL  (quant broke something — do not ship)
These are starting points, not physical constants; override with
--warn-threshold/--fail-threshold if a given model family needs different
margins.

Scoring mechanism (vLLM OpenAI-compatible /v1/completions):
  prompt = f"Issue:\n{problem_statement}\n\nFix patch:\n{patch}"
  POST /v1/completions {prompt, max_tokens: 1, temperature: 0,
                         "prompt_logprobs": 0}
max_tokens=1 (not 0) because some vLLM builds reject a zero-token request;
the one generated token is discarded, only `prompt_logprobs` is used. This
returns, for every PROMPT token, its own logprob under the model — exactly
the per-token NLL of the fixed reference text, no sampling involved. We then
need to know where in that list the patch region starts (we only want to
score the patch, not the issue text before it). Two methods, in preference
order (`--boundary-method auto` picks the first that works, logged in the
output as `boundary_method`):
  1. tokenize_endpoint  - POST /tokenize with just the prefix
                           ("Issue:\n...\n\nFix patch:\n") and take the
                           returned token count as the boundary index.
  2. prompt_logprobs_prefix_probe - not every server build exposes
                           /tokenize; fall back to requesting
                           prompt_logprobs for the prefix ALONE and using
                           the length of that list as the boundary index.
                           Costs one extra request per instance.
Both are an approximation: BPE merges can tokenize differently right at the
prefix/patch seam depending on what follows, so the boundary can be off by a
token or two. That's noise on the scale of hundreds of patch tokens per
instance and doesn't matter for a relative (baseline vs candidate) gate.

If the server doesn't return `prompt_logprobs` at all (some quant configs /
older server builds), this script fails loudly with a clear error rather
than silently reporting all-zero NLL — a quality gate that can go quiet on
its own precondition is worse than no gate.

Usage:
    VLLM_MODEL=repo:QUANT python scripts/eval_quality_swebench.py run --num-instances 100 --output out/quality_q4_k_m.json
    python scripts/eval_quality_swebench.py compare out/quality_fp16.json out/quality_q4_k_m.json
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache"
OUT_DIR = REPO_ROOT / "out"
DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
DATASET_SPLIT = "test"

# Mutated by main() from CLI args / env; module-level like the other bench
# scripts so the request helpers below don't need these threaded through.
MODEL = os.environ.get("VLLM_MODEL", "")
BASE_URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "300"))  # generous: prompt_logprobs over a long prompt is a big forward pass


# --------------------------------------------------------------------------
# Dataset: SWE-bench_Lite, truncated + cached (pattern copied from
# bench_serving.py's load_prompts)
# --------------------------------------------------------------------------

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
    — this script needs to run in GPU-less, vllm-less environments too."""
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


def _estimate_tokens(text: str, tokenizer_name: str | None) -> int:
    """Cheap token-count estimate used only for the --max-patch-tokens skip
    decision, not for scoring itself (scoring uses the server's own
    tokenizer via prompt_logprobs, which is authoritative)."""
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            return len(tok.encode(text, add_special_tokens=False))
        except ImportError:
            pass
    return len(text) // 4


def load_swebench_lite(num_instances: int, token_budget: int, tokenizer_name: str | None) -> list:
    """Load SWE-bench_Lite (test split), truncate `problem_statement` to
    `token_budget`, and cache the *result* to a local JSONL keyed by
    budget + tokenizer + count so a different config can't silently reuse a
    stale cache (same pattern as bench_serving.py's load_prompts).
    """
    tok_tag = tokenizer_name.replace("/", "_") if tokenizer_name else "chars4"
    cache_file = CACHE_DIR / f"swebench_lite_budget{token_budget}_{tok_tag}_n{num_instances}.jsonl"

    if not cache_file.exists():
        from datasets import load_dataset  # imported lazily: only needed on first run

        print(f"Downloading {DATASET_NAME} ({DATASET_SPLIT} split) -> {cache_file} ...", file=sys.stderr)
        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

        truncate = (
            _make_tokenizer_truncator(tokenizer_name, token_budget)
            if tokenizer_name
            else _make_char_truncator(token_budget)
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        n_written = 0
        with cache_file.open("w", encoding="utf-8") as f:
            for row in ds:
                problem_statement = row.get("problem_statement") or ""
                patch = row.get("patch") or ""
                if not problem_statement or not patch:
                    continue
                f.write(
                    json.dumps(
                        {
                            "instance_id": row.get("instance_id"),
                            "problem_statement": truncate(problem_statement),
                            "patch": patch,
                        }
                    )
                    + "\n"
                )
                n_written += 1
                if n_written >= num_instances:
                    break

    rows = []
    with cache_file.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# Boundary determination: where does the patch region start in the
# prompt_logprobs list?
# --------------------------------------------------------------------------

def probe_boundary_method(preferred: str) -> str:
    """Decide once, up front, how patch-region boundaries will be computed
    for the whole run, and print which method was chosen (this matters for
    interpreting results — see module docstring)."""
    if preferred == "prompt_logprobs_prefix_probe":
        return preferred
    if preferred in ("auto", "tokenize_endpoint"):
        try:
            r = requests.post(
                f"{BASE_URL}/tokenize",
                json={"model": MODEL, "prompt": "boundary probe"},
                timeout=(CONNECT_TIMEOUT, 30.0),
            )
            r.raise_for_status()
            data = r.json()
            if "count" in data or "tokens" in data:
                return "tokenize_endpoint"
        except requests.exceptions.RequestException:
            pass
        if preferred == "tokenize_endpoint":
            print("ERROR: --boundary-method tokenize_endpoint requested but /tokenize is unavailable", file=sys.stderr)
            sys.exit(1)
    return "prompt_logprobs_prefix_probe"


def _tokenize_count(text: str) -> int:
    r = requests.post(
        f"{BASE_URL}/tokenize",
        json={"model": MODEL, "prompt": text},
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    r.raise_for_status()
    data = r.json()
    if "count" in data:
        return int(data["count"])
    return len(data["tokens"])


def _prompt_logprobs_count(text: str) -> int:
    """Fallback boundary probe: ask the server for prompt_logprobs of the
    prefix ALONE and use the length of the returned list as the token
    count. Costs one extra request per instance versus the tokenize-endpoint
    method."""
    plogprobs = request_prompt_logprobs(text)
    return len(plogprobs)


def get_prefix_boundary(prefix: str, method: str) -> int:
    if method == "tokenize_endpoint":
        return _tokenize_count(prefix)
    return _prompt_logprobs_count(prefix)


# --------------------------------------------------------------------------
# Scoring: prompt_logprobs over the full (prefix + patch) text
# --------------------------------------------------------------------------

def request_prompt_logprobs(prompt: str) -> list:
    """POST /v1/completions with max_tokens=1 and prompt_logprobs=0, return
    the raw prompt_logprobs list (one entry per prompt token; the first
    entry is always None since there's nothing to condition it on).

    Fails loudly (raises) if the server doesn't return prompt_logprobs at
    all — some quant configs / server builds don't support it, and silently
    treating that as "zero NLL" would make a broken gate look healthy.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 0,
    }
    r = requests.post(
        f"{BASE_URL}/v1/completions", json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"server response had no choices: {data}")
    plogprobs = choices[0].get("prompt_logprobs")
    if plogprobs is None:
        raise RuntimeError(
            "server did not return `prompt_logprobs` for this request — this server/quant "
            "config does not support the prompt_logprobs field this quality gate depends on. "
            "Refusing to silently report a fake (zero) score; fix the server config or drop "
            "this gate for this config."
        )
    return plogprobs


def _entry_logprob(entry) -> float | None:
    """Each non-null prompt_logprobs entry is a dict {token_id_str: {"logprob":
    float, "rank": int, "decoded_token": str, ...}} with exactly one key
    since we requested prompt_logprobs=0 (no extra top-k alternatives)."""
    if entry is None:
        return None
    val = next(iter(entry.values()))
    if isinstance(val, dict):
        return val.get("logprob")
    return float(val)


def score_instance(row: dict, boundary_method: str) -> dict:
    """Score one SWE-bench instance: NLL sum + token count over the PATCH
    region only (the issue text before it is context, not scored)."""
    prefix = f"Issue:\n{row['problem_statement']}\n\nFix patch:\n"
    full_prompt = prefix + row["patch"]

    prefix_len = get_prefix_boundary(prefix, boundary_method)
    plogprobs = request_prompt_logprobs(full_prompt)

    if prefix_len >= len(plogprobs):
        raise RuntimeError(
            f"boundary probe ({prefix_len} tokens) covers the entire scored prompt "
            f"({len(plogprobs)} tokens) — empty patch region, instance {row.get('instance_id')}"
        )

    patch_entries = plogprobs[prefix_len:]
    nll_sum = 0.0
    n_tokens = 0
    for entry in patch_entries:
        lp = _entry_logprob(entry)
        if lp is None:  # first-token-of-prompt slot has no logprob; skip, don't zero-fill
            continue
        nll_sum += -lp
        n_tokens += 1

    if n_tokens == 0:
        raise RuntimeError(f"no scorable patch tokens for instance {row.get('instance_id')}")

    return {
        "instance_id": row.get("instance_id"),
        "prefix_tokens": prefix_len,
        "total_prompt_tokens": len(plogprobs),
        "patch_tokens": n_tokens,
        "nll_sum": nll_sum,
        "nll_per_token": nll_sum / n_tokens,
    }


# --------------------------------------------------------------------------
# Mode: run
# --------------------------------------------------------------------------

def mode_run(args) -> None:
    global MODEL, BASE_URL
    if args.model:
        MODEL = args.model
    if not MODEL:
        print("ERROR: set VLLM_MODEL or pass --model", file=sys.stderr)
        sys.exit(1)
    if args.url:
        BASE_URL = args.url.rstrip("/")

    rows = load_swebench_lite(args.num_instances, args.token_budget, args.tokenizer)
    if not rows:
        print("ERROR: no SWE-bench_Lite rows loaded", file=sys.stderr)
        sys.exit(1)
    print(f"loaded {len(rows)} SWE-bench_Lite instances (problem_statement budget={args.token_budget})")

    kept = []
    n_skipped = 0
    for row in rows:
        if _estimate_tokens(row["patch"], args.tokenizer) > args.max_patch_tokens:
            n_skipped += 1
            continue
        kept.append(row)
    print(f"{len(kept)}/{len(rows)} instances kept after --max-patch-tokens={args.max_patch_tokens} filter ({n_skipped} skipped)")
    if not kept:
        print("ERROR: every instance's gold patch exceeded --max-patch-tokens", file=sys.stderr)
        sys.exit(1)

    boundary_method = probe_boundary_method(args.boundary_method)
    print(f"boundary method: {boundary_method}")

    # Preflight: fail loudly and immediately if the server doesn't support
    # prompt_logprobs, rather than burning the whole sweep and reporting a
    # pile of per-instance errors that all say the same thing.
    print("preflight: checking server returns prompt_logprobs ...", flush=True)
    try:
        request_prompt_logprobs(f"Issue:\n{kept[0]['problem_statement'][:200]}\n\nFix patch:\n{kept[0]['patch'][:200]}")
    except Exception as e:
        print(f"ERROR: preflight failed — {e}", file=sys.stderr)
        sys.exit(1)
    print("preflight OK\n", flush=True)

    print(f"=== quality gate: model={MODEL} n={len(kept)} concurrency={args.concurrency} ===\n")
    t0 = time.monotonic()
    results = [None] * len(kept)
    errors = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(score_instance, row, boundary_method): i for i, row in enumerate(kept)}
        n_done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            n_done += 1
            try:
                results[i] = fut.result()
            except Exception as e:
                errors.append(f"{kept[i].get('instance_id')}: {type(e).__name__}: {e}")
            if n_done % 10 == 0 or n_done == len(kept):
                print(f"  {n_done}/{len(kept)} scored ({len(errors)} errors)", flush=True)
    elapsed = time.monotonic() - t0

    ok = [r for r in results if r is not None]
    if errors:
        print(f"\n{len(errors)} instance(s) failed to score:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
    if not ok:
        print("ERROR: no instances scored successfully", file=sys.stderr)
        sys.exit(1)

    total_nll = sum(r["nll_sum"] for r in ok)
    total_tokens = sum(r["patch_tokens"] for r in ok)
    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)

    output = {
        "model": MODEL,
        "dataset": f"{DATASET_NAME}:{DATASET_SPLIT}",
        "num_instances": len(ok),
        "num_errors": len(errors),
        "boundary_method": boundary_method,
        "patch_tokens_total": total_tokens,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "elapsed_s": elapsed,
        "per_instance": ok,
    }

    print(f"\n=== result: model={MODEL} ===")
    print(f"instances scored : {len(ok)}/{len(kept)}")
    print(f"patch tokens total: {total_tokens}")
    print(f"mean NLL/token    : {mean_nll:.4f}")
    print(f"perplexity        : {ppl:.4f}")

    print("\n" + json.dumps(output, indent=2, default=str))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n-> {out_path}")


# --------------------------------------------------------------------------
# Mode: compare
# --------------------------------------------------------------------------

def mode_compare(args) -> None:
    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        candidate = json.load(f)

    base_ppl = baseline["ppl"]
    cand_ppl = candidate["ppl"]
    ratio = cand_ppl / base_ppl

    if ratio < args.warn_threshold:
        verdict = "PASS"
    elif ratio < args.fail_threshold:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    print(f"baseline : {baseline.get('model')}  ppl={base_ppl:.4f}  ({baseline.get('num_instances')} instances)")
    print(f"candidate: {candidate.get('model')}  ppl={cand_ppl:.4f}  ({candidate.get('num_instances')} instances)")
    print(f"ppl ratio (candidate/baseline): {ratio:.4f}")
    print(
        f"thresholds: PASS < {args.warn_threshold}  WARN [{args.warn_threshold}, {args.fail_threshold})  "
        f"FAIL >= {args.fail_threshold}"
    )
    print(f"verdict: {verdict}")

    if verdict == "FAIL":
        sys.exit(1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Quality gate for quantized models: perplexity of SWE-bench gold patches under a served vLLM model"
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    run_ap = sub.add_parser("run", help="score gold patches against a live server")
    run_ap.add_argument("--num-instances", type=int, default=100, help="SWE-bench_Lite instances to load/score")
    run_ap.add_argument("--token-budget", type=int, default=2048, help="truncate problem_statement to this many tokens")
    run_ap.add_argument("--max-patch-tokens", type=int, default=1024, help="skip instances whose gold patch exceeds this many tokens")
    run_ap.add_argument("--tokenizer", default=None, help="tokenizer name for exact token counting (needs `transformers`); default: char-count estimate (~4 chars/token)")
    run_ap.add_argument(
        "--boundary-method",
        choices=["auto", "tokenize_endpoint", "prompt_logprobs_prefix_probe"],
        default="auto",
        help="how to locate the patch region in the scored token stream (default: auto-probe /tokenize, fall back to prompt_logprobs)",
    )
    run_ap.add_argument("--concurrency", type=int, default=4, help="parallel scoring requests")
    run_ap.add_argument("--model", default=None, help="overrides env VLLM_MODEL")
    run_ap.add_argument("--url", default=None, help="overrides env VLLM_URL (default http://localhost:8000)")
    run_ap.add_argument("--output", default=None, help="also write the result JSON to this path")

    cmp_ap = sub.add_parser("compare", help="compare two run outputs and print a PASS/WARN/FAIL verdict")
    cmp_ap.add_argument("baseline", help="baseline result JSON (e.g. fp16)")
    cmp_ap.add_argument("candidate", help="candidate result JSON (e.g. a quant under test)")
    cmp_ap.add_argument("--warn-threshold", type=float, default=1.10, help="ppl ratio below this is PASS")
    cmp_ap.add_argument("--fail-threshold", type=float, default=1.25, help="ppl ratio at/above this is FAIL; between warn and fail is WARN")

    args = ap.parse_args()

    if args.mode == "compare":
        mode_compare(args)
    else:
        mode_run(args)


if __name__ == "__main__":
    main()
