"""Build mixed, real-world agent-loop workloads for bench_agent_loop.py (or any
harness that consumes the JSONL schema below) out of THREE real sources:

  bfcl         Berkeley Function Calling Leaderboard (gorilla-llm/Berkeley-
               Function-Calling-Leaderboard on HuggingFace) — multi-turn tool-
               calling sessions, PLUS single-turn "live"/"exec" tool-calling
               and (important) "irrelevance" sessions where the *correct*
               model behavior is to refuse to call any of the offered tools.
  swebench     SWE-bench_Lite gold-patch instances, loaded via the exact same
               cached loader eval_quality_swebench.py already uses
               (`load_swebench_lite`) — imported, not reimplemented.
  public-test  This repo's public-test.jsonl (short Q&A / MCQ questions, no
               tool calling) — gitignored, may be absent on a fresh checkout.

Rationale for mixing (`--source mixed --mix "bfcl:50,public-test:30,swebench:20"`):
a production server never sees one workload shape at a time. It sees short
interactive Q&A, multi-turn agent/tool-calling sessions, and long
code-reasoning jobs all landing concurrently. A benchmark that only replays
one of those undersells (or oversells) real serving behavior — prefix-cache
hit rate, scheduler fairness, and tail latency all depend on the *mix*, not
just each workload type in isolation.

Output: JSONL, one session per line:
    {
      "session_id": str,
      "source": "bfcl" | "swebench" | "public-test",
      "tools": [ {"type": "function", "function": {...OpenAI function schema...}}, ... ],
      "turns": [ {"turn_index": int, "role": "user", "content": str}, ... ],
      "expected": [
        {"turn_index": int, "gt_format": "call_string" | "possible_answer_dict" | "text_patch" | "none",
         "calls": [...]},
        ...
      ],
      "meta": {"approx_tokens": int, "n_turns": int, ...source-specific fields...}
    }

`expected[i].calls` is scored offline by scripts/score_bfcl.py (BFCL sources
only — see that script's docstring for the two BFCL ground-truth formats and
the scorer's limitations). swebench/public-test sessions carry `gt_format`
"text_patch"/"none" for completeness but are not tool-call-scorable; use
eval_quality_swebench.py / manual grading for those.

BFCL data layout expected under --bfcl-dir (default datasets/bfcl/, gitignored
— never commit it). Download with huggingface_hub, no token needed (public
dataset):

    from huggingface_hub import hf_hub_download
    for f in [...]:  # see STATUS.md TASK H follow-up / this script's --help
        hf_hub_download(repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                         repo_type="dataset", filename=f, local_dir="datasets/bfcl")

Required files (all JSON-Lines despite the .json extension — one JSON object
per line, confirmed against the real downloaded files):
    BFCL_v3_multi_turn_{base,composite,long_context,miss_func,miss_param}.json
    BFCL_v3_live_multiple.json
    BFCL_v3_exec_multiple.json
    BFCL_v3_irrelevance.json
    multi_turn_func_doc/{gorilla_file_system,math_api,message_api,posting_api,
                          ticket_api,trading_bot,travel_booking,vehicle_control}.json
    possible_answer/BFCL_v3_multi_turn_{base,composite,long_context,miss_func,miss_param}.json
    possible_answer/BFCL_v3_live_multiple.json
(exec_multiple and irrelevance carry their ground truth / lack thereof inline
— no separate possible_answer file for those two.)

Real BFCL multi-turn schema, as read off the downloaded files (not the docs,
which drift from the actual data):
  question row:  {"id", "question": [[{"role":"user","content":str}], ...],
                   "initial_config": {ClassName: {...}}, "path": [...],
                   "involved_classes": [ClassName, ...]}
                  -- every turn observed in the wild is exactly one user
                  message; there is no assistant/tool text in the raw data
                  (BFCL executes tool calls against a live sandboxed class at
                  eval time, so intermediate tool results are NOT part of the
                  dataset — this converter therefore only emits user turns,
                  which is a real limitation, documented again below).
  possible_answer row: {"id", "ground_truth": [[<call-string>, ...], ...]}
                  -- one list of Python-call-syntax strings per turn, e.g.
                  "cd(folder='document')". A turn's list may be empty (that
                  turn expects zero tool calls).
  live_multiple / irrelevance question row: {"id", "question": [[{"role":
                  "user","content":str}]], "function": [<OpenAI-ish schema>, ...]}
                  -- always one turn in the files sampled.
  live_multiple possible_answer row: {"id", "ground_truth": [{"funcName": {
                  "param": [accepted_value, ...], ...}}, ...]} -- a flat list
                  (one dict per expected parallel call), NOT per-turn nested.
  exec_multiple question row: same as live_multiple's `function`/`question`
                  shape PLUS ground_truth inline (call-string format, flat
                  list, single turn) and `execution_result_type`.
  irrelevance: no ground truth at all (correct behavior: call nothing).

KNOWN LIMITATION (see above): because BFCL's raw data has no assistant/tool
turns, every emitted BFCL session is a sequence of USER turns only, each
carrying its own `expected[i].calls` — a harness driving this workload must
itself decide what (if anything) to feed back as a "tool result" turn; this
converter does not fabricate tool outputs (BFCL's real evaluator runs the
actual sandboxed API implementation to get those, which is out of scope
here).

Usage:
    python scripts/prepare_agent_workload.py --source bfcl --sessions 50 \\
        --output out_agent_workload/bfcl_only.jsonl

    python scripts/prepare_agent_workload.py --source mixed \\
        --mix "bfcl:50,public-test:30,swebench:20" --sessions 200 --seed 0 \\
        --output out_agent_workload/mixed_200.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BFCL_DIR = REPO_ROOT / "datasets" / "bfcl"
DEFAULT_PUBLIC_TEST_FILE = REPO_ROOT / "public-test.jsonl"

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import eval_quality_swebench`

# BFCL class name (as it appears in involved_classes / possible_answer keys)
# -> function-doc filename under multi_turn_func_doc/. Confirmed empirically
# by cross-referencing function names inside each file against `path` entries
# like "TwitterAPI.post_tweet" in the downloaded multi_turn_base.json.
CLASS_FILE_MAP = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

MULTI_TURN_SUBSETS = [
    "BFCL_v3_multi_turn_base",
    "BFCL_v3_multi_turn_composite",
    "BFCL_v3_multi_turn_long_context",
    "BFCL_v3_multi_turn_miss_func",
    "BFCL_v3_multi_turn_miss_param",
]
LIVE_MULTIPLE_SUBSET = "BFCL_v3_live_multiple"
EXEC_MULTIPLE_SUBSET = "BFCL_v3_exec_multiple"
IRRELEVANCE_SUBSET = "BFCL_v3_irrelevance"


class SourceUnavailable(Exception):
    """Raised when a requested source's data isn't present on disk / can't be
    fetched. Caught by mixed-mode so one missing source doesn't kill a run
    that only needed the others; re-raised (uncaught) when a single
    --source is explicitly requested and its data is missing."""


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def estimate_tokens(text: str) -> int:
    # ~4 chars/token: same crude-but-standard estimate used by
    # bench_serving.py / bench_skills.py / eval_quality_swebench.py elsewhere
    # in this repo when no real tokenizer is wired in.
    return max(1, len(text) // 4)


def _session_approx_tokens(tools: list, turns: list) -> int:
    tools_chars = len(json.dumps(tools, ensure_ascii=False)) if tools else 0
    turns_chars = sum(len(t.get("content") or "") for t in turns)
    return estimate_tokens(" " * (tools_chars + turns_chars))


def wrap_openai_tool(fn_schema: dict) -> dict:
    return {"type": "function", "function": fn_schema}


# --------------------------------------------------------------------------
# source: bfcl
# --------------------------------------------------------------------------

def _require_bfcl_file(bfcl_dir: Path, rel: str) -> Path:
    p = bfcl_dir / rel
    if not p.exists():
        raise SourceUnavailable(
            f"bfcl: missing {p} — download it first, e.g.:\n"
            f"  python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='gorilla-llm/Berkeley-Function-Calling-Leaderboard', "
            f"repo_type='dataset', filename='{rel}', local_dir='{bfcl_dir}')\"\n"
            "(see this script's module docstring for the full file list)"
        )
    return p


def _load_func_doc_cache(bfcl_dir: Path) -> dict:
    cache = {}
    for cls, fname in CLASS_FILE_MAP.items():
        p = bfcl_dir / "multi_turn_func_doc" / fname
        if p.exists():
            cache[cls] = _read_jsonl(p)
    return cache


def _tools_for_classes(classes: list, func_doc_cache: dict) -> list:
    tools = []
    seen = set()
    for cls in classes:
        for schema in func_doc_cache.get(cls, []):
            name = schema.get("name")
            if name in seen:
                continue
            seen.add(name)
            tools.append(wrap_openai_tool(schema))
    return tools


def _bfcl_multi_turn_sessions(bfcl_dir: Path, subset: str, func_doc_cache: dict) -> list:
    q_path = _require_bfcl_file(bfcl_dir, f"{subset}.json")
    a_path = _require_bfcl_file(bfcl_dir, f"possible_answer/{subset}.json")
    questions = {row["id"]: row for row in _read_jsonl(q_path)}
    answers = {row["id"]: row for row in _read_jsonl(a_path)}

    sessions = []
    for qid, q in questions.items():
        a = answers.get(qid)
        if a is None:
            continue  # no ground truth for this id; skip rather than emit an unscoreable session
        turns = []
        expected = []
        for i, turn_msgs in enumerate(q["question"]):
            content = "\n".join(m.get("content", "") for m in turn_msgs)
            turns.append({"turn_index": i, "role": "user", "content": content})
            calls = a["ground_truth"][i] if i < len(a["ground_truth"]) else []
            expected.append({"turn_index": i, "gt_format": "call_string", "calls": calls})

        tools = _tools_for_classes(q.get("involved_classes", []), func_doc_cache)
        sessions.append({
            "session_id": f"bfcl:{subset}:{qid}",
            "source": "bfcl",
            "tools": tools,
            "turns": turns,
            "expected": expected,
            "meta": {
                "bfcl_subset": subset,
                "bfcl_id": qid,
                "involved_classes": q.get("involved_classes", []),
                "n_turns": len(turns),
                "approx_tokens": _session_approx_tokens(tools, turns),
            },
        })
    return sessions


def _bfcl_live_multiple_sessions(bfcl_dir: Path) -> list:
    q_path = _require_bfcl_file(bfcl_dir, f"{LIVE_MULTIPLE_SUBSET}.json")
    a_path = _require_bfcl_file(bfcl_dir, f"possible_answer/{LIVE_MULTIPLE_SUBSET}.json")
    answers = {row["id"]: row for row in _read_jsonl(a_path)}

    sessions = []
    for q in _read_jsonl(q_path):
        a = answers.get(q["id"])
        if a is None:
            continue
        turn_msgs = q["question"][0] if q["question"] else []
        content = "\n".join(m.get("content", "") for m in turn_msgs)
        turns = [{"turn_index": 0, "role": "user", "content": content}]
        expected = [{"turn_index": 0, "gt_format": "possible_answer_dict", "calls": a["ground_truth"]}]
        tools = [wrap_openai_tool(fn) for fn in q.get("function", [])]
        sessions.append({
            "session_id": f"bfcl:{LIVE_MULTIPLE_SUBSET}:{q['id']}",
            "source": "bfcl",
            "tools": tools,
            "turns": turns,
            "expected": expected,
            "meta": {
                "bfcl_subset": LIVE_MULTIPLE_SUBSET,
                "bfcl_id": q["id"],
                "n_turns": 1,
                "approx_tokens": _session_approx_tokens(tools, turns),
            },
        })
    return sessions


def _bfcl_exec_multiple_sessions(bfcl_dir: Path) -> list:
    q_path = _require_bfcl_file(bfcl_dir, f"{EXEC_MULTIPLE_SUBSET}.json")
    sessions = []
    for q in _read_jsonl(q_path):
        turn_msgs = q["question"][0] if q["question"] else []
        content = "\n".join(m.get("content", "") for m in turn_msgs)
        turns = [{"turn_index": 0, "role": "user", "content": content}]
        expected = [{"turn_index": 0, "gt_format": "call_string", "calls": q.get("ground_truth", [])}]
        tools = [wrap_openai_tool(fn) for fn in q.get("function", [])]
        sessions.append({
            "session_id": f"bfcl:{EXEC_MULTIPLE_SUBSET}:{q['id']}",
            "source": "bfcl",
            "tools": tools,
            "turns": turns,
            "expected": expected,
            "meta": {
                "bfcl_subset": EXEC_MULTIPLE_SUBSET,
                "bfcl_id": q["id"],
                "n_turns": 1,
                "execution_result_type": q.get("execution_result_type"),
                "approx_tokens": _session_approx_tokens(tools, turns),
            },
        })
    return sessions


def _bfcl_irrelevance_sessions(bfcl_dir: Path) -> list:
    """Sessions where NO tool should be called — the offered functions are
    irrelevant to the user request. Important scenario per the task brief:
    a model that calls a tool anyway here is a false positive, exactly the
    kind of regression 4-bit quantization damage can introduce."""
    q_path = _require_bfcl_file(bfcl_dir, f"{IRRELEVANCE_SUBSET}.json")
    sessions = []
    for q in _read_jsonl(q_path):
        turn_msgs = q["question"][0] if q["question"] else []
        content = "\n".join(m.get("content", "") for m in turn_msgs)
        turns = [{"turn_index": 0, "role": "user", "content": content}]
        expected = [{"turn_index": 0, "gt_format": "none", "calls": []}]
        tools = [wrap_openai_tool(fn) for fn in q.get("function", [])]
        sessions.append({
            "session_id": f"bfcl:{IRRELEVANCE_SUBSET}:{q['id']}",
            "source": "bfcl",
            "tools": tools,
            "turns": turns,
            "expected": expected,
            "meta": {
                "bfcl_subset": IRRELEVANCE_SUBSET,
                "bfcl_id": q["id"],
                "n_turns": 1,
                "expect_refusal": True,
                "approx_tokens": _session_approx_tokens(tools, turns),
            },
        })
    return sessions


def load_bfcl_sessions(bfcl_dir: Path) -> list:
    """Loads every BFCL subset this converter knows about and concatenates
    them into one pool. Each subset is optional: a partial download (e.g.
    only BFCL_v3_multi_turn_base.json present) still produces a usable —
    just smaller — pool, with a warning per missing subset, rather than an
    all-or-nothing failure. Only raises if bfcl_dir itself is missing, or if
    literally every subset failed to load (nothing usable at all)."""
    if not bfcl_dir.exists():
        raise SourceUnavailable(
            f"bfcl: {bfcl_dir} does not exist — download BFCL first (see module docstring)."
        )
    func_doc_cache = _load_func_doc_cache(bfcl_dir)
    sessions = []
    loaders = [(subset, lambda subset=subset: _bfcl_multi_turn_sessions(bfcl_dir, subset, func_doc_cache))
               for subset in MULTI_TURN_SUBSETS]
    loaders += [
        (LIVE_MULTIPLE_SUBSET, lambda: _bfcl_live_multiple_sessions(bfcl_dir)),
        (EXEC_MULTIPLE_SUBSET, lambda: _bfcl_exec_multiple_sessions(bfcl_dir)),
        (IRRELEVANCE_SUBSET, lambda: _bfcl_irrelevance_sessions(bfcl_dir)),
    ]
    for name, loader in loaders:
        try:
            sessions.extend(loader())
        except SourceUnavailable as e:
            print(f"WARNING: bfcl subset {name!r} unavailable, skipping: {e}", file=sys.stderr)
    if not sessions:
        raise SourceUnavailable(f"bfcl: {bfcl_dir} exists but no sessions could be built from it")
    return sessions


# --------------------------------------------------------------------------
# source: public-test (no tool calling; short Q&A/MCQ)
# --------------------------------------------------------------------------

def load_public_test_sessions(path: Path) -> list:
    if not path.exists():
        raise SourceUnavailable(
            f"public-test: {path} not found — this file is gitignored and may simply "
            "not exist on this machine; drop it from --mix or pass --public-test-file."
        )
    rows = _read_jsonl(path)
    if not rows:
        raise SourceUnavailable(f"public-test: {path} is empty")

    sessions = []
    for row in rows:
        question = row.get("question", "")
        choices = row.get("choices")
        content = question
        if choices:
            if isinstance(choices, dict):
                items = sorted(choices.items())
            else:
                letters = [chr(ord("A") + i) for i in range(len(choices))]
                items = list(zip(letters, choices))
            content = question + "\n\n" + "\n".join(f"{l}. {t}" for l, t in items) + "\n\nĐáp án: "
        turns = [{"turn_index": 0, "role": "user", "content": content}]
        expected = [{"turn_index": 0, "gt_format": "none", "calls": []}]
        sessions.append({
            "session_id": f"public-test:{row.get('id')}",
            "source": "public-test",
            "tools": [],
            "turns": turns,
            "expected": expected,
            "meta": {
                "domain": row.get("domain"),
                "n_turns": 1,
                "approx_tokens": _session_approx_tokens([], turns),
            },
        })
    return sessions


# --------------------------------------------------------------------------
# source: swebench (reuses eval_quality_swebench.load_swebench_lite verbatim)
# --------------------------------------------------------------------------

def load_swebench_sessions(num_instances: int, token_budget: int) -> list:
    try:
        import eval_quality_swebench as swebench_mod
    except ImportError as e:
        raise SourceUnavailable(f"swebench: could not import eval_quality_swebench.py: {e}")

    try:
        rows = swebench_mod.load_swebench_lite(num_instances, token_budget, tokenizer_name=None)
    except ImportError as e:
        raise SourceUnavailable(
            f"swebench: `datasets` package (or another import) unavailable and no local "
            f".cache/ hit for this budget/count: {e}"
        )
    except Exception as e:  # network errors, HF hub errors, etc. — all treated as "source unavailable"
        raise SourceUnavailable(f"swebench: failed to load SWE-bench_Lite: {type(e).__name__}: {e}")

    if not rows:
        raise SourceUnavailable("swebench: loaded 0 rows")

    sessions = []
    for row in rows:
        content = row["problem_statement"]
        turns = [{"turn_index": 0, "role": "user", "content": content}]
        expected = [{"turn_index": 0, "gt_format": "text_patch", "calls": [row["patch"]]}]
        sessions.append({
            "session_id": f"swebench:{row.get('instance_id')}",
            "source": "swebench",
            "tools": [],
            "turns": turns,
            "expected": expected,
            "meta": {
                "instance_id": row.get("instance_id"),
                "n_turns": 1,
                "approx_tokens": _session_approx_tokens([], turns),
            },
        })
    return sessions


# --------------------------------------------------------------------------
# mixing
# --------------------------------------------------------------------------

def parse_mix(spec: str) -> dict:
    """'bfcl:50,public-test:30,swebench:20' -> {"bfcl": 50.0, ...}"""
    weights = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad --mix entry {part!r}, expected name:percent")
        name, pct = part.split(":", 1)
        name = name.strip()
        weights[name] = float(pct.strip())
    if not weights:
        raise ValueError(f"--mix {spec!r} parsed to no entries")
    return weights


def build_sessions(source_pools: dict, mix: dict, sessions_cap: int, rng) -> list:
    """source_pools: {name: list[session]} for sources that loaded successfully.
    mix: requested {name: pct}, may reference sources not in source_pools
    (those are dropped with a warning by the caller before this is called).
    Renormalizes remaining weights to sum to 1, allocates sessions_cap across
    them by that ratio (largest-remainder rounding so the total is exact),
    shuffles within each source (seeded) then interleaves+shuffles the final
    order (seeded) so a downstream consumer sees a realistic random mix
    rather than all-bfcl-then-all-swebench blocks.
    """
    available_mix = {k: v for k, v in mix.items() if k in source_pools and v > 0}
    if not available_mix:
        raise SourceUnavailable("no requested --mix sources have data available")
    total_weight = sum(available_mix.values())
    norm = {k: v / total_weight for k, v in available_mix.items()}

    raw_counts = {k: norm[k] * sessions_cap for k in norm}
    counts = {k: int(raw_counts[k]) for k in norm}
    remainder = sessions_cap - sum(counts.values())
    # largest-remainder method for exact total
    for k in sorted(norm, key=lambda k: raw_counts[k] - counts[k], reverse=True)[:remainder]:
        counts[k] += 1

    picked = []
    for name, n in counts.items():
        pool = list(source_pools[name])
        rng.shuffle(pool)
        n = min(n, len(pool))
        picked.extend(pool[:n])

    rng.shuffle(picked)
    return picked


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def print_stats(sessions: list) -> None:
    from collections import Counter
    import statistics

    if not sessions:
        print("no sessions to report stats on", file=sys.stderr)
        return

    by_source = Counter(s["source"] for s in sessions)
    n_turns = [s["meta"]["n_turns"] for s in sessions]
    tokens = [s["meta"]["approx_tokens"] for s in sessions]

    print(f"\n=== workload stats: {len(sessions)} sessions ===")
    print("by source:")
    for name, n in by_source.most_common():
        print(f"  {name:<14} {n:5d}  ({100 * n / len(sessions):.1f}%)")
    print(f"turns/session : min={min(n_turns)} max={max(n_turns)} mean={statistics.mean(n_turns):.2f}")
    print(
        f"approx tokens : min={min(tokens)} max={max(tokens)} "
        f"mean={statistics.mean(tokens):.1f} median={statistics.median(tokens):.1f}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert BFCL / SWE-bench / public-test into a mixed agent-loop workload JSONL."
    )
    ap.add_argument("--source", choices=["bfcl", "swebench", "public-test", "mixed"], required=True)
    ap.add_argument("--mix", default="bfcl:50,public-test:30,swebench:20",
                     help="only used with --source mixed: comma-separated name:percent")
    ap.add_argument("--sessions", type=int, default=None, help="cap total sessions written (default: all available for single-source; required effectively for mixed to bound the swebench download)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bfcl-dir", default=str(DEFAULT_BFCL_DIR))
    ap.add_argument("--public-test-file", default=str(DEFAULT_PUBLIC_TEST_FILE))
    ap.add_argument("--swebench-num-instances", type=int, default=100)
    ap.add_argument("--swebench-token-budget", type=int, default=512)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    bfcl_dir = Path(args.bfcl_dir)
    public_test_file = Path(args.public_test_file)

    def load(name):
        if name == "bfcl":
            return load_bfcl_sessions(bfcl_dir)
        if name == "public-test":
            return load_public_test_sessions(public_test_file)
        if name == "swebench":
            return load_swebench_sessions(args.swebench_num_instances, args.swebench_token_budget)
        raise ValueError(name)

    if args.source != "mixed":
        try:
            sessions = load(args.source)  # explicit single-source request: fail loudly, but with a clean message (no traceback)
        except SourceUnavailable as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        rng.shuffle(sessions)
        if args.sessions is not None:
            sessions = sessions[: args.sessions]
    else:
        mix = parse_mix(args.mix)
        pools = {}
        for name in mix:
            try:
                pools[name] = load(name)
            except SourceUnavailable as e:
                print(f"WARNING: dropping source {name!r} from mix: {e}", file=sys.stderr)
        if not pools:
            print("ERROR: --source mixed but every requested source failed to load", file=sys.stderr)
            sys.exit(1)
        cap = args.sessions if args.sessions is not None else sum(len(v) for v in pools.values())
        sessions = build_sessions(pools, mix, cap, rng)

    if not sessions:
        print("ERROR: no sessions produced", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"-> {out_path} ({len(sessions)} sessions)")
    print_stats(sessions)


if __name__ == "__main__":
    main()
