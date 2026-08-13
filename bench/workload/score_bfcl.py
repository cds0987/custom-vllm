"""Offline scorer for BFCL-derived agent-loop sessions produced by
scripts/prepare_agent_workload.py — measures TOOL-CALL QUALITY (did the model
call the right function with the right arguments, or correctly call nothing),
not speed. This is the check that has to pass before a 4-bit champion quant
is trusted with agent workloads: perplexity gates (eval_quality_swebench.py)
and throughput benchmarks (bench_skills.py etc.) can look fine while tool
calling quietly degrades, because argument formatting is a small, brittle
slice of total output likelihood.

WHAT THIS IS: a from-scratch reimplementation of the *idea* behind BFCL's own
AST-match checker (github.com/ShishirPatil/gorilla, bfcl_eval/), not a
vendored copy of it. It reuses the real BFCL ground-truth data (downloaded by
prepare_agent_workload.py / this repo's scripts) but scores against it with a
deliberately minimal rule set.

GROUND-TRUTH FORMATS HANDLED (both occur in real BFCL data — see
prepare_agent_workload.py's docstring for where each comes from):

  call_string            expected["calls"] = ["cd(folder='document')", ...]
                          Python-call-syntax strings. Parsed with `ast`
                          (Constant/UnaryOp/BinOp/List/Tuple/Dict only — NOT
                          full `eval`, so no name lookups or function calls
                          inside argument expressions; BFCL's own ground
                          truth only ever uses these plus simple arithmetic
                          division like `p=1/6`, which BinOp covers).
                          Every keyword argument present in the expected call
                          is treated as REQUIRED (the ground truth only lists
                          the arguments the reference solution actually
                          passed — there's no separate "required" list at
                          this format).

  possible_answer_dict   expected["calls"] = [{"funcName": {"param": [v1, v2, ...]}}]
                          One dict per expected (possibly parallel) call.
                          Each parameter's value list is a set of ACCEPTABLE
                          alternatives (BFCL uses this to accept paraphrases,
                          e.g. several equivalent free-text strings). A
                          candidate argument value matches if it equals ANY
                          alternative after normalization (see `_values_match`).

  none                    expected["calls"] = [] — correct behavior is to call
                          NOTHING this turn (this is how BFCL's `irrelevance`
                          split and the "no calls this turn" slots inside
                          multi-turn sessions are represented). A prediction
                          that calls anything here scores 0 for the turn.

  text_patch              (swebench sessions) not scorable by this tool at
                          all; `score_session` skips these turns and they are
                          excluded from every aggregate.

MATCHING RULE (deliberately the minimal reasonable bar, NOT full BFCL
semantics — the task calling for this script explicitly allows "khớp tên hàm
+ tập tham số bắt buộc" as the floor when full AST-match is out of scope):
  1. Function/tool name must match exactly.
  2. Every REQUIRED parameter in the expected call must be present in the
     predicted call with a matching value (see `_values_match`).
  3. Extra parameters in the prediction that aren't in the expected call are
     IGNORED (not penalized) — BFCL's real checker sometimes penalizes these
     for `possible_answer_dict`-format entries that declare an explicit
     "no other params" constraint; this scorer does not implement that.
  4. Within one turn, expected calls and predicted calls are matched
     greedily by (name, then most-required-params-satisfied) — this is NOT
     the optimal bipartite matching BFCL's real checker uses for the
     parallel-call case, so a pathological ordering could under- or
     over-count on turns with multiple ambiguous same-name calls. Rare in
     the datasets sampled here (composite/parallel splits with 2-3 calls),
     but noted for the record.
  5. Value matching (`_values_match`) does exact comparison after: numeric
     coercion (`1` == `1.0` == `"1"`), and for strings, `.strip().lower()`
     case/whitespace folding. It does NOT do semantic/paraphrase matching
     beyond the literal alternatives BFCL's own `possible_answer_dict` lists
     — e.g. if the ground truth lists "coconut" and the model says
     "coconut milk", that is a MISS under this scorer even though a human
     grader might accept it. Widening this (substring/fuzzy match) was
     considered and rejected: it trades false negatives for false positives
     in a way that's hard to bound, whereas exact-after-normalization is at
     least a stable, reproducible floor.

Nothing here executes model-predicted code or the real BFCL sandboxed classes
(GorillaFileSystem etc.) — this is pure AST/string comparison against
reference answers, safe to run with arbitrary model output.

Usage:
    python scripts/score_bfcl.py \\
        --workload out_agent_workload/bfcl_only.jsonl \\
        --predictions out_agent_workload/bfcl_only.predictions.jsonl \\
        --output out_agent_workload/bfcl_only.score.json

Predictions JSONL schema (one line per session_id in --workload):
    {"session_id": "...", "predicted": [ [call, call, ...], [call, ...], ... ] }
`predicted` is a list aligned by index to the session's `turns`/`expected`
(one list of calls per turn). Each call may be either:
  - a call-syntax string, e.g. "cd(folder='document')"  (matches call_string
    ground truth directly), or
  - an OpenAI-style dict {"name": "...", "arguments": {...}} or
    {"function": {"name": "...", "arguments": {...}}} (arguments may be a
    dict already or a JSON-encoded string, both accepted) — the natural
    shape of a vLLM/OpenAI tool_calls response.
A turn predicting zero calls should use an empty list `[]`.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # bench/workload/ -> repo root


# --------------------------------------------------------------------------
# call_string parsing: restricted, safe evaluator (no name lookups, no calls
# inside arguments) — covers everything BFCL's own ground truth actually uses.
# --------------------------------------------------------------------------

class UnsupportedExpression(Exception):
    pass


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _safe_eval(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
        }
        for op_type, fn in ops.items():
            if isinstance(node.op, op_type):
                return fn(left, right)
        raise UnsupportedExpression(f"unsupported binop {ast.dump(node.op)}")
    if isinstance(node, ast.List):
        return [_safe_eval(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_safe_eval(k): _safe_eval(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[node.id]
    raise UnsupportedExpression(f"unsupported node {ast.dump(node)}")


def parse_call_string(call_str: str) -> tuple:
    """'cd(folder=\\'document\\')' -> ('cd', {'folder': 'document'}, []).
    Real BFCL ground truth also uses POSITIONAL args for single-arg calls,
    e.g. "activateParkingBrake('engage')" — these come back as a separate
    `positional` list (evaluated values, in call order) since we don't know
    the parameter name without the function's schema; `resolve_positional`
    below maps them to names using the session's tool schema when available.
    Raises UnsupportedExpression / SyntaxError on anything outside the
    restricted grammar above — callers should treat that as "no match"
    rather than crash the whole scoring run."""
    tree = ast.parse(call_str.strip(), mode="eval")
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise UnsupportedExpression(f"not a simple call: {call_str!r}")
    kwargs = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise UnsupportedExpression(f"**kwargs not supported: {call_str!r}")
        kwargs[kw.arg] = _safe_eval(kw.value)
    positional = [_safe_eval(a) for a in call.args]
    return call.func.id, kwargs, positional


def resolve_positional(name: str, positional: list, kwargs: dict, param_order: dict) -> dict:
    """Merge positional-arg values into `kwargs` by name, using the ordered
    parameter list declared in the tool's JSON schema (`param_order[name]`,
    built from session["tools"] — see `build_param_order`). If the function
    isn't in `param_order` (tools list wasn't supplied, or schema/name
    mismatch), positional args fall back to synthetic keys "_pos0", "_pos1",
    ... which will not match a real (named-argument) model prediction — a
    deliberate conservative miss rather than a guess. See module docstring
    limitation notes."""
    if not positional:
        return kwargs
    names = param_order.get(name, [])
    merged = dict(kwargs)
    for i, val in enumerate(positional):
        key = names[i] if i < len(names) else f"_pos{i}"
        merged.setdefault(key, val)
    return merged


def build_param_order(tools: list) -> dict:
    """{function_name: [param_name, ...]} in declared schema order, from an
    OpenAI-style tools list [{"type":"function","function":{"name":...,
    "parameters":{"properties": {...}}}}, ...]."""
    order = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        name = fn.get("name")
        if not name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        order[name] = list(props.keys())
    return order


# --------------------------------------------------------------------------
# predicted-call normalization (accepts string or OpenAI tool_call dict shape)
# --------------------------------------------------------------------------

def normalize_predicted_call(call, param_order: dict):
    """Returns (name, kwargs) or None if unparseable (counts as a miss, not
    a crash)."""
    if isinstance(call, str):
        try:
            name, kwargs, positional = parse_call_string(call)
            return name, resolve_positional(name, positional, kwargs, param_order)
        except (SyntaxError, UnsupportedExpression, ValueError):
            return None
    if isinstance(call, dict):
        fn = call.get("function", call)  # unwrap {"type":"function","function":{...}} or tool_call shape
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if not isinstance(args, dict) or not name:
            return None
        return name, args
    return None


# --------------------------------------------------------------------------
# value matching
# --------------------------------------------------------------------------

def _numeric(v):
    if isinstance(v, bool):
        return None  # keep bool distinct from int/float
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _values_match(predicted, expected) -> bool:
    if predicted == expected:
        return True
    pn, en = _numeric(predicted), _numeric(expected)
    if pn is not None and en is not None:
        return pn == en
    if isinstance(predicted, str) and isinstance(expected, str):
        return predicted.strip().lower() == expected.strip().lower()
    if isinstance(predicted, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(predicted) == len(expected) and all(
            _values_match(p, e) for p, e in zip(predicted, expected)
        )
    return False


def _call_matches_call_string(pred_name, pred_kwargs, exp_name, exp_kwargs) -> bool:
    if pred_name != exp_name:
        return False
    for k, v in exp_kwargs.items():  # every expected kwarg is required (see module docstring)
        if k not in pred_kwargs or not _values_match(pred_kwargs[k], v):
            return False
    return True


def _call_matches_possible_answer(pred_name, pred_kwargs, exp_call: dict) -> bool:
    if len(exp_call) != 1:
        return False
    (exp_name, param_alternatives), = exp_call.items()
    if pred_name != exp_name:
        return False
    for param, alternatives in param_alternatives.items():
        if param not in pred_kwargs:
            return False
        if not any(_values_match(pred_kwargs[param], alt) for alt in alternatives):
            return False
    return True


# --------------------------------------------------------------------------
# per-turn / per-session scoring
# --------------------------------------------------------------------------

def score_turn(gt_format: str, expected_calls: list, predicted_calls: list, param_order: dict) -> dict:
    """Greedy matching (see module docstring limitation #4). Returns
    {"n_expected", "n_predicted", "n_matched", "turn_score"}."""
    if gt_format == "none":
        n_matched = 1 if not predicted_calls else 0
        return {
            "n_expected": 0, "n_predicted": len(predicted_calls),
            "n_matched": n_matched, "turn_score": float(n_matched),
        }
    if gt_format not in ("call_string", "possible_answer_dict"):
        return {"n_expected": 0, "n_predicted": len(predicted_calls), "n_matched": 0, "turn_score": None}

    predicted_parsed = [normalize_predicted_call(c, param_order) for c in predicted_calls]
    predicted_parsed = [p for p in predicted_parsed if p is not None]
    used = [False] * len(predicted_parsed)

    n_matched = 0
    for exp in expected_calls:
        if gt_format == "call_string":
            try:
                exp_name, exp_kwargs, exp_positional = parse_call_string(exp) if isinstance(exp, str) else (None, None, [])
                if exp_name is not None:
                    exp_kwargs = resolve_positional(exp_name, exp_positional, exp_kwargs, param_order)
            except (SyntaxError, UnsupportedExpression, ValueError):
                exp_name, exp_kwargs = None, None
        for i, pred in enumerate(predicted_parsed):
            if used[i] or pred is None:
                continue
            pred_name, pred_kwargs = pred
            if gt_format == "call_string":
                ok = exp_name is not None and _call_matches_call_string(pred_name, pred_kwargs, exp_name, exp_kwargs)
            else:
                ok = isinstance(exp, dict) and _call_matches_possible_answer(pred_name, pred_kwargs, exp)
            if ok:
                used[i] = True
                n_matched += 1
                break

    n_expected = len(expected_calls)
    turn_score = (n_matched / n_expected) if n_expected else (1.0 if not predicted_calls else 0.0)
    return {
        "n_expected": n_expected, "n_predicted": len(predicted_calls),
        "n_matched": n_matched, "turn_score": turn_score,
    }


def score_session(session: dict, predicted_turns: list) -> dict:
    param_order = build_param_order(session.get("tools"))
    turn_results = []
    for exp_entry in session["expected"]:
        idx = exp_entry["turn_index"]
        gt_format = exp_entry["gt_format"]
        if gt_format == "text_patch":
            continue  # not scorable by this tool (see module docstring)
        predicted_calls = predicted_turns[idx] if idx < len(predicted_turns) else []
        result = score_turn(gt_format, exp_entry["calls"], predicted_calls, param_order)
        result["turn_index"] = idx
        result["gt_format"] = gt_format
        turn_results.append(result)

    scorable = [t for t in turn_results if t["turn_score"] is not None]
    session_score = (sum(t["turn_score"] for t in scorable) / len(scorable)) if scorable else None
    return {
        "session_id": session["session_id"],
        "source": session["source"],
        "n_turns_scored": len(scorable),
        "session_score": session_score,
        "turns": turn_results,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Score BFCL agent-loop predictions against reference answers.")
    ap.add_argument("--workload", required=True, help="JSONL from prepare_agent_workload.py")
    ap.add_argument("--predictions", required=True, help="JSONL: {session_id, predicted: [[call,...], ...]}")
    ap.add_argument("--output", default=None, help="write full per-session/per-turn results here (JSON)")
    args = ap.parse_args()

    sessions = {s["session_id"]: s for s in _read_jsonl(Path(args.workload))}
    preds = {p["session_id"]: p.get("predicted", []) for p in _read_jsonl(Path(args.predictions))}

    results = []
    missing = []
    for sid, session in sessions.items():
        if session["source"] not in ("bfcl",):
            continue  # swebench/public-test aren't BFCL-scorable
        if sid not in preds:
            missing.append(sid)
            continue
        results.append(score_session(session, preds[sid]))

    if missing:
        print(f"WARNING: {len(missing)} bfcl session(s) in workload have no prediction (excluded from scoring)", file=sys.stderr)

    scored = [r for r in results if r["session_score"] is not None]
    overall = sum(r["session_score"] for r in scored) / len(scored) if scored else None

    n_refusal_turns = sum(1 for r in results for t in r["turns"] if t["gt_format"] == "none")
    n_refusal_correct = sum(
        t["n_matched"] for r in results for t in r["turns"] if t["gt_format"] == "none"
    )

    summary = {
        "n_sessions_scored": len(scored),
        "n_sessions_missing_prediction": len(missing),
        "mean_session_score": overall,
        "n_refusal_turns": n_refusal_turns,
        "refusal_accuracy": (n_refusal_correct / n_refusal_turns) if n_refusal_turns else None,
        "sessions": results,
    }

    print(f"sessions scored          : {len(scored)}")
    print(f"missing predictions      : {len(missing)}")
    print(f"mean session score       : {overall}")
    print(f"refusal turns (no-call)  : {n_refusal_turns}  accuracy={summary['refusal_accuracy']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"-> {out_path}")


if __name__ == "__main__":
    main()
