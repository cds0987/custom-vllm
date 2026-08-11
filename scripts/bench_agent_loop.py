"""Multi-turn agent-loop benchmark for an OpenAI-compatible vLLM endpoint.

New workload class this project has never measured before: bench_skills.py
and bench_swebench.py both fire ONE request per task (shared prefix + one
short suffix). Real agent traffic looks different -- a session runs K turns,
each turn's prompt is the PREVIOUS turn's prompt with the model's own output
plus a simulated tool result appended, and between turns the client is
blocked on a tool call (a real network/DB/shell call in production; here a
configurable sleep). Concretely, per session:

    turn 1        : [system_prefix] + [user request]              -> model emits a short tool-call
    (sleep T)     : simulated tool execution
    turn 2..K-1   : [system_prefix] + [everything so far, extended] -> next tool-call
    turn K        : [system_prefix] + [everything so far, extended] -> final natural-language answer

The property that matters and that this script is built to verify, not just
assume, is that turn n's prompt is a literal PREFIX (byte string, not just
"semantically similar") of turn n+1's prompt -- that's the only way vLLM's
automatic prefix-cache can reuse turn n's KV blocks for turn n+1. See
build_initial_transcript/append_tool_round below: the transcript is only
ever appended to, never rewritten (except by the --context-overflow-policy
stress mode, which deliberately breaks this and says so in the record).

The central open question this workload raises that bench_skills does not
answer: after a tool sleeps for T seconds, is the session's prefix still
hot in the KV cache, or has it been evicted and does turn n+1 pay a full
history re-prefill? --resume-probe is a dedicated mode for exactly that
question (a TTFT-vs-T curve); every normal run also buckets per-turn TTFT
by the tool gap that preceded it for the same read.

Prefix-cache instrumentation is deliberately NOT reimplemented here: TASK H
(see STATUS.md) already found and fixed the vllm 0.26 metric-name alias
(bare vllm:prefix_cache_queries/hits vs the _total-suffixed Prometheus
counter names) in scripts/bench_skills.py's scrape_prefix_cache_metrics().
This script imports and reuses that function (and diff_prefix_cache_metrics
/ percentile) rather than forking the bug fix a second time.

Two data sources, both optional and both gitignored (never read into any
report, never committed):
  --prefix-file      plain-text system prompt (skills_pack/system_prefix.txt
                      in this project: ~29K real tokens, 12 tool JSON
                      schemas already embedded)
  --questions-file    JSONL {"id","question",...} (skills_pack/
                      selected_questions.jsonl: 74 curated questions,
                      including an agentic subset)
Without them (or with --synthetic), the script generates its own small
fake tool-schema prefix and question list so it is fully runnable, and
fully testable (scripts/test_bench_agent_loop.py), with no project data on
disk at all.

Stress modes (all opt-in, default OFF, each independently testable against
the fake server in scripts/test_bench_agent_loop.py):
  --tool-latency-tail P:SEC     P% of tool calls take SEC seconds instead of
                                 the configured latency (a stuck/slow tool).
  --tool-result-spike P:TOKENS  P% of tool results are TOKENS tokens instead
                                 of the configured size (a huge document
                                 dropped into the middle of the transcript).
  --context-overflow-policy     {error,truncate-oldest,summarize-stub}: what
                                 happens when the (estimated) transcript
                                 exceeds --context-limit-tokens.
                                 truncate-oldest/summarize-stub deliberately
                                 break the prefix-extension property from
                                 that turn on (context_overflow_applied=True
                                 in the record) -- that's the point: this is
                                 the one policy proven to nuke the session's
                                 prefix cache, and the record says so.
  --toolcall-invalid-rate P     P% of tool-call turns are treated as invalid
                                 JSON, forcing one extra retry request (the
                                 real cost of not using guided decoding).
  --burst-sync                  all sessions in a level rendezvous on a
                                 threading.Barrier before firing each turn,
                                 instead of drifting apart naturally --
                                 simulates "everyone's tool finishes and they
                                 all hit the server at once".
  --abandon-rate P              P% of sessions have their connection closed
                                 mid-stream (requests Response.close()) part
                                 way through a random turn, simulating a
                                 client that walked away; record whether the
                                 client-side abort actually happened.
  --max-model-len N             optional: if set, validate that each turn's
                                 estimated prompt + max_tokens does not exceed
                                 this limit. If validation fails, skip that
                                 turn (mark it skipped_reason="context_budget_exceeded"
                                 in JSONL, issue a warning to stderr, and
                                 exclude from level summary stats). Agent-loop
                                 workload differs from single-turn bench: prompt
                                 grows with each turn (system prefix + cumulative
                                 history). Example: prefix ~30K tokens + 10 turns
                                 × ~800-token tool results = ~38K cumulative by
                                 turn 10; thus a 32K max-model-len will fail
                                 late turns. Default: None (no budget check).
  --mixed-chat N                N background short ordinary-chat workers run
                                 for the duration of the level, so the level
                                 summary also reports how much noise the
                                 agent-loop sessions add to unrelated traffic
                                 (and vice versa, via the two TTFT distros).
  --resume-probe                bypasses --sessions/--turns entirely: one
                                 session, sweeps a fixed tool-gap T over
                                 --resume-probe-gaps, repeats each T
                                 --resume-probe-trials times, and reports the
                                 TTFT(T) curve directly. This is the single
                                 most decisive measurement in the file.

Usage:
    python scripts/bench_agent_loop.py --synthetic --sessions 1,4,8 \\
        --turns 5 --tool-latency 2 --tool-result-tokens 200-800 \\
        --output out_agent/run1.jsonl

    python scripts/bench_agent_loop.py \\
        --prefix-file skills_pack/system_prefix.txt \\
        --questions-file skills_pack/selected_questions.jsonl \\
        --sessions 1,4,8,16 --turns 5 --tool-latency 2 \\
        --tool-result-tokens 200-800 --output out_agent/agent_normal.jsonl
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_skills as bsk  # noqa: E402  -- reused for prefix-cache scrape/diff/percentile + loaders

MODEL = os.environ.get("VLLM_MODEL", "")
URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
CHAT_URL = URL + "/v1/chat/completions"
METRICS_URL = URL + "/metrics"
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "300"))
BARRIER_TIMEOUT_S = 120.0

percentile = bsk.percentile  # re-export, same implementation


# --------------------------------------------------------------------------
# Transcript construction -- the append-only invariant lives entirely here
# --------------------------------------------------------------------------

HDR_USER = "### Yeu cau nguoi dung\n"
HDR_TOOLCALL = "### Tro ly (goi cong cu, JSON ngan gon)\n"
HDR_FINAL = "### Tro ly (cau tra loi cuoi cung, ngon ngu tu nhien)\n"
HDR_TOOL_RESULT = "### Ket qua cong cu\n"
HDR_RETRY = "### He thong: JSON cong cu khong hop le, hay sinh lai\n\n"
HDR_SUMMARY_STUB = "[TOM TAT LICH SU TRUOC -- DA BI CAT DE VUA CONTEXT]\n"


def build_initial_transcript(user_question: str) -> str:
    return HDR_USER + user_question + "\n\n" + HDR_TOOLCALL


def append_tool_round(transcript: str, tool_call_text: str, tool_result_text: str, next_is_final: bool) -> str:
    """The one and only place a turn's prompt is derived from the previous
    turn's prompt -- always by concatenation, never by rewriting, so that
    prompt(n) is always a literal string prefix of prompt(n+1)."""
    header = HDR_FINAL if next_is_final else HDR_TOOLCALL
    return transcript + tool_call_text + "\n\n" + HDR_TOOL_RESULT + tool_result_text + "\n\n" + header


def append_retry_round(transcript: str, invalid_tool_call_text: str) -> str:
    return transcript + invalid_tool_call_text + "\n\n" + HDR_RETRY + HDR_TOOLCALL


def est_tokens(text: str) -> int:
    """Word-count token estimate (no tokenizer available offline). Used only
    for the --context-overflow-policy trigger and --dry-run, never for
    billing/reporting real token counts (those come from server `usage`)."""
    return len(text.split())


def apply_overflow_policy(transcript: str, prefix_est_tokens: int, limit_tokens: int, policy: str):
    """Returns (new_transcript, applied, removed_word_count). `error` policy
    is handled by the caller before this is reached."""
    total = prefix_est_tokens + est_tokens(transcript)
    if limit_tokens <= 0 or total <= limit_tokens:
        return transcript, False, 0
    over = total - limit_tokens
    words = transcript.split(" ")
    cut = min(len(words), over + max(10, over // 10))
    if policy == "truncate-oldest":
        return " ".join(words[cut:]), True, cut
    if policy == "summarize-stub":
        return HDR_SUMMARY_STUB + " ".join(words[cut:]), True, cut
    return transcript, False, 0


def make_filler_text(rng: random.Random, n_tokens: int, tag: str = "tool") -> str:
    n_tokens = max(int(n_tokens), 1)
    words = [f"{tag}{rng.randint(0, 999999)}" for _ in range(n_tokens)]
    return "du_lieu_gia: " + " ".join(words)


# --------------------------------------------------------------------------
# Synthetic data (used when --synthetic or when files are not supplied)
# --------------------------------------------------------------------------

def synthetic_prefix(n_tools: int = 12) -> str:
    tools = []
    for i in range(n_tools):
        tools.append(
            json.dumps(
                {
                    "name": f"tool_{i}",
                    "description": f"Synthetic test tool {i}.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                },
                ensure_ascii=False,
            )
        )
    body = "Ban la mot tro ly AI co the goi cong cu. Danh sach cong cu kha dung:\n\n" + "\n".join(tools)
    body += (
        "\n\nKhi can them thong tin, goi MOT cong cu bang JSON ngan gon (khong giai thich). "
        "Khi da du thong tin, tra loi bang ngon ngu tu nhien, day du.\n"
    )
    return body


SYNTHETIC_TOPICS = ["gia san pham X", "trang thai don hang", "lich su giao dich", "chinh sach doi tra", "ton kho"]


def synthetic_questions(n: int = 20, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [
        {"id": f"synthetic-{i}", "question": f"Hay tra cuu {rng.choice(SYNTHETIC_TOPICS)} cho khach hang so {i}."}
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# SSE streaming client (same wire parsing as bench_skills.py), with an
# optional early-abort hook for --abandon-rate
# --------------------------------------------------------------------------

def stream_chat_completion(messages: list, max_tokens: int, abort_after_tokens: int | None = None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    buf = b""
    n_delta = 0
    with requests.post(CHAT_URL, json=payload, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
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
                data = line[len(b"data:"):].strip()
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
                if delta_text:
                    n_delta += 1
                yield t_recv, delta_text, usage, finish_reason
                if abort_after_tokens is not None and n_delta >= abort_after_tokens:
                    r.close()  # actually tears down the TCP connection -- real client-side cancel
                    return


def extract_cached_tokens(usage) -> int | None:
    """usage.prompt_tokens_details.cached_tokens, when the server was
    started with --enable-prompt-tokens-details (see vllm/entrypoints/
    openai/chat_completion/serving.py:_make_prompt_tokens_details). Not all
    deployments enable it; callers must handle None."""
    if not usage:
        return None
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        return ptd.get("cached_tokens")
    return None


def run_one_turn(messages: list, max_tokens: int, abort_after_tokens: int | None = None) -> dict:
    record = {"ok": False, "error": None, "abandoned": False}
    t_start = time.monotonic()
    token_times, text_parts = [], []
    last_usage = None
    finish_reason = None
    try:
        for t_recv, delta_text, usage, fr in stream_chat_completion(messages, max_tokens, abort_after_tokens):
            if usage:
                last_usage = usage
            if fr:
                finish_reason = fr
            if delta_text:
                token_times.append(t_recv)
                text_parts.append(delta_text)
        t_end = time.monotonic()
        text = "".join(text_parts)
        aborted = abort_after_tokens is not None and len(text_parts) >= abort_after_tokens
        completion_tokens = (last_usage or {}).get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = len(token_times)
        e2e_s = t_end - t_start
        ttft_s = (token_times[0] - t_start) if token_times else None
        decode_s = (t_end - token_times[0]) if token_times else None
        decode_tok_s = (completion_tokens / decode_s) if decode_s and decode_s > 0 else None
        record.update(
            ok=True,
            abandoned=aborted,
            ttft_s=ttft_s,
            e2e_s=e2e_s,
            decode_s=decode_s,
            decode_tok_s=decode_tok_s,
            prompt_tokens=(last_usage or {}).get("prompt_tokens"),
            completion_tokens=completion_tokens,
            cached_tokens=extract_cached_tokens(last_usage),
            finish_reason=finish_reason,
            output_text=text,
        )
    except requests.exceptions.RequestException as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["e2e_s"] = time.monotonic() - t_start
    except Exception as e:  # never let one bad turn crash the whole run
        record["error"] = f"{type(e).__name__}: {e}"
        record["e2e_s"] = time.monotonic() - t_start
    return record


# --------------------------------------------------------------------------
# Mixed ordinary-chat background traffic (--mixed-chat)
# --------------------------------------------------------------------------

MIXED_CHAT_QUESTIONS = [
    "Xin chao, ban khoe khong?",
    "Thu do cua Phap la gi?",
    "Ke mot cau chuyen ngan.",
    "1 cong 1 bang may?",
    "Hom nay thoi tiet the nao?",
]


def run_mixed_chat_worker(worker_id: int, seed: int, stop_event: threading.Event, out_records: list, lock: threading.Lock,
                           rate_s: float = 0.5, max_tokens: int = 64) -> None:
    rng = random.Random(seed * 7919 + worker_id + 555)
    i = 0
    while not stop_event.is_set():
        q = MIXED_CHAT_QUESTIONS[i % len(MIXED_CHAT_QUESTIONS)]
        rec = run_one_turn([{"role": "user", "content": q}], max_tokens)
        rec.update(worker_id=worker_id, seq=i, type="mixed_chat")
        with lock:
            out_records.append(rec)
        i += 1
        rng.random()  # keep the rng "used" for determinism parity even though question is cyclic
        stop_event.wait(rate_s)


# --------------------------------------------------------------------------
# Session runner: one agent conversation, K turns, all stress hooks live here
# --------------------------------------------------------------------------

def run_session(session_id: int, level_sessions: int, question: dict, prefix: str, prefix_est_tokens: int,
                 cfg: SimpleNamespace, rng: random.Random, barriers: list | None = None):
    turns = cfg.turns
    transcript = build_initial_transcript(bsk.build_user_content(question))
    session_t0 = time.monotonic()
    prev_prompt_tokens = None
    tool_wait_total = 0.0
    turn_records = []
    pending_gap = None
    abandoned = False
    budget_warning_issued = False

    do_abandon = cfg.abandon_rate > 0 and rng.random() * 100 < cfg.abandon_rate
    abandon_turn = rng.randint(1, turns) if do_abandon else None
    abandon_after_tokens = rng.randint(1, 5) if do_abandon else None

    turn_idx = 1
    while turn_idx <= turns:
        is_final = turn_idx == turns
        overflow_applied = False

        if cfg.overflow_policy != "none" and cfg.context_limit_tokens > 0:
            total_est = prefix_est_tokens + est_tokens(transcript)
            if total_est > cfg.context_limit_tokens:
                if cfg.overflow_policy == "error":
                    turn_records.append(
                        {
                            "ok": False,
                            "error": "context_overflow",
                            "context_overflow": True,
                            "session_id": session_id,
                            "level_sessions": level_sessions,
                            "turn_index": turn_idx,
                            "turns_total": turns,
                            "turn_role": "final" if is_final else "toolcall",
                            "question_id": question.get("id"),
                            "prior_tool_gap_s": pending_gap,
                        }
                    )
                    break
                transcript, overflow_applied, _removed = apply_overflow_policy(
                    transcript, prefix_est_tokens, cfg.context_limit_tokens, cfg.overflow_policy
                )

        messages = [{"role": "system", "content": prefix}, {"role": "user", "content": transcript}]
        max_tok = cfg.final_max_tokens if is_final else cfg.toolcall_max_tokens

        # Budget check: estimate whether prompt + max_tokens exceeds max_model_len
        if cfg.max_model_len is not None and cfg.max_model_len > 0:
            user_content = messages[1]["content"]
            est_prompt_tokens = est_tokens(user_content)
            total_est = est_prompt_tokens + max_tok
            if total_est > cfg.max_model_len:
                # Skip this turn due to budget overflow
                if not budget_warning_issued:
                    print(
                        f"WARNING: session {session_id} turn {turn_idx}: "
                        f"estimated {total_est} tokens exceeds max-model-len {cfg.max_model_len} "
                        f"-- serve with larger max-model-len (agent prompt grows with each turn)",
                        file=sys.stderr,
                    )
                    budget_warning_issued = True
                turn_records.append(
                    {
                        "ok": False,
                        "error": "context_budget_exceeded",
                        "skipped_reason": "context_budget_exceeded",
                        "est_prompt_tokens": est_prompt_tokens,
                        "max_tokens": max_tok,
                        "max_model_len": cfg.max_model_len,
                        "session_id": session_id,
                        "level_sessions": level_sessions,
                        "turn_index": turn_idx,
                        "turns_total": turns,
                        "turn_role": "final" if is_final else "toolcall",
                        "question_id": question.get("id"),
                        "prior_tool_gap_s": pending_gap,
                    }
                )
                break

        abort_after = abandon_after_tokens if (do_abandon and turn_idx == abandon_turn) else None
        rec = run_one_turn(messages, max_tok, abort_after_tokens=abort_after)
        rec.update(
            session_id=session_id,
            level_sessions=level_sessions,
            turn_index=turn_idx,
            turns_total=turns,
            turn_role="final" if is_final else "toolcall",
            question_id=question.get("id"),
            prior_tool_gap_s=pending_gap,
            prompt_tokens_delta=(
                rec.get("prompt_tokens") - prev_prompt_tokens
                if rec.get("prompt_tokens") is not None and prev_prompt_tokens is not None
                else None
            ),
            context_overflow_applied=overflow_applied,
            is_retry=False,
        )
        turn_records.append(rec)

        if rec.get("abandoned"):
            abandoned = True
            break
        if not rec.get("ok"):
            break

        prev_prompt_tokens = rec.get("prompt_tokens")
        if is_final:
            break

        tool_call_text = rec.get("output_text", "")

        if cfg.invalid_rate > 0 and rng.random() * 100 < cfg.invalid_rate:
            retry_transcript = append_retry_round(transcript, tool_call_text)
            retry_messages = [{"role": "system", "content": prefix}, {"role": "user", "content": retry_transcript}]
            retry_rec = run_one_turn(retry_messages, cfg.toolcall_max_tokens)
            retry_rec.update(
                session_id=session_id,
                level_sessions=level_sessions,
                turn_index=turn_idx,
                turns_total=turns,
                turn_role="toolcall_retry",
                question_id=question.get("id"),
                is_retry=True,
                retry_of_turn_index=turn_idx,
                prior_tool_gap_s=None,
                prompt_tokens_delta=(
                    retry_rec.get("prompt_tokens") - rec.get("prompt_tokens")
                    if retry_rec.get("prompt_tokens") is not None and rec.get("prompt_tokens") is not None
                    else None
                ),
            )
            turn_records.append(retry_rec)
            if not retry_rec.get("ok"):
                break
            transcript = retry_transcript
            tool_call_text = retry_rec.get("output_text", "")
            prev_prompt_tokens = retry_rec.get("prompt_tokens")

        lo, hi = cfg.tool_latency_range
        gap = rng.uniform(lo, hi) if hi > lo else lo
        tail_triggered = False
        if cfg.tail_pct > 0 and rng.random() * 100 < cfg.tail_pct:
            gap = cfg.tail_sec
            tail_triggered = True

        rlo, rhi = cfg.tool_result_tokens_range
        n_tok = rng.randint(int(rlo), int(rhi)) if rhi > rlo else int(rlo)
        spike_triggered = False
        if cfg.spike_pct > 0 and rng.random() * 100 < cfg.spike_pct:
            n_tok = cfg.spike_tokens
            spike_triggered = True

        time.sleep(gap)
        if barriers is not None and (turn_idx - 1) < len(barriers):
            try:
                barriers[turn_idx - 1].wait(timeout=BARRIER_TIMEOUT_S)
            except threading.BrokenBarrierError:
                pass
        tool_wait_total += gap

        tool_result_text = make_filler_text(rng, n_tok, tag="tool")
        rec["tail_latency_triggered"] = tail_triggered
        rec["tool_result_spike_triggered"] = spike_triggered
        rec["tool_gap_used_s"] = gap
        rec["tool_result_tokens_target"] = n_tok

        next_is_final = turn_idx + 1 == turns
        transcript = append_tool_round(transcript, tool_call_text, tool_result_text, next_is_final)
        pending_gap = gap
        turn_idx += 1

    session_wall_s = time.monotonic() - session_t0
    summary = summarize_session(session_id, level_sessions, question, turn_records, session_wall_s, tool_wait_total, abandoned)
    return turn_records, summary


def summarize_session(session_id, level_sessions, question, turn_records, session_wall_s, tool_wait_total, abandoned) -> dict:
    ok_turns = [r for r in turn_records if r.get("ok")]
    gpu_time_s = sum(r.get("e2e_s") or 0.0 for r in turn_records if r.get("e2e_s") is not None)
    total_prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok_turns)
    total_completion_tokens = sum(r.get("completion_tokens") or 0 for r in ok_turns)
    reconnect_ttfts = [
        r["ttft_s"] for r in ok_turns
        if (r.get("turn_index") or 0) >= 2 and r.get("ttft_s") is not None and not r.get("is_retry")
    ]
    prefill_reconnect_s = sum(reconnect_ttfts)
    completed = (not abandoned) and any(r.get("turn_role") == "final" and r.get("ok") for r in turn_records)
    overflowed = any(r.get("context_overflow") or r.get("context_overflow_applied") for r in turn_records)

    return {
        "type": "session_summary",
        "session_id": session_id,
        "level_sessions": level_sessions,
        "question_id": question.get("id"),
        "n_turn_records": len(turn_records),
        "n_turns_ok": len(ok_turns),
        "completed": completed,
        "abandoned": abandoned,
        "context_overflow": overflowed,
        "session_wall_s": session_wall_s,
        "tool_wait_total_s": tool_wait_total,
        "gpu_time_total_s": gpu_time_s,
        "pct_time_tool": (tool_wait_total / session_wall_s) if session_wall_s > 0 else None,
        "pct_time_gpu": (gpu_time_s / session_wall_s) if session_wall_s > 0 else None,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "prefill_reconnect_s": prefill_reconnect_s,
        "pct_time_prefill_reconnect": (prefill_reconnect_s / session_wall_s) if session_wall_s > 0 else None,
        "n_retries": sum(1 for r in turn_records if r.get("is_retry")),
    }


# --------------------------------------------------------------------------
# Level runner: `sessions` concurrent agent conversations + optional
# mixed-chat noise, one prefix-cache metrics delta for the whole level
# --------------------------------------------------------------------------

def scrape_cache_metrics() -> dict:
    bsk.METRICS_URL = METRICS_URL
    return bsk.scrape_prefix_cache_metrics()


def diff_cache_metrics(before: dict, after: dict) -> dict:
    return bsk.diff_prefix_cache_metrics(before, after)


def bucket_gap(g):
    if g is None:
        return None
    return round(g * 2) / 2.0  # nearest 0.5s


def run_level(sessions_n: int, questions: list, prefix: str, prefix_est_tokens: int, cfg: SimpleNamespace, seed: int):
    barriers = [threading.Barrier(sessions_n) for _ in range(max(cfg.turns - 1, 0))] if cfg.burst_sync else None

    stop_mixed = threading.Event()
    mixed_records: list = []
    mixed_lock = threading.Lock()
    mixed_threads = []
    for i in range(cfg.mixed_chat):
        t = threading.Thread(
            target=run_mixed_chat_worker, args=(i, seed, stop_mixed, mixed_records, mixed_lock), daemon=True
        )
        mixed_threads.append(t)
        t.start()

    before = scrape_cache_metrics()
    t0 = time.monotonic()

    def task(i):
        q = questions[i % len(questions)]
        rng = random.Random((seed * 1_000_003) + i)
        return run_session(i, sessions_n, q, prefix, prefix_est_tokens, cfg, rng, barriers)

    with ThreadPoolExecutor(max_workers=sessions_n) as pool:
        results = list(pool.map(task, range(sessions_n)))

    wall_elapsed = time.monotonic() - t0
    after = scrape_cache_metrics()
    cache_delta = diff_cache_metrics(before, after)

    stop_mixed.set()
    for t in mixed_threads:
        t.join(timeout=5)

    all_turn_records, all_session_summaries = [], []
    for turn_records, summary in results:
        all_turn_records.extend(turn_records)
        all_session_summaries.append(summary)

    level_summary = summarize_level(sessions_n, all_turn_records, all_session_summaries, wall_elapsed, cache_delta, mixed_records)
    return all_turn_records, all_session_summaries, level_summary, mixed_records


def summarize_level(sessions_n, turn_records, session_summaries, wall_elapsed, cache_delta, mixed_records) -> dict:
    ok_turns = [r for r in turn_records if r.get("ok")]
    completed_sessions = [s for s in session_summaries if s.get("completed")]
    abandoned_sessions = [s for s in session_summaries if s.get("abandoned")]
    overflow_sessions = [s for s in session_summaries if s.get("context_overflow")]
    budget_exceeded_turns = [r for r in turn_records if r.get("skipped_reason") == "context_budget_exceeded"]

    total_completion_tokens = sum(r.get("completion_tokens") or 0 for r in ok_turns)
    total_prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok_turns)

    queries_delta = cache_delta.get("vllm:prefix_cache_queries")
    hits_delta = cache_delta.get("vllm:prefix_cache_hits")
    hit_rate = (hits_delta / queries_delta) if queries_delta else None

    buckets: dict = {}
    for r in ok_turns:
        if (r.get("turn_index") or 0) < 2 or r.get("ttft_s") is None:
            continue
        buckets.setdefault(bucket_gap(r.get("prior_tool_gap_s")), []).append(r["ttft_s"])
    gap_curve = [
        {"tool_gap_bucket_s": b, "n": len(v), "ttft_mean_s": statistics.mean(v), "ttft_p95_s": percentile(v, 0.95)}
        for b, v in sorted(buckets.items(), key=lambda kv: (kv[0] is None, kv[0]))
    ]

    mixed_ttfts = [m["ttft_s"] for m in mixed_records if m.get("ok") and m.get("ttft_s") is not None]

    return {
        "type": "level_summary",
        "sessions": sessions_n,
        "n_sessions_completed": len(completed_sessions),
        "n_sessions_abandoned": len(abandoned_sessions),
        "n_sessions_context_overflow": len(overflow_sessions),
        "n_turns_skipped_budget_exceeded": len(budget_exceeded_turns),
        "wall_elapsed_s": wall_elapsed,
        "tasks_per_hour": (len(completed_sessions) / wall_elapsed * 3600.0) if wall_elapsed > 0 else None,
        "total_throughput_tok_s": (total_completion_tokens / wall_elapsed) if wall_elapsed > 0 else None,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "prefix_cache_queries_delta": queries_delta,
        "prefix_cache_hits_delta": hits_delta,
        "prefix_cache_hit_rate": hit_rate,
        "prefix_cache_metrics_raw_delta": cache_delta,
        "ttft_vs_tool_gap_curve": gap_curve,
        "n_tail_latency_events": sum(1 for r in turn_records if r.get("tail_latency_triggered")),
        "n_tool_result_spike_events": sum(1 for r in turn_records if r.get("tool_result_spike_triggered")),
        "n_invalid_toolcall_retries": sum(1 for r in turn_records if r.get("is_retry")),
        "n_context_overflow_policy_applied": sum(1 for r in turn_records if r.get("context_overflow_applied")),
        "mixed_chat_n_requests": len(mixed_records),
        "mixed_chat_ttft_p50": percentile(mixed_ttfts, 0.5),
        "mixed_chat_ttft_p95": percentile(mixed_ttfts, 0.95),
    }


# --------------------------------------------------------------------------
# --resume-probe: the dedicated TTFT-vs-tool-gap curve
# --------------------------------------------------------------------------

def run_resume_probe(prefix: str, question: dict, gaps: list, trials: int, cfg: SimpleNamespace, seed: int):
    records = []
    curve = []
    for T in gaps:
        ttfts = []
        for trial in range(trials):
            rng = random.Random(seed * 104729 + trial)
            transcript = build_initial_transcript(bsk.build_user_content(question))
            messages = [{"role": "system", "content": prefix}, {"role": "user", "content": transcript}]
            rec1 = run_one_turn(messages, cfg.toolcall_max_tokens)
            rec1.update(type="turn", probe_tool_gap_s=T, trial=trial, turn_index=1, question_id=question.get("id"))
            records.append(rec1)
            if not rec1.get("ok"):
                continue

            rlo, rhi = cfg.tool_result_tokens_range
            n_tok = rng.randint(int(rlo), int(rhi)) if rhi > rlo else int(rlo)
            tool_result_text = make_filler_text(rng, n_tok, tag="tool")
            transcript2 = append_tool_round(transcript, rec1.get("output_text", ""), tool_result_text, next_is_final=True)

            time.sleep(T)

            messages2 = [{"role": "system", "content": prefix}, {"role": "user", "content": transcript2}]
            rec2 = run_one_turn(messages2, cfg.final_max_tokens)
            rec2.update(
                type="turn",
                probe_tool_gap_s=T,
                trial=trial,
                turn_index=2,
                question_id=question.get("id"),
                prior_tool_gap_s=T,
                prompt_tokens_delta=(
                    rec2.get("prompt_tokens") - rec1.get("prompt_tokens")
                    if rec2.get("prompt_tokens") is not None and rec1.get("prompt_tokens") is not None
                    else None
                ),
            )
            records.append(rec2)
            if rec2.get("ok") and rec2.get("ttft_s") is not None:
                ttfts.append(rec2["ttft_s"])

        curve.append(
            {
                "tool_gap_s": T,
                "n_ok": len(ttfts),
                "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
                "ttft_p95_s": percentile(ttfts, 0.95) if ttfts else None,
            }
        )
    return records, {"type": "resume_probe_summary", "trials_per_gap": trials, "curve": curve}


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

def parse_range(spec: str, cast=float):
    """'2' -> (2,2); '0.5-3' -> (0.5,3). Splits on the first '-' at index>=1
    so a leading negative sign (not expected here, but safe) isn't mistaken
    for the range separator."""
    spec = str(spec)
    if "-" in spec[1:]:
        idx = spec.index("-", 1)
        lo, hi = cast(spec[:idx]), cast(spec[idx + 1:])
    else:
        lo = hi = cast(spec)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def parse_int_list(spec: str) -> list:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def parse_float_list(spec: str) -> list:
    return [float(x) for x in str(spec).split(",") if x.strip()]


def parse_pct_value(spec: str | None, cast=float):
    """'P:VALUE' -> (pct, value); None if spec is None."""
    if not spec:
        return None
    pct_s, val_s = spec.split(":", 1)
    return float(pct_s), cast(val_s)


def build_cfg(args) -> SimpleNamespace:
    tail = parse_pct_value(args.tool_latency_tail, cast=float)
    spike = parse_pct_value(args.tool_result_spike, cast=int)
    return SimpleNamespace(
        turns=args.turns,
        tool_latency_range=parse_range(args.tool_latency, cast=float),
        tool_result_tokens_range=parse_range(args.tool_result_tokens, cast=float),
        toolcall_max_tokens=args.toolcall_max_tokens,
        final_max_tokens=args.final_max_tokens,
        tail_pct=tail[0] if tail else 0.0,
        tail_sec=tail[1] if tail else 0.0,
        spike_pct=spike[0] if spike else 0.0,
        spike_tokens=spike[1] if spike else 0,
        overflow_policy=args.context_overflow_policy,
        context_limit_tokens=args.context_limit_tokens,
        invalid_rate=args.toolcall_invalid_rate,
        burst_sync=args.burst_sync,
        abandon_rate=args.abandon_rate,
        mixed_chat=args.mixed_chat,
        max_model_len=args.max_model_len,
    )


def dry_run(prefix: str, questions: list, cfg: SimpleNamespace) -> None:
    q0 = questions[0]
    prefix_tok_est = est_tokens(prefix)
    print("=== dry run (no server contact) ===")
    print(f"prefix: {len(prefix)} chars, ~{prefix_tok_est} tokens (est., word-count)")
    print(f"questions loaded: {len(questions)}, first id: {q0['id']}")
    print(f"turns={cfg.turns} tool_latency_range={cfg.tool_latency_range} tool_result_tokens_range={cfg.tool_result_tokens_range}")
    print(f"toolcall_max_tokens={cfg.toolcall_max_tokens} final_max_tokens={cfg.final_max_tokens}")
    print(
        f"stress: tail_pct={cfg.tail_pct} tail_sec={cfg.tail_sec} spike_pct={cfg.spike_pct} spike_tokens={cfg.spike_tokens} "
        f"overflow_policy={cfg.overflow_policy} context_limit_tokens={cfg.context_limit_tokens} "
        f"invalid_rate={cfg.invalid_rate} burst_sync={cfg.burst_sync} abandon_rate={cfg.abandon_rate} mixed_chat={cfg.mixed_chat}"
    )
    transcript = build_initial_transcript(bsk.build_user_content(q0))
    print(f"first turn transcript: {len(transcript)} chars, ~{est_tokens(transcript)} tokens (est.)")
    print("payload constructed OK, exiting (--dry-run).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None, help="overrides env VLLM_URL (default http://localhost:8000)")
    ap.add_argument("--model", default=None, help="overrides env VLLM_MODEL")
    ap.add_argument("--prefix-file", default=None)
    ap.add_argument("--questions-file", default=None)
    ap.add_argument("--synthetic", action="store_true", help="ignore --prefix-file/--questions-file, generate fake data")
    ap.add_argument("--synthetic-tools", type=int, default=12)
    ap.add_argument("--synthetic-questions", type=int, default=20)

    ap.add_argument("--sessions", type=parse_int_list, default=[1], help="comma-separated concurrent-session levels, e.g. 1,4,8,16")
    ap.add_argument("--turns", type=int, default=5, help="K: turns per session, last turn is the final natural-language answer")
    ap.add_argument("--tool-latency", default="2", help="seconds, fixed or 'lo-hi' range, e.g. 0.5-3")
    ap.add_argument("--tool-result-tokens", default="200-800", help="tokens (word-count est.), fixed or 'lo-hi' range")
    ap.add_argument("--toolcall-max-tokens", type=int, default=120)
    ap.add_argument("--final-max-tokens", type=int, default=400)

    ap.add_argument("--tool-latency-tail", default=None, help="'P:SEC' -- P%% of tool gaps take SEC seconds instead")
    ap.add_argument("--tool-result-spike", default=None, help="'P:TOKENS' -- P%% of tool results are TOKENS tokens instead")
    ap.add_argument("--context-overflow-policy", choices=["none", "error", "truncate-oldest", "summarize-stub"], default="none")
    ap.add_argument("--context-limit-tokens", type=int, default=0, help="estimated-token ceiling that triggers --context-overflow-policy")
    ap.add_argument("--toolcall-invalid-rate", type=float, default=0.0, help="%% of tool-call turns forced to retry once")
    ap.add_argument("--burst-sync", action="store_true", help="sessions rendezvous on a barrier before each turn instead of drifting")
    ap.add_argument("--abandon-rate", type=float, default=0.0, help="%% of sessions whose connection is closed mid-stream")
    ap.add_argument("--max-model-len", type=int, default=None, help="optional: validate budget per turn (skip if exceeds this limit)")
    ap.add_argument("--mixed-chat", type=int, default=0, help="N background ordinary short-chat workers run alongside each level")

    ap.add_argument("--resume-probe", action="store_true", help="dedicated TTFT-vs-tool-gap curve mode; ignores --sessions/--turns")
    ap.add_argument("--resume-probe-gaps", type=parse_float_list, default=[0.5, 2, 5, 15, 60])
    ap.add_argument("--resume-probe-trials", type=int, default=3)

    ap.add_argument("--output", default=None, help="output JSONL path (default: out_agent/bench_agent_loop_<ts>.jsonl)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=None, help="per-chunk read timeout seconds (env BENCH_TIMEOUT)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.synthetic or not (args.prefix_file and args.questions_file):
        prefix = synthetic_prefix(args.synthetic_tools)
        questions = synthetic_questions(args.synthetic_questions, args.seed)
    else:
        prefix = bsk.load_prefix(args.prefix_file)
        questions = bsk.load_questions(args.questions_file)

    cfg = build_cfg(args)
    prefix_est_tokens = est_tokens(prefix)

    if args.dry_run:
        dry_run(prefix, questions, cfg)
        return

    global MODEL, URL, CHAT_URL, METRICS_URL, READ_TIMEOUT
    if args.model:
        MODEL = args.model
    if not MODEL:
        print("ERROR: set VLLM_MODEL or pass --model", file=sys.stderr)
        sys.exit(1)
    if args.base_url:
        URL = args.base_url.rstrip("/")
        CHAT_URL = URL + "/v1/chat/completions"
        METRICS_URL = URL + "/metrics"
    if args.timeout:
        READ_TIMEOUT = args.timeout

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else REPO_ROOT / "out_agent" / f"bench_agent_loop_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    with out_path.open("w", encoding="utf-8") as f:
        if args.resume_probe:
            print(f"=== resume-probe: gaps={args.resume_probe_gaps} trials={args.resume_probe_trials} ===\n", flush=True)
            records, summary = run_resume_probe(prefix, questions[0], args.resume_probe_gaps, args.resume_probe_trials, cfg, args.seed)
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
            f.write(json.dumps(summary, default=str) + "\n")
            print("tool_gap_s  n_ok  ttft_mean_s  ttft_p95_s")
            for row in summary["curve"]:
                print(f"{row['tool_gap_s']:>10}  {row['n_ok']:>4}  {row['ttft_mean_s']}  {row['ttft_p95_s']}")
            print(f"\nresults -> {out_path}")
            return

        print(f"=== agent-loop benchmark: model={MODEL} turns={cfg.turns} sessions={args.sessions} ===\n", flush=True)
        for n in args.sessions:
            print(f"--- sessions={n} ---", flush=True)
            turn_records, session_summaries, level_summary, mixed_records = run_level(n, questions, prefix, prefix_est_tokens, cfg, args.seed)
            for r in turn_records:
                f.write(json.dumps(r, default=str) + "\n")
            for s in session_summaries:
                f.write(json.dumps(s, default=str) + "\n")
            for m in mixed_records:
                f.write(json.dumps(m, default=str) + "\n")
            f.write(json.dumps(level_summary, default=str) + "\n")
            f.flush()

            print(
                f"  completed={level_summary['n_sessions_completed']}/{n} "
                f"abandoned={level_summary['n_sessions_abandoned']} "
                f"overflow={level_summary['n_sessions_context_overflow']} "
                f"budget_exceeded={level_summary['n_turns_skipped_budget_exceeded']} "
                f"tasks/hr={level_summary['tasks_per_hour']} "
                f"throughput={level_summary['total_throughput_tok_s']} tok/s "
                f"cache_hit_rate={level_summary['prefix_cache_hit_rate']}",
                flush=True,
            )
            if level_summary['n_turns_skipped_budget_exceeded'] > 0:
                print(
                    f"    WARNING: {level_summary['n_turns_skipped_budget_exceeded']} turn(s) skipped due to context budget exceeded -- "
                    f"consider increasing --max-model-len or reducing --turns/--tool-result-tokens",
                    flush=True,
                )
            for row in level_summary["ttft_vs_tool_gap_curve"]:
                print(f"    gap~{row['tool_gap_bucket_s']}s n={row['n']} ttft_mean={row['ttft_mean_s']:.3f}s ttft_p95={row['ttft_p95_s']:.3f}s")

    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
