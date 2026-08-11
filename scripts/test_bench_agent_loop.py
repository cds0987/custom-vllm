#!/usr/bin/env python3
"""
Local test suite for scripts/bench_agent_loop.py -- no real vLLM server or
GPU needed. Follows the pattern in scripts/test_warmup_prefix.py: a tiny
http.server-based fake OpenAI-compatible server runs in a background thread,
so this exercises the real requests.post/SSE-parsing code path against a
real socket instead of mocking `requests`.

Covers the base workload (per the task spec) plus one behavioral case for
each opt-in stress flag:

  base:
    (a) prompt(n+1) literally contains prompt(n) as a string prefix -- the
        one invariant every prefix-cache conclusion in STATUS.md depends on.
    (b) --tool-latency is actually respected (measured gap between the POST
        that ends turn n and the POST that starts turn n+1).
    (c) end-to-end / session / level metrics are computed correctly.
    (d) --seed makes a run reproducible (same tool-gap and tool-result-size
        sequence).
    (e) N concurrent sessions never interleave/mix each other's transcripts.

  stress:
    --tool-latency-tail, --tool-result-spike, --context-overflow-policy
    (error / truncate-oldest / summarize-stub), --toolcall-invalid-rate,
    --burst-sync, --abandon-rate, --mixed-chat, --resume-probe.

Run: python scripts/test_bench_agent_loop.py
"""

import http.server
import json
import os
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_agent_loop as bal  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeAgentServer:
    """Stands in for vLLM's OpenAI-compatible /v1/chat/completions + /metrics.

    Every POST body is captured (with an arrival timestamp) so tests can
    inspect exactly what the client sent and when. The streamed response is
    `n_chunks` small SSE deltas (content f"out{i}_") each preceded by
    `chunk_delay` seconds, then a final usage chunk, then [DONE]. If the
    client aborts mid-stream (bench_agent_loop's --abandon-rate calling
    Response.close()), the write loop hits a socket error and increments
    `client_aborted` -- that is the server-side proof the abort was a real
    TCP-level cancel, not just the client giving up on reading.
    """

    def __init__(self, n_chunks=4, chunk_delay=0.0, include_cached_tokens=True):
        self.port = free_port()
        self.n_chunks = n_chunks
        self.chunk_delay = chunk_delay
        self.include_cached_tokens = include_cached_tokens
        self.captured = []  # list of {"body":..., "t": monotonic}
        self.lock = threading.Lock()
        self.client_aborted = 0
        self.metrics_calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/metrics":
                    with outer.lock:
                        outer.metrics_calls += 1
                        n = outer.metrics_calls
                    body = (
                        f'vllm:prefix_cache_queries_total{{model_name="m"}} {n * 10}\n'
                        f'vllm:prefix_cache_hits_total{{model_name="m"}} {n * 9}\n'
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with outer.lock:
                    outer.captured.append({"body": body, "t": time.monotonic()})

                user_content = ""
                for m in body.get("messages", []):
                    if m.get("role") == "user":
                        user_content = m.get("content", "")
                prompt_tokens = len(user_content.split()) + 50
                max_tokens = body.get("max_tokens", 50)
                n_out = min(outer.n_chunks, max(max_tokens, 1))

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for i in range(n_out):
                        chunk = {"choices": [{"delta": {"content": f"out{i}_"}, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        if outer.chunk_delay:
                            time.sleep(outer.chunk_delay)
                    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n_out}
                    if outer.include_cached_tokens:
                        usage["prompt_tokens_details"] = {"cached_tokens": max(prompt_tokens - 50, 0)}
                    final_chunk = {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": usage}
                    self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
                    with outer.lock:
                        outer.client_aborted += 1

        self.httpd = HTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def shutdown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)


def point_module_at(srv: FakeAgentServer, timeout=20.0):
    bal.MODEL = "fake-model"
    bal.URL = srv.base_url
    bal.CHAT_URL = srv.base_url + "/v1/chat/completions"
    bal.METRICS_URL = srv.base_url + "/metrics"
    bal.READ_TIMEOUT = timeout


def make_cfg(**overrides):
    base = dict(
        turns=3,
        tool_latency_range=(0.05, 0.05),
        tool_result_tokens_range=(20, 20),
        toolcall_max_tokens=10,
        final_max_tokens=10,
        tail_pct=0.0,
        tail_sec=0.0,
        spike_pct=0.0,
        spike_tokens=0,
        overflow_policy="none",
        context_limit_tokens=0,
        invalid_rate=0.0,
        burst_sync=False,
        abandon_rate=0.0,
        mixed_chat=0,
        max_model_len=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def one_question(qid="q0"):
    return {"id": qid, "question": f"Cau hoi kiem thu {qid}"}


class TestPrefixExtension(unittest.TestCase):
    """(a) prompt(n+1) literally contains prompt(n) as a string prefix."""

    def test_transcript_grows_by_append_only(self):
        srv = FakeAgentServer(n_chunks=3)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=4)
            rng = random.Random(1)
            turn_records, summary = bal.run_session(0, 1, one_question(), "SYS_PREFIX", bal.est_tokens("SYS_PREFIX"), cfg, rng)
            self.assertTrue(summary["completed"])
            self.assertEqual(len(srv.captured), 4)
            bodies = [c["body"] for c in srv.captured]
            contents = [next(m["content"] for m in b["messages"] if m["role"] == "user") for b in bodies]
            for n in range(len(contents) - 1):
                self.assertTrue(
                    contents[n + 1].startswith(contents[n]),
                    f"turn {n+1} prompt is not an extension of turn {n} prompt",
                )
                self.assertGreater(len(contents[n + 1]), len(contents[n]))
            # system message must stay byte-identical every turn (the actual
            # shared prefix in the real skills_pack scenario)
            for b in bodies:
                sys_msg = next(m["content"] for m in b["messages"] if m["role"] == "system")
                self.assertEqual(sys_msg, "SYS_PREFIX")
        finally:
            srv.shutdown()


class TestToolLatencyRespected(unittest.TestCase):
    def test_gap_between_turns_matches_configured_latency(self):
        srv = FakeAgentServer(n_chunks=2, chunk_delay=0.0)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=3, tool_latency_range=(0.3, 0.3))
            rng = random.Random(2)
            bal.run_session(0, 1, one_question(), "P", bal.est_tokens("P"), cfg, rng)
            times = [c["t"] for c in srv.captured]
            self.assertEqual(len(times), 3)
            for i in range(len(times) - 1):
                gap = times[i + 1] - times[i]
                self.assertGreaterEqual(gap, 0.28)
                self.assertLess(gap, 1.5)
        finally:
            srv.shutdown()


class TestEndToEndMetrics(unittest.TestCase):
    def test_session_and_level_summaries(self):
        srv = FakeAgentServer(n_chunks=3)
        point_module_at(srv)
        try:
            cfg = make_cfg(turns=3, tool_latency_range=(0.05, 0.05))
            _turn_records, session_summaries, level_summary, _mixed = bal.run_level(
                2, [one_question("qa"), one_question("qb")], "PREFIX", bal.est_tokens("PREFIX"), cfg, seed=7
            )
            self.assertEqual(len(session_summaries), 2)
            for s in session_summaries:
                self.assertTrue(s["completed"])
                self.assertEqual(s["n_turns_ok"], 3)
                self.assertGreater(s["session_wall_s"], 0)
                self.assertIsNotNone(s["pct_time_tool"])
                self.assertIsNotNone(s["pct_time_gpu"])
                self.assertGreaterEqual(s["pct_time_tool"] + s["pct_time_gpu"], 0.0)
            self.assertEqual(level_summary["n_sessions_completed"], 2)
            self.assertGreater(level_summary["total_throughput_tok_s"], 0)
            self.assertIsNotNone(level_summary["tasks_per_hour"])
            self.assertIsNotNone(level_summary["prefix_cache_hit_rate"])
            # 2 sessions x 3 turns = 6 requests -> metrics scraped before/after
            self.assertGreaterEqual(srv.metrics_calls, 2)
        finally:
            srv.shutdown()


class TestSeedReproducibility(unittest.TestCase):
    def test_same_seed_same_tool_gap_and_result_size_sequence(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=4, tool_latency_range=(0.01, 0.05), tool_result_tokens_range=(10, 50))

            def gaps_and_sizes(seed):
                rng = random.Random(seed)
                turn_records, _ = bal.run_session(0, 1, one_question(), "P", bal.est_tokens("P"), cfg, rng)
                return [
                    (r.get("tool_gap_used_s"), r.get("tool_result_tokens_target"))
                    for r in turn_records
                    if "tool_gap_used_s" in r
                ]

            seq1 = gaps_and_sizes(99)
            seq2 = gaps_and_sizes(99)
            seq3 = gaps_and_sizes(100)
            self.assertEqual(seq1, seq2)
            self.assertNotEqual(seq1, seq3)
        finally:
            srv.shutdown()


class TestSessionIsolation(unittest.TestCase):
    def test_concurrent_sessions_do_not_mix_transcripts(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            questions = [one_question(f"q{i}") for i in range(4)]
            cfg = make_cfg(turns=3, tool_latency_range=(0.02, 0.08))
            bal.run_level(4, questions, "P", bal.est_tokens("P"), cfg, seed=3)

            by_question = {}
            for c in srv.captured:
                content = next(m["content"] for m in c["body"]["messages"] if m["role"] == "user")
                first_line = content.split("\n", 1)[0]
                qid = next(q["id"] for q in questions if q["question"] in content)
                by_question.setdefault(qid, []).append(content)

            self.assertEqual(len(by_question), 4)
            for qid, contents in by_question.items():
                self.assertEqual(len(contents), 3)
                # strictly growing, each an extension of the previous, and no
                # other session's question text ever appears in this one's turns
                for other_qid, other_q in [(q["id"], q["question"]) for q in questions if q["id"] != qid]:
                    for c in contents:
                        self.assertNotIn(other_q, c)
                for n in range(len(contents) - 1):
                    self.assertTrue(contents[n + 1].startswith(contents[n]))
        finally:
            srv.shutdown()


class TestToolLatencyTail(unittest.TestCase):
    def test_tail_always_overrides_gap_when_pct_100(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=3, tool_latency_range=(0.01, 0.02), tail_pct=100.0, tail_sec=0.4)
            rng = random.Random(5)
            turn_records, _ = bal.run_session(0, 1, one_question(), "P", bal.est_tokens("P"), cfg, rng)
            tail_turns = [r for r in turn_records if "tail_latency_triggered" in r]
            self.assertEqual(len(tail_turns), 2)
            for r in tail_turns:
                self.assertTrue(r["tail_latency_triggered"])
                self.assertAlmostEqual(r["tool_gap_used_s"], 0.4, delta=0.01)
        finally:
            srv.shutdown()


class TestToolResultSpike(unittest.TestCase):
    def test_spike_forces_large_tool_result(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=3, tool_result_tokens_range=(10, 10), spike_pct=100.0, spike_tokens=500)
            rng = random.Random(6)
            turn_records, _ = bal.run_session(0, 1, one_question(), "P", bal.est_tokens("P"), cfg, rng)
            spiked = [r for r in turn_records if "tool_result_spike_triggered" in r]
            self.assertEqual(len(spiked), 2)
            for r in spiked:
                self.assertTrue(r["tool_result_spike_triggered"])
                self.assertEqual(r["tool_result_tokens_target"], 500)
            # the transcript actually grew by ~500 words, not ~10 -- check the
            # next captured body is much bigger than a non-spiked round would be
            bodies = [c["body"] for c in srv.captured]
            contents = [next(m["content"] for m in b["messages"] if m["role"] == "user") for b in bodies]
            self.assertGreater(len(contents[1].split()) - len(contents[0].split()), 400)
        finally:
            srv.shutdown()


class TestContextOverflowError(unittest.TestCase):
    def test_error_policy_fails_the_turn(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(
                turns=6, tool_result_tokens_range=(200, 200), overflow_policy="error", context_limit_tokens=100
            )
            rng = random.Random(7)
            turn_records, summary = bal.run_session(0, 1, one_question(), "P", bal.est_tokens("P"), cfg, rng)
            overflow_recs = [r for r in turn_records if r.get("context_overflow")]
            self.assertGreaterEqual(len(overflow_recs), 1)
            self.assertEqual(overflow_recs[0]["error"], "context_overflow")
            self.assertFalse(summary["completed"])
            self.assertTrue(summary["context_overflow"])
        finally:
            srv.shutdown()


class TestContextOverflowTruncateAndSummarize(unittest.TestCase):
    def _run(self, policy):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(
                turns=6, tool_result_tokens_range=(200, 200), overflow_policy=policy, context_limit_tokens=100
            )
            rng = random.Random(8)
            turn_records, summary = bal.run_session(0, 1, one_question(), "P", bal.est_tokens("P"), cfg, rng)
            bodies = [c["body"] for c in srv.captured]
            contents = [next(m["content"] for m in b["messages"] if m["role"] == "user") for b in bodies]
            return turn_records, summary, contents
        finally:
            srv.shutdown()

    def test_truncate_oldest_breaks_prefix_and_shrinks(self):
        turn_records, summary, contents = self._run("truncate-oldest")
        applied = [r for r in turn_records if r.get("context_overflow_applied")]
        self.assertGreaterEqual(len(applied), 1)
        # the extension invariant MUST be broken at least once -- proof the
        # session's cached prefix is gone (this is the point of the test)
        broke_somewhere = any(not contents[n + 1].startswith(contents[n]) for n in range(len(contents) - 1))
        self.assertTrue(broke_somewhere)
        self.assertTrue(summary["context_overflow"])

    def test_summarize_stub_prepends_marker_and_shrinks(self):
        turn_records, summary, contents = self._run("summarize-stub")
        applied = [r for r in turn_records if r.get("context_overflow_applied")]
        self.assertGreaterEqual(len(applied), 1)
        self.assertTrue(any(c.startswith(bal.HDR_SUMMARY_STUB) for c in contents[1:]))
        self.assertTrue(summary["context_overflow"])


class TestInvalidToolcallRetry(unittest.TestCase):
    def test_invalid_rate_100_forces_a_retry_each_toolcall_turn(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=3, invalid_rate=100.0)
            rng = random.Random(9)
            turn_records, summary = bal.run_session(0, 1, one_question(), "P", bal.est_tokens("P"), cfg, rng)
            retries = [r for r in turn_records if r.get("is_retry")]
            # turns=3 -> 2 tool-call turns (1,2), each forced to retry once
            self.assertEqual(len(retries), 2)
            for r in retries:
                self.assertEqual(r["turn_role"], "toolcall_retry")
                self.assertIn(r["retry_of_turn_index"], (1, 2))
            self.assertEqual(summary["n_retries"], 2)
            # extra HTTP calls actually happened: 3 normal turns + 2 retries
            self.assertEqual(len(srv.captured), 5)
        finally:
            srv.shutdown()


class TestBurstSync(unittest.TestCase):
    def test_sessions_rendezvous_despite_different_gaps(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            questions = [one_question(f"b{i}") for i in range(3)]
            # wide, session-dependent gap range so without the barrier the
            # sessions would naturally drift far apart
            cfg = make_cfg(turns=3, tool_latency_range=(0.05, 0.5), burst_sync=True)
            bal.run_level(3, questions, "P", bal.est_tokens("P"), cfg, seed=11)

            # group arrival timestamps by turn position (2nd request per session)
            by_qid_times = {}
            for c in srv.captured:
                content = next(m["content"] for m in c["body"]["messages"] if m["role"] == "user")
                qid = next(q["id"] for q in questions if q["question"] in content)
                by_qid_times.setdefault(qid, []).append(c["t"])
            second_turn_times = [sorted(v)[1] for v in by_qid_times.values()]
            spread = max(second_turn_times) - min(second_turn_times)
            self.assertLess(spread, 0.3, "burst-sync should make all sessions fire turn 2 nearly simultaneously")
        finally:
            srv.shutdown()


class TestAbandonRate(unittest.TestCase):
    def test_abandon_closes_connection_and_server_sees_it(self):
        # many slow chunks so the client's early r.close() actually lands
        # mid-stream instead of racing the server to completion
        srv = FakeAgentServer(n_chunks=50, chunk_delay=0.02)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=3, abandon_rate=100.0, toolcall_max_tokens=50, final_max_tokens=50)
            rng = random.Random(12)
            turn_records, summary = bal.run_session(0, 1, one_question(), "P", bal.est_tokens("P"), cfg, rng)
            self.assertTrue(summary["abandoned"])
            abandoned_recs = [r for r in turn_records if r.get("abandoned")]
            self.assertEqual(len(abandoned_recs), 1)
            self.assertFalse(abandoned_recs[0].get("ok") is False and abandoned_recs[0].get("error") not in (None,))
            time.sleep(0.3)  # let the server-side write loop hit the broken pipe
            self.assertGreaterEqual(srv.client_aborted, 1, "server never observed the client disconnect")
        finally:
            srv.shutdown()


class TestMixedChat(unittest.TestCase):
    def test_mixed_chat_workers_produce_records(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            cfg = make_cfg(turns=2, tool_latency_range=(0.05, 0.05), mixed_chat=2)
            _turn_records, _sessions, level_summary, mixed_records = bal.run_level(
                1, [one_question()], "P", bal.est_tokens("P"), cfg, seed=13
            )
            self.assertGreater(len(mixed_records), 0)
            self.assertEqual(level_summary["mixed_chat_n_requests"], len(mixed_records))
            self.assertIsNotNone(level_summary["mixed_chat_ttft_p50"])
        finally:
            srv.shutdown()


class TestResumeProbe(unittest.TestCase):
    def test_curve_has_one_entry_per_gap_with_full_trials(self):
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            cfg = make_cfg(tool_result_tokens_range=(10, 10), toolcall_max_tokens=10, final_max_tokens=10)
            records, summary = bal.run_resume_probe("PREFIX", one_question(), [0.05, 0.15], trials=2, cfg=cfg, seed=21)
            self.assertEqual(len(summary["curve"]), 2)
            for row in summary["curve"]:
                self.assertEqual(row["n_ok"], 2)
                self.assertIsNotNone(row["ttft_mean_s"])
            # 2 gaps x 2 trials x 2 turns = 8 turn records
            self.assertEqual(len([r for r in records if r.get("ok")]), 8)
            # turn-2 prompt must extend turn-1 prompt within each trial
            gap_bodies = [c["body"] for c in srv.captured]
            contents = [next(m["content"] for m in b["messages"] if m["role"] == "user") for b in gap_bodies]
            for i in range(0, len(contents), 2):
                self.assertTrue(contents[i + 1].startswith(contents[i]))
        finally:
            srv.shutdown()


class TestMaxModelLenBudget(unittest.TestCase):
    def test_no_budget_check_when_not_specified(self):
        """Default behavior: no budget checking."""
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=3, tool_latency_range=(0.05, 0.05))
            # Note: cfg has max_model_len=None by default (see make_cfg)
            rng = random.Random(20)
            turn_records, summary = bal.run_session(0, 1, one_question(), "PREFIX", bal.est_tokens("PREFIX"), cfg, rng)
            # All turns should succeed or fail only on network issues, not budget
            skipped = [r for r in turn_records if r.get("skipped_reason") == "context_budget_exceeded"]
            self.assertEqual(len(skipped), 0, "no turns should be skipped when max_model_len is None")
            # Transcript should grow normally
            self.assertTrue(summary["completed"] or summary["abandoned"], "session should complete or be abandoned, not budget-blocked")
        finally:
            srv.shutdown()

    def test_budget_exceeded_skips_turn_and_records_skip_reason(self):
        """When max_model_len is set and exceeded, turn is skipped."""
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            # Small max_model_len to trigger budget check early
            cfg = make_cfg(turns=4, tool_latency_range=(0.05, 0.05), tool_result_tokens_range=(50, 50))
            cfg.max_model_len = 70  # Very small limit (forces skip on turn 2+)
            rng = random.Random(21)
            turn_records, summary = bal.run_session(0, 1, one_question(), "PREFIX", bal.est_tokens("PREFIX"), cfg, rng)

            # At least one turn should be skipped due to budget
            skipped = [r for r in turn_records if r.get("skipped_reason") == "context_budget_exceeded"]
            self.assertGreater(len(skipped), 0, "at least one turn should be skipped with small max_model_len")

            # Skipped record should have budget details
            for r in skipped:
                self.assertEqual(r["error"], "context_budget_exceeded")
                self.assertIn("est_prompt_tokens", r)
                self.assertIn("max_tokens", r)
                self.assertIn("max_model_len", r)
                self.assertFalse(r.get("ok"), "skipped turn should have ok=False")

            # Session should not be marked completed (stopped early)
            self.assertFalse(summary["completed"], "session with budget overflow should not be marked completed")
        finally:
            srv.shutdown()

    def test_budget_check_does_not_send_http_request_for_skipped_turn(self):
        """When a turn is skipped, no HTTP request is sent."""
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            import random

            cfg = make_cfg(turns=3, tool_latency_range=(0.05, 0.05), tool_result_tokens_range=(50, 50))
            cfg.max_model_len = 70  # Very small, will trigger early
            rng = random.Random(22)
            turn_records, summary = bal.run_session(0, 1, one_question(), "PREFIX", bal.est_tokens("PREFIX"), cfg, rng)

            # Count how many turns succeeded vs were skipped
            ok_turns = [r for r in turn_records if r.get("ok")]
            skipped_turns = [r for r in turn_records if r.get("skipped_reason") == "context_budget_exceeded"]

            # Each skipped turn means one fewer HTTP request
            # Requests to server = ok turns + retries (none here)
            expected_requests = len(ok_turns)
            actual_requests = len(srv.captured)

            self.assertEqual(
                actual_requests, expected_requests,
                f"HTTP requests sent ({actual_requests}) should match ok turns ({len(ok_turns)}), "
                f"not including skipped turns ({len(skipped_turns)})"
            )
        finally:
            srv.shutdown()

    def test_level_summary_counts_budget_exceeded_turns(self):
        """Level summary includes count of budget-exceeded turns."""
        srv = FakeAgentServer(n_chunks=2)
        point_module_at(srv)
        try:
            questions = [one_question(f"b{i}") for i in range(2)]
            cfg = make_cfg(turns=3, tool_latency_range=(0.05, 0.05), tool_result_tokens_range=(50, 50))
            cfg.max_model_len = 70  # Small limit

            turn_records, session_summaries, level_summary, _mixed = bal.run_level(
                2, questions, "PREFIX", bal.est_tokens("PREFIX"), cfg, seed=23
            )

            # Level summary should report budget-exceeded turns
            self.assertIn("n_turns_skipped_budget_exceeded", level_summary)
            budget_exceeded = level_summary["n_turns_skipped_budget_exceeded"]

            # Count manually from turn records
            manual_count = sum(1 for r in turn_records if r.get("skipped_reason") == "context_budget_exceeded")
            self.assertEqual(budget_exceeded, manual_count)

            # If there are budget-exceeded turns, session(s) should not all be completed
            if budget_exceeded > 0:
                self.assertLess(
                    level_summary["n_sessions_completed"],
                    2,
                    "sessions with budget overflow should not all complete"
                )
        finally:
            srv.shutdown()


class TestCliSmoke(unittest.TestCase):
    def test_synthetic_dry_run_does_not_touch_network(self):
        import contextlib
        import io

        argv_backup = sys.argv
        sys.argv = [
            "bench_agent_loop.py",
            "--synthetic",
            "--dry-run",
            "--turns",
            "4",
            "--tool-latency",
            "0.5-3",
            "--tool-result-tokens",
            "200-800",
        ]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                bal.main()
        finally:
            sys.argv = argv_backup
        self.assertIn("payload constructed OK", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
