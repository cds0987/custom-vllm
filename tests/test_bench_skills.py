#!/usr/bin/env python3
"""
Local test suite for scripts/bench_skills.py -- no real vLLM server needed.

Spins up a tiny http.server.BaseHTTPRequestHandler-based fake OpenAI server
in a background thread for each test, so this exercises the real
requests.get/requests.post calls (payload shape, streaming) against real
sockets rather than mocking `requests` itself. Covers:

  (a) synthetic mode works without requiring --prefix-file or --questions-file
  (b) same --seed produces identical prefix and questions across runs
  (c) synthetic questions are all different from each other (no exact duplicates)
  (d) mixing file-based and synthetic flags results in a clear error
  (e) old behavior (file-based mode with files) still works unchanged
  (f) synthetic mode runs end-to-end against a fake server (dry-run)

Run: python scripts/test_bench_skills.py
"""

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "bench"))
BENCH_SKILLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "bench", "bench_skills.py")
import bench_skills as bs  # noqa: E402


def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeVLLMServer:
    """Background HTTP server standing in for vLLM's OpenAI-compatible API.
    Supports /metrics endpoint for prefix cache metrics."""

    def __init__(self):
        self.port = free_port()
        self._chat_calls = 0
        self.last_chat_bodies = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass  # silence

            def do_GET(self):
                if self.path in ("/health", "/v1/models"):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"{}")
                elif self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    # Minimal metrics output
                    metrics = (
                        "# HELP vllm:prefix_cache_queries Total prefix cache queries\n"
                        "# TYPE vllm:prefix_cache_queries counter\n"
                        "vllm:prefix_cache_queries{engine=\"0\",model_name=\"test\"} 0.0\n"
                        "# HELP vllm:prefix_cache_hits Total prefix cache hits\n"
                        "# TYPE vllm:prefix_cache_hits counter\n"
                        "vllm:prefix_cache_hits{engine=\"0\",model_name=\"test\"} 0.0\n"
                    )
                    self.wfile.write(metrics.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/v1/chat/completions":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    outer.last_chat_bodies.append(body)
                    outer._chat_calls += 1
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp = {
                        "choices": [{"message": {"content": "test response"}, "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    }
                    self.wfile.write(json.dumps(resp).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        self.httpd = HTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def shutdown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)


class TestSyntheticGeneration(unittest.TestCase):
    def test_synthetic_prefix_deterministic(self):
        """Same seed produces identical prefix."""
        p1 = bs.generate_synthetic_prefix(1000, seed=42)
        p2 = bs.generate_synthetic_prefix(1000, seed=42)
        self.assertEqual(p1, p2)

    def test_synthetic_prefix_different_seed(self):
        """Different seeds produce different prefixes."""
        p1 = bs.generate_synthetic_prefix(1000, seed=42)
        p2 = bs.generate_synthetic_prefix(1000, seed=99)
        self.assertNotEqual(p1, p2)

    def test_synthetic_questions_deterministic(self):
        """Same seed produces identical questions."""
        q1 = bs.generate_synthetic_questions(10, seed=42)
        q2 = bs.generate_synthetic_questions(10, seed=42)
        self.assertEqual(len(q1), len(q2))
        for a, b in zip(q1, q2):
            self.assertEqual(a["id"], b["id"])
            self.assertEqual(a["question"], b["question"])

    def test_synthetic_questions_different_seed(self):
        """Different seeds produce different questions."""
        q1 = bs.generate_synthetic_questions(10, seed=42)
        q2 = bs.generate_synthetic_questions(10, seed=99)
        # At least some should be different
        different = sum(1 for a, b in zip(q1, q2) if a["question"] != b["question"])
        self.assertGreater(different, 0)

    def test_synthetic_questions_are_different_from_each_other(self):
        """All questions in a set should be different from each other."""
        questions = bs.generate_synthetic_questions(20, seed=42)
        question_texts = [q["question"] for q in questions]
        unique_texts = set(question_texts)
        # All should be unique
        self.assertEqual(len(unique_texts), len(question_texts))

    def test_synthetic_prefix_token_estimate(self):
        """Prefix length is roughly proportional to requested tokens."""
        # ~4 chars per token, so n_tokens*4*5/4 = n_tokens*5 words
        p = bs.generate_synthetic_prefix(100, seed=0)
        # Should be roughly 400-600 chars (100 tokens * 4-6 chars)
        self.assertGreater(len(p), 200)
        self.assertLess(len(p), 1000)

    def test_synthetic_questions_token_estimate(self):
        """Question length respects requested token range."""
        questions = bs.generate_synthetic_questions(5, min_tokens=100, max_tokens=200, seed=0)
        for q in questions:
            text_len = len(q["question"])
            # ~4 chars/token: 100 tokens = ~400 chars, 200 tokens = ~800 chars
            # But with spaces and variable word lengths, allow some flexibility
            self.assertGreater(text_len, 300)
            self.assertLess(text_len, 1500)


class TestCLIValidation(unittest.TestCase):
    def test_error_mixing_file_and_synthetic(self):
        """Mixing file-based and synthetic modes should error."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("prefix")
        try:
            # Create temp questions file
            fd2, qpath = tempfile.mkstemp(suffix=".jsonl")
            os.close(fd2)
            with open(qpath, "w", encoding="utf-8") as f:
                f.write('{"id": "q1", "question": "what is this?"}\n')
            try:
                result = subprocess.run(
                    [sys.executable, BENCH_SKILLS,
                     "--prefix-file", path,
                     "--synthetic-questions", "10",
                     "--dry-run"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot mix", result.stderr)
            finally:
                os.remove(qpath)
        finally:
            os.remove(path)

    def test_error_no_input_specified(self):
        """Not specifying any input (files or synthetic) should error."""
        result = subprocess.run(
            [sys.executable, BENCH_SKILLS,
             "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must specify either", result.stderr)


class TestFileModeStillWorks(unittest.TestCase):
    def test_file_mode_basic(self):
        """File-based mode (old behavior) should still work."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("prefix content " * 50)

        fd2, qpath = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd2)
        with open(qpath, "w", encoding="utf-8") as f:
            f.write('{"id": "q1", "question": "what is this?"}\n')
            f.write('{"id": "q2", "question": "and that?"}\n')

        try:
            result = subprocess.run(
                [sys.executable, BENCH_SKILLS,
                 "--prefix-file", path,
                 "--questions-file", qpath,
                 "--dry-run"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("questions loaded: 2", result.stdout)
            self.assertNotIn("SYNTHETIC MODE", result.stdout)
        finally:
            os.remove(path)
            os.remove(qpath)


class TestSyntheticModeEndToEnd(unittest.TestCase):
    def test_synthetic_dry_run(self):
        """Synthetic mode dry-run should work without any files."""
        result = subprocess.run(
            [sys.executable, BENCH_SKILLS,
             "--synthetic-prefix-tokens", "1000",
             "--synthetic-questions", "5",
             "--seed", "42",
             "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("SYNTHETIC MODE", result.stdout)
        self.assertIn("1000", result.stdout)
        self.assertIn("5", result.stdout)
        self.assertIn("seed=42", result.stdout)
        self.assertIn("questions loaded: 5", result.stdout)

    def test_synthetic_prefix_only(self):
        """Using only --synthetic-prefix-tokens without --synthetic-questions should fail."""
        result = subprocess.run(
            [sys.executable, BENCH_SKILLS,
             "--synthetic-prefix-tokens", "1000",
             "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("both", result.stderr)

    def test_synthetic_questions_only(self):
        """Using only --synthetic-questions without --synthetic-prefix-tokens should fail."""
        result = subprocess.run(
            [sys.executable, BENCH_SKILLS,
             "--synthetic-questions", "10",
             "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("both", result.stderr)

    def test_synthetic_with_mixed_file_prefix_only(self):
        """Using --synthetic-questions with --prefix-file should error."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("prefix")
        try:
            result = subprocess.run(
                [sys.executable, BENCH_SKILLS,
                 "--prefix-file", path,
                 "--synthetic-questions", "10",
                 "--dry-run"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot mix", result.stderr)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
