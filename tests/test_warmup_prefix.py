#!/usr/bin/env python3
"""
Local test suite for scripts/warmup_prefix.py -- no real vLLM server needed.

Spins up a tiny http.server.BaseHTTPRequestHandler-based fake OpenAI server
in a background thread for each test, so this exercises the real
requests.get/requests.post calls (readiness polling, payload shape, timing)
against real sockets rather than mocking `requests` itself. Covers:

  (a) wait_for_server retries through connection-refused/503 before the
      fake server starts answering, and returns True once it does.
  (b) wait_for_server returns False (no hang) when nothing is listening at
      all, once --timeout/--retries is exhausted.
  (c) send_warmup_request's payload puts the prefix content in the system
      message (not the user message), and the user message is small.
  (d) --verify: a fake server that answers the 2nd request much faster
      than the 1st -> main() returns 0 (PASS case).
  (e) --verify: a fake server whose 2nd request is NOT faster -> main()
      returns non-zero and prints a warning (cache-miss case).
  (f) exit codes: prefix-file-not-found -> 2, server never ready -> 3,
      warmup request itself erroring (500) -> 4.

Run: python scripts/test_warmup_prefix.py
"""

import http.server
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warmup_prefix as wp  # noqa: E402


def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeVLLMServer:
    """Background HTTP server standing in for vLLM's OpenAI-compatible API.

    `not_ready_for` requests to /health and /v1/models get a 503 before the
    server starts answering 200 (simulates engine still loading).
    `delays` is a list of per-call response delays (seconds) for
    /v1/chat/completions, consumed in order (last value repeats).
    """

    def __init__(self, not_ready_for=0, delays=None, chat_fails_first_n=0):
        self.port = free_port()
        self.not_ready_for = not_ready_for
        self.delays = delays or [0.0]
        self.chat_fails_first_n = chat_fails_first_n
        self._health_calls = 0
        self._chat_calls = 0
        self.last_chat_bodies = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass  # silence

            def do_GET(self):
                if self.path in ("/health", "/v1/models"):
                    outer._health_calls += 1
                    if outer._health_calls <= outer.not_ready_for:
                        self.send_response(503)
                        self.end_headers()
                    else:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"{}")
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/v1/chat/completions":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    outer.last_chat_bodies.append(body)
                    idx = outer._chat_calls
                    outer._chat_calls += 1
                    if idx < outer.chat_fails_first_n:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(b"{}")
                        return
                    delay = outer.delays[min(idx, len(outer.delays) - 1)]
                    time.sleep(delay)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp = {
                        "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 12345, "completion_tokens": 1},
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


class TestWaitForServer(unittest.TestCase):
    def test_retries_until_ready(self):
        srv = FakeVLLMServer(not_ready_for=3)
        try:
            t0 = time.monotonic()
            ok = wp.wait_for_server(srv.base_url, timeout=30)
            elapsed = time.monotonic() - t0
            self.assertTrue(ok)
            # backoff starts at 0.5s and doubles; 3 failed attempts should
            # take at least ~0.5+1.0=1.5s but well under the 30s timeout.
            self.assertGreater(elapsed, 0.3)
            self.assertLess(elapsed, 15)
        finally:
            srv.shutdown()

    def test_never_ready_returns_false_without_hanging(self):
        # nothing listening on this port at all
        port = free_port()
        t0 = time.monotonic()
        ok = wp.wait_for_server(f"http://127.0.0.1:{port}", timeout=2, retries=3)
        elapsed = time.monotonic() - t0
        self.assertFalse(ok)
        self.assertLess(elapsed, 10)


class TestPayloadShape(unittest.TestCase):
    def test_prefix_in_system_message(self):
        srv = FakeVLLMServer()
        try:
            prefix_text = "THIS IS THE LONG PREFIX CONTENT " * 50
            result = wp.send_warmup_request(srv.base_url, prefix_text, "some-model", 1, 30)
            self.assertTrue(result["ok"], result.get("error"))
            self.assertEqual(result["prompt_tokens"], 12345)
            body = srv.last_chat_bodies[0]
            roles = [m["role"] for m in body["messages"]]
            self.assertEqual(roles, ["system", "user"])
            self.assertEqual(body["messages"][0]["content"], prefix_text)
            # user turn must be minimal, not another copy of the prefix
            self.assertLess(len(body["messages"][1]["content"]), 10)
            self.assertEqual(body["max_tokens"], 1)
        finally:
            srv.shutdown()


class TestVerifyMode(unittest.TestCase):
    def _run_main(self, srv, verify=True, extra_args=None):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("prefix content " * 200)
        argv = ["--base-url", srv.base_url, "--prefix-file", path, "--timeout", "10", "--read-timeout", "10"]
        if verify:
            argv.append("--verify")
        if extra_args:
            argv += extra_args
        try:
            return wp.main(argv)
        finally:
            os.remove(path)

    def test_verify_pass_when_second_request_much_faster(self):
        srv = FakeVLLMServer(delays=[0.6, 0.05])
        try:
            rc = self._run_main(srv, verify=True)
            self.assertEqual(rc, 0)
        finally:
            srv.shutdown()

    def test_verify_warns_when_no_speedup(self):
        srv = FakeVLLMServer(delays=[0.2, 0.2])
        try:
            rc = self._run_main(srv, verify=True)
            self.assertNotEqual(rc, 0)
        finally:
            srv.shutdown()


class TestExitCodes(unittest.TestCase):
    def test_missing_prefix_file(self):
        rc = wp.main(["--base-url", "http://127.0.0.1:1", "--prefix-file", "/no/such/file.txt"])
        self.assertEqual(rc, 2)

    def test_server_never_ready(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("prefix")
        port = free_port()
        try:
            rc = wp.main([
                "--base-url", f"http://127.0.0.1:{port}",
                "--prefix-file", path,
                "--timeout", "1",
                "--retries", "1",
            ])
            self.assertEqual(rc, 3)
        finally:
            os.remove(path)

    def test_warmup_request_failure(self):
        srv = FakeVLLMServer(chat_fails_first_n=99)
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("prefix")
        try:
            rc = wp.main([
                "--base-url", srv.base_url,
                "--prefix-file", path,
                "--timeout", "10",
                "--read-timeout", "10",
            ])
            self.assertEqual(rc, 4)
        finally:
            os.remove(path)
            srv.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
