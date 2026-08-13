#!/usr/bin/env python3
"""
Local test suite for scripts/prepare_agent_workload.py -- no network, no GPU,
no real BFCL/SWE-bench/public-test data required.

Builds tiny, hand-written fixtures that copy the *structure* of the real
downloaded BFCL files (verified against gorilla-llm/Berkeley-Function-
Calling-Leaderboard on 2026-08-11 -- see that script's module docstring) but
contain none of the real dataset's content. Covers:

  (a) bfcl multi-turn conversion: turns/expected alignment, tools resolved
      from involved_classes via the func-doc mapping, gt_format="call_string".
  (b) bfcl single-turn conversion: live-style (possible_answer_dict format)
      and irrelevance (gt_format="none", no ground truth file needed).
  (c) public-test conversion: MCQ choices formatted into the user turn.
  (d) --mix ratio: build_sessions() allocates sessions across sources by the
      requested percentage (exact via largest-remainder rounding).
  (e) --seed reproducibility: same seed -> identical output order/content;
      different seed -> different order.
  (f) missing-source handling: a source file that doesn't exist raises
      SourceUnavailable with a clear message (public-test / bfcl-dir); the
      mixed-mode CLI drops a missing source instead of crashing the whole run.

Run: python scripts/test_prepare_agent_workload.py
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "bench", "workload"))
import prepare_agent_workload as paw  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "bench" / "workload" / "prepare_agent_workload.py"


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_fake_bfcl_dir(root: Path) -> Path:
    """Fabricated BFCL directory: real structure (field names, nesting,
    JSONL-despite-.json quirk, multi_turn_func_doc mapping), fake content."""
    bfcl_dir = root / "bfcl"

    # func doc for a made-up class, JSONL of function schemas (structure
    # copied from the real gorilla_file_system.json/etc. files)
    _write_jsonl(bfcl_dir / "multi_turn_func_doc" / "fake_class.json", [
        {
            "name": "do_thing",
            "description": "Do a fake thing.",
            "parameters": {
                "type": "dict",
                "properties": {"x": {"type": "string", "description": "the x"}},
                "required": ["x"],
            },
        },
        {
            "name": "do_other",
            "description": "Do another fake thing.",
            "parameters": {
                "type": "dict",
                "properties": {"y": {"type": "integer", "description": "the y"}},
                "required": ["y"],
            },
        },
    ])

    # multi-turn question file: 2 turns, both user-only (matches the real
    # data's shape: question = list[turn], turn = list[{role, content}])
    _write_jsonl(bfcl_dir / "BFCL_v3_multi_turn_base.json", [
        {
            "id": "multi_turn_fake_0",
            "question": [
                [{"role": "user", "content": "please do the thing with a"}],
                [{"role": "user", "content": "now do nothing"}],
            ],
            "initial_config": {"FakeClass": {}},
            "path": ["FakeClass.do_thing"],
            "involved_classes": ["FakeClass"],
        },
        {
            "id": "multi_turn_fake_1",
            "question": [
                [{"role": "user", "content": "positional style please"}],
            ],
            "initial_config": {"FakeClass": {}},
            "path": ["FakeClass.do_thing"],
            "involved_classes": ["FakeClass"],
        },
    ])
    _write_jsonl(bfcl_dir / "possible_answer" / "BFCL_v3_multi_turn_base.json", [
        {"id": "multi_turn_fake_0", "ground_truth": [["do_thing(x='a')"], []]},
        {"id": "multi_turn_fake_1", "ground_truth": [["do_thing('a')"]]},  # positional-arg ground truth, like real BFCL
    ])

    # live-style single-turn file (possible_answer_dict ground truth format)
    _write_jsonl(bfcl_dir / "BFCL_v3_live_multiple.json", [
        {
            "id": "live_fake_0",
            "question": [[{"role": "user", "content": "please do the other thing with 5"}]],
            "function": [
                {
                    "name": "do_other",
                    "description": "Do another fake thing.",
                    "parameters": {
                        "type": "dict",
                        "properties": {"y": {"type": "integer"}},
                        "required": ["y"],
                    },
                }
            ],
        }
    ])
    _write_jsonl(bfcl_dir / "possible_answer" / "BFCL_v3_live_multiple.json", [
        {"id": "live_fake_0", "ground_truth": [{"do_other": {"y": [5, "5"]}}]},
    ])

    # exec_multiple: ground truth inline, call-string format, single turn
    _write_jsonl(bfcl_dir / "BFCL_v3_exec_multiple.json", [
        {
            "id": "exec_fake_0",
            "question": [[{"role": "user", "content": "compute the fake thing"}]],
            "function": [
                {
                    "name": "do_thing",
                    "description": "Do a fake thing.",
                    "parameters": {"type": "dict", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                }
            ],
            "execution_result_type": ["exact_match"],
            "ground_truth": ["do_thing(x='exec')"],
        }
    ])

    # irrelevance: no ground truth at all -- correct behavior is "call nothing"
    _write_jsonl(bfcl_dir / "BFCL_v3_irrelevance.json", [
        {
            "id": "irr_fake_0",
            "question": [[{"role": "user", "content": "what's the weather like on Mars"}]],
            "function": [
                {
                    "name": "do_thing",
                    "description": "Do a fake thing.",
                    "parameters": {"type": "dict", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                }
            ],
        }
    ])

    return bfcl_dir


def build_fake_public_test(path: Path, n: int = 10) -> None:
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "question": f"fake question number {i}?",
            "choices": ["A choice", "B choice"] if i % 2 == 0 else [],
            "function": [],
            "domain": "fake_domain",
        })
    _write_jsonl(path, rows)


class TestBfclConversion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bfcl_dir = build_fake_bfcl_dir(Path(self.tmp.name))
        # patch CLASS_FILE_MAP so involved_classes=["FakeClass"] resolves to
        # our fabricated fake_class.json rather than a real BFCL class name
        self._orig_map = paw.CLASS_FILE_MAP
        paw.CLASS_FILE_MAP = {"FakeClass": "fake_class.json"}

    def tearDown(self):
        paw.CLASS_FILE_MAP = self._orig_map
        self.tmp.cleanup()

    def test_multi_turn_session_shape(self):
        sessions = paw.load_bfcl_sessions(self.bfcl_dir)
        multi_turn = [s for s in sessions if s["meta"].get("bfcl_subset") == "BFCL_v3_multi_turn_base"]
        by_id = {s["meta"]["bfcl_id"]: s for s in multi_turn}
        self.assertIn("multi_turn_fake_0", by_id)
        s = by_id["multi_turn_fake_0"]

        self.assertEqual(s["source"], "bfcl")
        self.assertEqual(len(s["turns"]), 2)
        self.assertEqual(s["turns"][0]["content"], "please do the thing with a")
        self.assertEqual(s["turns"][1]["content"], "now do nothing")
        self.assertEqual(len(s["expected"]), 2)
        self.assertEqual(s["expected"][0]["gt_format"], "call_string")
        self.assertEqual(s["expected"][0]["calls"], ["do_thing(x='a')"])
        self.assertEqual(s["expected"][1]["calls"], [])  # turn expects zero calls
        self.assertEqual(s["meta"]["n_turns"], 2)
        self.assertEqual(s["meta"]["involved_classes"], ["FakeClass"])

        # tools resolved from involved_classes via func_doc, both functions present
        tool_names = {t["function"]["name"] for t in s["tools"]}
        self.assertEqual(tool_names, {"do_thing", "do_other"})
        self.assertTrue(all(t["type"] == "function" for t in s["tools"]))

    def test_live_multiple_possible_answer_dict(self):
        sessions = paw.load_bfcl_sessions(self.bfcl_dir)
        s = next(s for s in sessions if s["meta"].get("bfcl_id") == "live_fake_0")
        self.assertEqual(len(s["turns"]), 1)
        self.assertEqual(s["expected"][0]["gt_format"], "possible_answer_dict")
        self.assertEqual(s["expected"][0]["calls"], [{"do_other": {"y": [5, "5"]}}])
        self.assertEqual([t["function"]["name"] for t in s["tools"]], ["do_other"])

    def test_exec_multiple_call_string_inline(self):
        sessions = paw.load_bfcl_sessions(self.bfcl_dir)
        s = next(s for s in sessions if s["meta"].get("bfcl_id") == "exec_fake_0")
        self.assertEqual(s["expected"][0]["gt_format"], "call_string")
        self.assertEqual(s["expected"][0]["calls"], ["do_thing(x='exec')"])
        self.assertEqual(s["meta"]["execution_result_type"], ["exact_match"])

    def test_irrelevance_no_ground_truth_means_expect_no_calls(self):
        sessions = paw.load_bfcl_sessions(self.bfcl_dir)
        s = next(s for s in sessions if s["meta"].get("bfcl_id") == "irr_fake_0")
        self.assertEqual(s["expected"][0]["gt_format"], "none")
        self.assertEqual(s["expected"][0]["calls"], [])
        self.assertTrue(s["meta"]["expect_refusal"])

    def test_missing_bfcl_dir_raises_source_unavailable(self):
        missing = Path(self.tmp.name) / "does_not_exist"
        with self.assertRaises(paw.SourceUnavailable):
            paw.load_bfcl_sessions(missing)

    def test_missing_possible_answer_file_skips_row_not_crash(self):
        # Row exists in the question file but not in possible_answer -> the
        # converter should skip that one id rather than emit an unscoreable
        # session or crash the whole load.
        extra_dir = Path(self.tmp.name) / "bfcl_partial"
        _write_jsonl(extra_dir / "multi_turn_func_doc" / "fake_class.json", [
            {"name": "do_thing", "description": "d", "parameters": {"type": "dict", "properties": {}, "required": []}},
        ])
        _write_jsonl(extra_dir / "BFCL_v3_multi_turn_base.json", [
            {"id": "orphan_0", "question": [[{"role": "user", "content": "hi"}]],
             "initial_config": {}, "path": [], "involved_classes": ["FakeClass"]},
        ])
        _write_jsonl(extra_dir / "possible_answer" / "BFCL_v3_multi_turn_base.json", [])
        for fname in ("BFCL_v3_live_multiple.json", "BFCL_v3_exec_multiple.json", "BFCL_v3_irrelevance.json"):
            _write_jsonl(extra_dir / fname, [])
        _write_jsonl(extra_dir / "possible_answer" / "BFCL_v3_live_multiple.json", [])
        with self.assertRaises(paw.SourceUnavailable):
            # every subset ends up empty -> "no sessions could be built" error
            paw.load_bfcl_sessions(extra_dir)


class TestPublicTestConversion(unittest.TestCase):
    def test_mcq_choices_formatted_and_no_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "public-test.jsonl"
            build_fake_public_test(path, n=4)
            sessions = paw.load_public_test_sessions(path)
            self.assertEqual(len(sessions), 4)
            mcq = sessions[0]  # id=0, even -> has choices
            self.assertIn("A. A choice", mcq["turns"][0]["content"])
            self.assertIn("Đáp án:", mcq["turns"][0]["content"])
            self.assertEqual(mcq["expected"][0]["gt_format"], "none")
            self.assertEqual(mcq["tools"], [])

    def test_missing_public_test_file_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.jsonl"
            with self.assertRaises(paw.SourceUnavailable) as ctx:
                paw.load_public_test_sessions(missing)
            self.assertIn("not found", str(ctx.exception))


class TestMixing(unittest.TestCase):
    def test_build_sessions_respects_ratio_exactly(self):
        rng = random.Random(0)
        pools = {
            "bfcl": [{"source": "bfcl", "id": i} for i in range(100)],
            "public-test": [{"source": "public-test", "id": i} for i in range(100)],
            "swebench": [{"source": "swebench", "id": i} for i in range(100)],
        }
        mix = {"bfcl": 50, "public-test": 30, "swebench": 20}
        sessions = paw.build_sessions(pools, mix, 40, rng)
        self.assertEqual(len(sessions), 40)
        counts = {"bfcl": 0, "public-test": 0, "swebench": 0}
        for s in sessions:
            counts[s["source"]] += 1
        self.assertEqual(counts, {"bfcl": 20, "public-test": 12, "swebench": 8})

    def test_build_sessions_drops_unavailable_source_and_renormalizes(self):
        rng = random.Random(0)
        pools = {
            "bfcl": [{"source": "bfcl", "id": i} for i in range(100)],
            "public-test": [{"source": "public-test", "id": i} for i in range(100)],
            # swebench NOT in pools (simulates it being unavailable / dropped upstream)
        }
        mix = {"bfcl": 50, "public-test": 30, "swebench": 20}
        sessions = paw.build_sessions(pools, mix, 40, rng)
        self.assertEqual(len(sessions), 40)
        self.assertTrue(all(s["source"] in ("bfcl", "public-test") for s in sessions))
        counts = {"bfcl": 0, "public-test": 0}
        for s in sessions:
            counts[s["source"]] += 1
        # 50:30 renormalized over 80 -> 62.5%/37.5% of 40 = 25/15
        self.assertEqual(counts, {"bfcl": 25, "public-test": 15})

    def test_build_sessions_no_available_sources_raises(self):
        rng = random.Random(0)
        with self.assertRaises(paw.SourceUnavailable):
            paw.build_sessions({}, {"bfcl": 100}, 10, rng)

    def test_parse_mix(self):
        self.assertEqual(paw.parse_mix("bfcl:50,public-test:30,swebench:20"),
                          {"bfcl": 50.0, "public-test": 30.0, "swebench": 20.0})
        with self.assertRaises(ValueError):
            paw.parse_mix("bfcl-50")


class TestSeedReproducibility(unittest.TestCase):
    def _pools(self):
        return {
            "bfcl": [{"source": "bfcl", "id": i} for i in range(50)],
            "public-test": [{"source": "public-test", "id": i} for i in range(50)],
        }

    def test_same_seed_same_output(self):
        mix = {"bfcl": 60, "public-test": 40}
        out1 = paw.build_sessions(self._pools(), mix, 20, random.Random(42))
        out2 = paw.build_sessions(self._pools(), mix, 20, random.Random(42))
        self.assertEqual(out1, out2)

    def test_different_seed_different_order(self):
        mix = {"bfcl": 60, "public-test": 40}
        out1 = paw.build_sessions(self._pools(), mix, 20, random.Random(1))
        out2 = paw.build_sessions(self._pools(), mix, 20, random.Random(2))
        self.assertNotEqual([s["id"] for s in out1], [s["id"] for s in out2])


class TestCliEndToEnd(unittest.TestCase):
    """Exercises main() through subprocess so argparse/exit-code behavior is
    covered too, not just the library functions."""

    def test_single_source_missing_file_exits_nonzero_with_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.jsonl"
            missing = Path(tmp) / "no_public_test.jsonl"
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--source", "public-test",
                 "--public-test-file", str(missing), "--output", str(out_path)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("public-test", result.stderr)
            self.assertIn("not found", result.stderr)
            self.assertFalse(out_path.exists())

    def test_public_test_source_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = Path(tmp) / "public-test.jsonl"
            build_fake_public_test(pt_path, n=6)
            out_path = Path(tmp) / "out.jsonl"
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--source", "public-test",
                 "--public-test-file", str(pt_path), "--sessions", "6",
                 "--seed", "0", "--output", str(out_path)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out_path.exists())
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 6)
            for line in lines:
                row = json.loads(line)
                self.assertEqual(row["source"], "public-test")


if __name__ == "__main__":
    unittest.main()
