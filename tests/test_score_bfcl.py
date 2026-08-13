#!/usr/bin/env python3
"""
Local test suite for scripts/score_bfcl.py -- no network, no GPU, no real
BFCL data required (all fixtures are hand-written, matching the two real
ground-truth formats documented in that script's module docstring).

Covers the matching cases the task brief called out explicitly:
  - exact match (call_string and possible_answer_dict formats)
  - name mismatch / wrong value -> no match
  - missing required parameter -> no match
  - extra (non-required) predicted parameter -> ignored, still matches
  - positional-arg ground truth resolved via the tool schema's param order
  - "none" ground truth (irrelevance / no-call turn): predicting nothing
    scores 1.0, predicting anything scores 0.0
  - numeric/string normalization (1 == 1.0 == "1"; case/whitespace folding)
  - unparseable call strings counted as a miss, not a crash

Run: python scripts/test_score_bfcl.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "bench", "workload"))
import score_bfcl as sb  # noqa: E402


class TestParseCallString(unittest.TestCase):
    def test_keyword_args(self):
        name, kwargs, positional = sb.parse_call_string("cd(folder='document')")
        self.assertEqual(name, "cd")
        self.assertEqual(kwargs, {"folder": "document"})
        self.assertEqual(positional, [])

    def test_positional_arg(self):
        name, kwargs, positional = sb.parse_call_string("activateParkingBrake('engage')")
        self.assertEqual(name, "activateParkingBrake")
        self.assertEqual(kwargs, {})
        self.assertEqual(positional, ["engage"])

    def test_arithmetic_expression_in_kwarg(self):
        name, kwargs, positional = sb.parse_call_string("f(p=1/6)")
        self.assertEqual(name, "f")
        self.assertAlmostEqual(kwargs["p"], 1 / 6)

    def test_list_and_dict_literals(self):
        name, kwargs, _ = sb.parse_call_string("g(a=[1, 2], b={'k': 'v'})")
        self.assertEqual(kwargs["a"], [1, 2])
        self.assertEqual(kwargs["b"], {"k": "v"})

    def test_unsupported_name_lookup_raises(self):
        with self.assertRaises(sb.UnsupportedExpression):
            sb.parse_call_string("f(x=some_variable)")


class TestResolvePositional(unittest.TestCase):
    def test_resolved_via_param_order(self):
        param_order = {"activateParkingBrake": ["mode"]}
        merged = sb.resolve_positional("activateParkingBrake", ["engage"], {}, param_order)
        self.assertEqual(merged, {"mode": "engage"})

    def test_unresolved_falls_back_to_synthetic_key(self):
        merged = sb.resolve_positional("mystery", ["engage"], {}, {})
        self.assertEqual(merged, {"_pos0": "engage"})

    def test_no_positional_args_passthrough(self):
        merged = sb.resolve_positional("f", [], {"x": 1}, {})
        self.assertEqual(merged, {"x": 1})


class TestValuesMatch(unittest.TestCase):
    def test_numeric_cross_type(self):
        self.assertTrue(sb._values_match(1, 1.0))
        self.assertTrue(sb._values_match("1", 1))
        self.assertTrue(sb._values_match(1.0, "1.0"))

    def test_string_case_and_whitespace_fold(self):
        self.assertTrue(sb._values_match("  Coconut ", "coconut"))
        self.assertFalse(sb._values_match("coconut milk", "coconut"))

    def test_bool_vs_string_not_coerced_through_numeric_path(self):
        # True == 1 is a Python truth, caught by the direct `==` shortcut
        # (fine — that's real equality, not the numeric-coercion path).
        # What the numeric guard actually prevents is a bool matching a
        # NON-1/0 numeric-looking string via coercion.
        self.assertFalse(sb._values_match(True, "yes"))
        self.assertTrue(sb._values_match(True, 1))

    def test_list_values(self):
        self.assertTrue(sb._values_match(["a", "b"], ["a", "b"]))
        self.assertTrue(sb._values_match([1, 2], [1.0, "2"]))  # elementwise numeric coercion
        self.assertFalse(sb._values_match(["a", "b"], ["a"]))  # length mismatch


class TestScoreTurnCallString(unittest.TestCase):
    def test_exact_match(self):
        r = sb.score_turn("call_string", ["cd(folder='document')"], ["cd(folder='document')"], {})
        self.assertEqual(r["turn_score"], 1.0)
        self.assertEqual(r["n_matched"], 1)

    def test_wrong_value_no_match(self):
        r = sb.score_turn("call_string", ["cd(folder='document')"], ["cd(folder='other')"], {})
        self.assertEqual(r["turn_score"], 0.0)

    def test_wrong_name_no_match(self):
        r = sb.score_turn("call_string", ["cd(folder='document')"], ["ls(folder='document')"], {})
        self.assertEqual(r["turn_score"], 0.0)

    def test_missing_required_param_no_match(self):
        r = sb.score_turn("call_string", ["mv(source='a', destination='b')"], ["mv(source='a')"], {})
        self.assertEqual(r["turn_score"], 0.0)

    def test_extra_predicted_param_ignored(self):
        r = sb.score_turn("call_string", ["cd(folder='document')"], ["cd(folder='document', extra=1)"], {})
        self.assertEqual(r["turn_score"], 1.0)

    def test_positional_ground_truth_resolved_via_param_order(self):
        param_order = {"activateParkingBrake": ["mode"]}
        r = sb.score_turn(
            "call_string",
            ["activateParkingBrake('engage')"],
            [{"name": "activateParkingBrake", "arguments": {"mode": "engage"}}],
            param_order,
        )
        self.assertEqual(r["turn_score"], 1.0)

    def test_positional_ground_truth_unresolved_without_schema_is_conservative_miss(self):
        r = sb.score_turn(
            "call_string",
            ["activateParkingBrake('engage')"],
            [{"name": "activateParkingBrake", "arguments": {"mode": "engage"}}],
            {},  # no schema -> can't map positional arg 0 to "mode"
        )
        self.assertEqual(r["turn_score"], 0.0)

    def test_unparseable_predicted_call_is_a_miss_not_a_crash(self):
        r = sb.score_turn("call_string", ["cd(folder='document')"], ["not a call at all!!"], {})
        self.assertEqual(r["turn_score"], 0.0)
        self.assertEqual(r["n_predicted"], 1)

    def test_multiple_calls_greedy_match(self):
        expected = ["a(x=1)", "b(y=2)"]
        predicted = [{"name": "b", "arguments": {"y": 2}}, {"name": "a", "arguments": {"x": 1}}]
        r = sb.score_turn("call_string", expected, predicted, {})
        self.assertEqual(r["turn_score"], 1.0)
        self.assertEqual(r["n_matched"], 2)


class TestScoreTurnPossibleAnswerDict(unittest.TestCase):
    def test_exact_and_alternative_match(self):
        expected = [{"do_other": {"y": [5, "5"]}}]
        r = sb.score_turn("possible_answer_dict", expected, [{"name": "do_other", "arguments": {"y": "5"}}], {})
        self.assertEqual(r["turn_score"], 1.0)

    def test_value_outside_alternatives_no_match(self):
        expected = [{"do_other": {"y": [5, "5"]}}]
        r = sb.score_turn("possible_answer_dict", expected, [{"name": "do_other", "arguments": {"y": 6}}], {})
        self.assertEqual(r["turn_score"], 0.0)

    def test_missing_param_no_match(self):
        expected = [{"do_other": {"y": [5]}}]
        r = sb.score_turn("possible_answer_dict", expected, [{"name": "do_other", "arguments": {}}], {})
        self.assertEqual(r["turn_score"], 0.0)


class TestScoreTurnNone(unittest.TestCase):
    def test_correct_refusal(self):
        r = sb.score_turn("none", [], [], {})
        self.assertEqual(r["turn_score"], 1.0)

    def test_incorrect_call_when_none_expected(self):
        r = sb.score_turn("none", [], [{"name": "anything", "arguments": {}}], {})
        self.assertEqual(r["turn_score"], 0.0)


class TestScoreSession(unittest.TestCase):
    def _session(self):
        return {
            "session_id": "bfcl:fake:0",
            "source": "bfcl",
            "tools": [{"type": "function", "function": {
                "name": "activateParkingBrake",
                "parameters": {"type": "dict", "properties": {"mode": {"type": "string"}}, "required": ["mode"]},
            }}],
            "expected": [
                {"turn_index": 0, "gt_format": "call_string", "calls": ["activateParkingBrake('engage')"]},
                {"turn_index": 1, "gt_format": "none", "calls": []},
            ],
        }

    def test_full_session_perfect_score(self):
        predicted = [
            [{"name": "activateParkingBrake", "arguments": {"mode": "engage"}}],
            [],
        ]
        result = sb.score_session(self._session(), predicted)
        self.assertEqual(result["session_score"], 1.0)
        self.assertEqual(result["n_turns_scored"], 2)

    def test_partial_session_score(self):
        predicted = [
            [{"name": "activateParkingBrake", "arguments": {"mode": "wrong"}}],
            [],
        ]
        result = sb.score_session(self._session(), predicted)
        self.assertEqual(result["session_score"], 0.5)

    def test_text_patch_turns_excluded_from_scoring(self):
        session = {
            "session_id": "swebench:1",
            "source": "swebench",
            "tools": [],
            "expected": [{"turn_index": 0, "gt_format": "text_patch", "calls": ["diff --git a b"]}],
        }
        result = sb.score_session(session, [[]])
        self.assertIsNone(result["session_score"])
        self.assertEqual(result["n_turns_scored"], 0)


if __name__ == "__main__":
    unittest.main()
