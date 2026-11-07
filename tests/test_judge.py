#!/usr/bin/env python3
"""Tests for the checklist LLM judge (judge.py)."""

import os
import sys
import unittest
from unittest import mock
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import judge
import reporting


class TestParseChecklist(unittest.TestCase):
    def test_json_object(self):
        self.assertEqual(judge.parse_checklist('{"1": "yes", "2": "no", "3": "Yes."}', 3), [True, False, True])

    def test_fenced_json_with_prose(self):
        text = 'Here you go:\n```json\n{"1": "no", "2": "yes"}\n```'
        self.assertEqual(judge.parse_checklist(text, 2), [False, True])

    def test_missing_criterion_is_none(self):
        self.assertEqual(judge.parse_checklist('{"1": "yes"}', 2), [True, None])

    def test_numbered_lines_fallback(self):
        self.assertEqual(judge.parse_checklist("1. yes\n2) no", 2), [True, False])

    def test_out_of_range_keys_ignored(self):
        self.assertEqual(judge.parse_checklist('{"7": "yes", "1": "no"}', 1), [False])


class TestJudgeSettings(unittest.TestCase):
    def test_cli_overrides_config_and_env(self):
        with mock.patch.dict(os.environ, {"JUDGE_ENDPOINT": "http://env/v1"}):
            s = judge.resolve_judge_settings("http://cli/v1", "", "", {"judge": {"endpoint": "http://cfg/v1"}})
        self.assertEqual(s["endpoint"], "http://cli/v1")

    def test_config_overrides_env(self):
        with mock.patch.dict(os.environ, {"JUDGE_ENDPOINT": "http://env/v1"}):
            s = judge.resolve_judge_settings("", "", "", {"judge": {"endpoint": "http://cfg/v1", "model": "m"}})
        self.assertEqual((s["endpoint"], s["model"]), ("http://cfg/v1", "m"))

    def test_env_used_last(self):
        with mock.patch.dict(os.environ, {"JUDGE_ENDPOINT": "http://env/v1"}, clear=False):
            self.assertEqual(judge.resolve_judge_settings("", "", "", {})["endpoint"], "http://env/v1")

    def test_unconfigured_judge_means_objective_grading(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            j = judge.build_judge(judge.resolve_judge_settings())
        self.assertFalse(j.configured)
        out = j.checklist("t", "r", ["a", "b"])
        self.assertEqual((out["judged_by"], out["verdicts"], out["failure_kind"]), ("objective", [None, None], None))
        self.assertEqual(j.describe()["objective_reason"], judge.NO_JUDGE_REASON)
        self.assertEqual(j.describe()["items_judged"], {"judge": 0, "objective": 1})


_YES = {"content": '{"1": "yes"}'}
_REFUSED = {"content": "", "error": "Connection refused", "failure_kind": "infra", "endpoint_unreachable": True}


class TestChecklistCall(unittest.TestCase):
    def test_blind_deterministic_checklist(self):
        client = MagicMock()
        client.call.return_value = {"content": '{"1": "yes", "2": "no"}'}
        j = judge.Judge(client, endpoint="e", model="judge-m")
        out = j.checklist("Write an email", "Hi team...", ["Is it polite?", "Does it propose a time?"])
        self.assertEqual((out["passed"], out["total"], out["judged_by"]), (1, 2, "judge"))
        kwargs = client.call.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.0)
        sent = " ".join(m["content"] for m in client.call.call_args[0][0])
        self.assertNotIn("judge-m", sent)  # the judge is not told model identities

    def test_judge_streams_with_a_stall_timeout(self):
        """A non-streamed judge call waits for the whole generation, so a hung
        judge was noticed only after 600 s x 3 attempts."""
        client = MagicMock()
        client.call.return_value = _YES
        judge.Judge(client).checklist("t", "r", ["a"])
        kwargs = client.call.call_args.kwargs
        self.assertEqual((kwargs["stream"], kwargs["timeout"], kwargs["retries"]),
                         (True, judge.JUDGE_STALL_TIMEOUT, 1))


class TestJudgeDownMeansObjective(unittest.TestCase):
    """When the judge goes down, the rest of the run is graded objectively
    (before: the tested model judged its own answers)."""

    def _judge(self, results):
        client = MagicMock()
        client.call.side_effect = results
        return judge.Judge(client, endpoint="http://judge/v1"), client

    def test_unreachable_judge_is_not_called_again(self):
        j, client = self._judge([_REFUSED])
        first, second = j.checklist("t", "r", ["a"]), j.checklist("t", "r", ["a", "b"])
        self.assertEqual(client.call.call_count, 1)
        self.assertEqual((first["judged_by"], second["judged_by"], second["verdicts"]),
                         ("objective", "objective", [None, None]))
        self.assertIn("Connection refused", j.describe()["objective_reason"])
        self.assertEqual(j.describe()["items_judged"], {"judge": 0, "objective": 2})

    def test_judge_that_stops_responding_is_not_called_again(self):
        timeout = {"content": "", "error": "The read operation timed out", "failure_kind": "infra"}
        j, client = self._judge([timeout] * judge.JUDGE_MAX_CONSECUTIVE_FAILURES)
        for _ in range(judge.JUDGE_MAX_CONSECUTIVE_FAILURES + 2):
            j.checklist("t", "r", ["a"])
        self.assertEqual(client.call.call_count, judge.JUDGE_MAX_CONSECUTIVE_FAILURES)
        self.assertIn("in a row", j.describe()["objective_reason"])

    def test_one_server_error_does_not_end_judging(self):
        """A CUDA out-of-memory on one long prompt (v5 run, Test 48) switched
        the whole rest of the run to objective grading."""
        oom = {"content": "", "error": "Stream error: CUDA out of memory", "failure_kind": "infra"}
        j, client = self._judge([oom, _YES, oom, oom, _YES])
        results = [j.checklist("t", "r", ["a"])["judged_by"] for _ in range(5)]
        self.assertEqual(results, ["objective", "judge", "objective", "objective", "judge"])
        self.assertIsNone(j.describe()["objective_reason"])
        self.assertEqual(j.describe()["failed_calls"], 3)
        import report_md
        self.assertIn("3 calls failed", report_md._judge_status({**j.describe(), "model": "m"}))

    def test_missing_judge_model_is_not_called_again(self):
        j, client = self._judge([{"content": "", "error": "HTTPError 404: model not found", "status_code": 404,
                                  "failure_kind": "client"}])
        j.checklist("t", "r", ["a"])
        j.checklist("t", "r", ["a"])
        self.assertEqual(client.call.call_count, 1)

    def test_one_bad_request_makes_only_that_item_objective(self):
        bad = {"content": "", "error": "HTTPError 400: prompt too long", "status_code": 400, "failure_kind": "client"}
        j, _ = self._judge([bad, _YES])
        self.assertEqual(j.checklist("t", "r", ["a"])["judged_by"], "objective")
        self.assertEqual(j.checklist("t", "r", ["a"])["judged_by"], "judge")
        self.assertIsNone(j.describe()["objective_reason"])

    def test_judge_failure_is_never_infra(self):
        """A judge failure is not the tested server's failure, and the item
        still gets a grade (objective), so the test keeps its score."""
        j, _ = self._judge([{"content": "", "error": "HTTPError 503", "status_code": 503, "failure_kind": "infra"}])
        self.assertIsNone(j.checklist("t", "r", ["a"])["failure_kind"])

    def test_uncounted_checks(self):
        j, _ = self._judge([_YES, _REFUSED])
        j.checklist("t", "r", ["a"], count=False)
        j.checklist("t", "r", ["a"], count=False)
        self.assertEqual(j.describe()["items_judged"], {"judge": 0, "objective": 0})


class TestObjectiveLabels(unittest.TestCase):
    def test_mixed_grading_is_labeled(self):
        import suites
        j = MagicMock()
        j.describe.return_value = {}
        res = {"accuracy_pct": 80.0, "status": "PASS", "details": [
            {"attempts": [{"judged_by": "judge"}]}, {"attempts": [{"judged_by": "objective"}]}]}
        out = suites._judged(res, j)
        self.assertEqual(out["judged_items"], {"judge": 1, "objective": 1})
        self.assertTrue(out["partly_objective"])
        self.assertFalse(out["objective_only"])
        self.assertIn("partly no judge", reporting.extract_test_metrics("writing_tasks", out)["desc"])

    def test_all_objective(self):
        import suites
        j = MagicMock()
        j.describe.return_value = {}
        res = {"accuracy_pct": 80.0, "details": [{"attempts": [{"judged_by": "objective"}]}]}
        self.assertTrue(suites._judged(res, j)["objective_only"])

    def test_scorecard_row_keeps_the_label(self):
        """The 24-character column cut ' self-judged' to ' s' (old reports)."""
        import io
        from unittest import mock as m
        data = {"status": "PASS", "accuracy_pct": 94.4, "ci95_low_pct": 83.3, "ci95_high_pct": 100.0,
                "pass_marks": {"pass_pct": 80.0}, "self_judged": True, "tok_s": 27.6}
        out = io.StringIO()
        with m.patch.object(sys, "stdout", out):
            reporting.print_summary_table({"selected_tests": [41], "results": {"writing_tasks": data}})
        row = next(l for l in out.getvalue().splitlines() if l.startswith("41."))
        self.assertTrue(row.endswith("| self-judged"), row)


class TestJudgeComparability(unittest.TestCase):
    def test_different_judges_change_the_fingerprint(self):
        base = {"model": "m", "suite": "writing"}
        a = reporting.build_config_fingerprint(dict(base, judge={"model": "j1", "self_judged": False}))
        b = reporting.build_config_fingerprint(dict(base, judge={"model": "m", "self_judged": True}))
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
