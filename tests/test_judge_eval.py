#!/usr/bin/env python3
"""Regression tests for the standalone quality-judging CLI."""

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import client as client_module  # noqa: E402
import judge_eval  # noqa: E402


class TestJudgeRequests(unittest.TestCase):
    def test_average_speed_includes_measured_zero_but_not_missing_values(self):
        records = [{"server1": {"tok_s": value}} for value in (0.0, 10.0, None)]
        self.assertEqual(judge_eval._average_speed(records, "server1"), 5.0)

    def test_unreachable_server_is_not_retried(self):
        response = {"error": "connection refused", "failure_kind": "infra", "endpoint_unreachable": True}
        with mock.patch.object(judge_eval, "LLMClient") as factory, \
             mock.patch.object(judge_eval.time, "sleep") as sleep, \
             mock.patch("builtins.print"):
            factory.return_value.call.return_value = response
            result = judge_eval.send_chat_completion_robust("http://server/v1", "model", "", [])
        self.assertTrue(result[0].startswith("[ERROR]:"))
        self.assertEqual(factory.return_value.call.call_count, 1)
        sleep.assert_not_called()

    def test_retryable_failures_do_not_sleep_after_the_last_attempt(self):
        response = {"error": "timeout", "failure_kind": "infra"}
        with mock.patch.object(judge_eval, "LLMClient") as factory, \
             mock.patch.object(judge_eval.time, "sleep") as sleep, \
             mock.patch("builtins.print"):
            factory.return_value.call.return_value = response
            judge_eval.send_chat_completion_robust("http://server/v1", "model", "", [], retries=3)
        self.assertEqual(factory.return_value.call.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


class TestJudgeSyntax(unittest.TestCase):
    def test_missing_code_is_not_valid_python(self):
        valid, reason = judge_eval.test_code_syntax("I cannot provide an implementation.")
        self.assertFalse(valid)
        self.assertIn("code", reason.lower())

    def test_unparseable_unicode_is_not_valid_python(self):
        valid, reason = judge_eval.test_code_syntax("```python\nx = '\ud800'\n```")
        self.assertFalse(valid)
        self.assertIn("syntax", reason.lower())

    def test_valid_python_is_still_accepted(self):
        self.assertTrue(judge_eval.test_code_syntax("```python\ndef f():\n    return 1\n```")[0])

    def test_bare_valid_python_without_function_or_import_is_accepted(self):
        for answer in ("x = 1", "pass", "print('hello')"):
            with self.subTest(answer=answer):
                self.assertTrue(judge_eval.test_code_syntax(answer)[0])


class TestJudgeServerFailures(unittest.TestCase):
    def test_failed_answer_does_not_invent_a_score_for_the_other_model(self):
        with mock.patch.object(judge_eval, "send_chat_completion_robust") as send:
            result = judge_eval.run_llm_judge_10_pair(
                "http://judge/v1", "judge", "", {}, "[ERROR]: timeout", "an answer", "one", "two")
        self.assertIsNone(result["m1_score_10"])
        self.assertIsNone(result["m2_score_10"])
        self.assertEqual(result["winner"], "SERVER ERROR")
        send.assert_not_called()

    def test_empty_answer_scores_zero_and_the_other_answer_is_still_judged(self):
        response = '{"model_a_total_10": 7, "model_b_total_10": 1}'
        task = {"name": "task", "prompt": "prompt"}
        with mock.patch.object(judge_eval, "send_chat_completion_robust",
                               return_value=(response, 0.0, 0, 0.0)) as send:
            result = judge_eval.run_llm_judge_10_pair(
                "http://judge/v1", "judge", "", task, "   ", "an answer", "one", "two")
        self.assertEqual(send.call_count, 2)
        self.assertEqual(result["m1_score_10"], 0.0)
        self.assertEqual(result["m2_score_10"], 4.0)
        self.assertEqual(result["winner"], "Server 2 (two)")
        self.assertIn(judge_eval.EMPTY_ANSWER_TEXT, send.call_args_list[0][0][3][1]["content"])

    def test_two_empty_answers_score_zero_without_judge_calls(self):
        with mock.patch.object(judge_eval, "send_chat_completion_robust") as send:
            result = judge_eval.run_llm_judge_10_pair(
                "http://judge/v1", "judge", "", {}, "", "", "one", "two")
        self.assertEqual((result["m1_score_10"], result["m2_score_10"]), (0.0, 0.0))
        send.assert_not_called()

    def test_empty_answer_ranks_both_models_in_the_report(self):
        task = {"id": 1, "name": "task", "category": "code", "system": "Python",
                "prompt": "write Python code", "max_tokens": 10, "temperature": 0.0}
        judged = '{"model_a_total_10": 8, "model_b_total_10": 2}'
        responses = [("", 1.0, 10, 10.0), ("x = 1", 1.0, 10, 10.0),
                     (judged, 0.0, 0, 0.0), (judged, 0.0, 0, 0.0)]
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [task]), \
             mock.patch.object(judge_eval, "send_chat_completion_robust", side_effect=responses), \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            judge_eval.run_hard_complex_benchmark(
                "http://one/v1", "one", "", "http://two/v1", "two", "", out,
                judge_endpoint="http://judge/v1", judge_model="judge")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(report["tasks"][0]["server1"]["score_10"], 0.0)
        self.assertEqual(report["tasks"][0]["server2"]["score_10"], 5.0)
        ranks = {entry["model"]: entry["rank"] for entry in report["leaderboard"]}
        self.assertEqual(ranks, {"two": 1, "one": 2})

    def test_empty_single_answer_scores_zero(self):
        task = {"id": 1, "name": "task", "category": "code", "system": "Python",
                "prompt": "write Python code", "max_tokens": 10, "temperature": 0.0}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [task]), \
             mock.patch.object(judge_eval, "send_chat_completion_robust",
                               return_value=("", 1.0, 10, 10.0)) as send, \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            judge_eval.run_hard_complex_benchmark(
                "http://server/v1", "model", "", out_file=out,
                judge_endpoint="http://judge/v1", judge_model="judge")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(report["tasks"][0]["server1"]["score_10"], 0.0)
        self.assertEqual(report["leaderboard"][0]["score_10"], 0.0)

    def test_error_text_within_a_valid_answer_is_still_judged(self):
        response = '{"model_a_total_10": 7, "model_b_total_10": 6}'
        task = {"name": "task", "prompt": "prompt"}
        with mock.patch.object(judge_eval, "send_chat_completion_robust",
                               return_value=(response, 0.0, 0, 0.0)) as send:
            result = judge_eval.run_llm_judge_10_pair(
                "http://judge/v1", "judge", "", task, 'print("[ERROR]")', "other answer", "one", "two")
        self.assertEqual(send.call_count, 2)
        self.assertEqual((result["m1_score_10"], result["m2_score_10"]), (6.5, 6.5))

    def test_pair_judge_does_not_send_second_round_after_server_is_down(self):
        error = urllib.error.URLError(ConnectionRefusedError(111, "connection refused"))
        with mock.patch.object(client_module.urllib.request, "urlopen", side_effect=error) as urlopen, \
             mock.patch.object(client_module, "UNREACHABLE_RETRY_DELAYS", ()), \
             mock.patch("builtins.print"):
            result = judge_eval.run_llm_judge_10_pair(
                "http://judge/v1", "judge", "", {"name": "task", "prompt": "prompt"},
                "one answer", "other answer", "one", "two")
        self.assertEqual(result["winner"], "JUDGE ERROR")
        self.assertEqual(urlopen.call_count, 1)

    def test_unavailable_single_model_is_unscored_in_the_report(self):
        task = {"id": 1, "name": "task", "category": "code", "system": "Python",
                "prompt": "write Python code", "max_tokens": 10, "temperature": 0.0}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [task]), \
             mock.patch.object(judge_eval, "send_chat_completion_robust",
                               return_value=("[ERROR]: timeout", 0.0, 0, 0.0)) as send, \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            judge_eval.run_hard_complex_benchmark(
                "http://server/v1", "model", "", out_file=out,
                judge_endpoint="http://judge/v1", judge_model="judge")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
            with open(out, encoding="utf-8") as f:
                markdown = f.read()
        self.assertEqual(send.call_count, 1)
        self.assertIsNone(report["tasks"][0]["server1"]["score_10"])
        self.assertIsNone(report["tasks"][0]["server1"]["syntax_ok"])
        self.assertIsNone(report["leaderboard"][0]["score_10"])
        self.assertIsNone(report["leaderboard"][0]["rank"])
        self.assertEqual(report["leaderboard"][0]["tier"], "Unscored")
        self.assertIsNone(report["tasks"][0]["server1"]["tok_s"])
        self.assertIsNone(report["tasks"][0]["server1"]["wall_s"])
        self.assertIsNone(report["tasks"][0]["server1"]["tokens"])
        self.assertIsNone(report["leaderboard"][0]["tok_s"])
        self.assertIn("Unscored", markdown)
        self.assertIn("N/A", markdown)
        self.assertIn("Output Throughput", markdown)
        self.assertNotIn("Decode Throughput", markdown)

    def test_a_failed_server_leaves_both_pair_scores_unscored(self):
        task = {"id": 1, "name": "task", "category": "code", "system": "Python",
                "prompt": "write Python code", "max_tokens": 10, "temperature": 0.0}
        responses = [("x = 1", 1.0, 10, 10.0), ("[ERROR]: timeout", 0.0, 0, 0.0)]
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [task]), \
             mock.patch.object(judge_eval, "send_chat_completion_robust", side_effect=responses) as send, \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            judge_eval.run_hard_complex_benchmark(
                "http://one/v1", "one", "", "http://two/v1", "two", "", out,
                judge_endpoint="http://judge/v1", judge_model="judge")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(send.call_count, 2)
        self.assertIsNone(report["tasks"][0]["server1"]["score_10"])
        self.assertIsNone(report["tasks"][0]["server2"]["score_10"])
        self.assertEqual(report["tasks"][0]["winner"], "SERVER ERROR")
        self.assertTrue(all(entry["rank"] is None for entry in report["leaderboard"]))
        speeds = {entry["model"]: entry["tok_s"] for entry in report["leaderboard"]}
        self.assertEqual(speeds, {"one": 10.0, "two": None})

    def test_partial_run_does_not_rank_a_model_on_an_incomplete_task_set(self):
        task = {"id": 1, "name": "task", "category": "code", "system": "Python",
                "prompt": "write Python code", "max_tokens": 10, "temperature": 0.0}
        responses = [
            ("x = 1", 1.0, 10, 10.0),
            ('{"score_10": 8}', 1.0, 10, 10.0),
            ("[ERROR]: timeout", 0.0, 0, 0.0),
        ]
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [task, dict(task, id=2)]), \
             mock.patch.object(judge_eval, "send_chat_completion_robust", side_effect=responses), \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            judge_eval.run_hard_complex_benchmark(
                "http://server/v1", "model", "", out_file=out,
                judge_endpoint="http://judge/v1", judge_model="judge")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(report["tasks"][0]["server1"]["score_10"], 8)
        self.assertIsNone(report["tasks"][1]["server1"]["score_10"])
        self.assertIsNone(report["leaderboard"][0]["score_10"])
        self.assertIsNone(report["leaderboard"][0]["rank"])
        self.assertEqual(report["leaderboard"][0]["scored_tasks"], 1)
        self.assertEqual(report["leaderboard"][0]["total_tasks"], 2)
        self.assertEqual(report["leaderboard"][0]["tok_s"], 10.0)

    def test_unreachable_server_stops_remaining_tasks_and_writes_incomplete_report(self):
        task = {"id": 1, "name": "task", "category": "code", "system": "Python",
                "prompt": "write Python code", "max_tokens": 10, "temperature": 0.0}
        refused = urllib.error.URLError(ConnectionRefusedError(111, "connection refused"))
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [task, dict(task, id=2)]), \
             mock.patch.object(client_module.urllib.request, "urlopen", side_effect=refused) as urlopen, \
             mock.patch.object(client_module, "UNREACHABLE_RETRY_DELAYS", ()), \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            completed = judge_eval.run_hard_complex_benchmark(
                "http://server/v1", "model", "", out_file=out,
                judge_endpoint="http://judge/v1", judge_model="judge")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(urlopen.call_count, 1)
        self.assertFalse(completed)
        self.assertTrue(report["incomplete"])
        self.assertEqual(report["not_run"], [2])
        self.assertEqual(len(report["tasks"]), 1)
        self.assertIsNone(report["leaderboard"][0]["rank"])

    def test_unreachable_judge_stops_before_the_next_model_task(self):
        task = {"id": 1, "name": "task", "category": "code", "system": "Python",
                "prompt": "write Python code", "max_tokens": 10, "temperature": 0.0}
        requests = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def urlopen(req, timeout=None):
            requests.append(req.full_url)
            if "judge" in req.full_url:
                raise urllib.error.URLError(ConnectionRefusedError(111, "connection refused"))
            return Response(b'{"choices":[{"message":{"content":"x = 1"}}]}')

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [task, dict(task, id=2)]), \
             mock.patch.object(client_module.urllib.request, "urlopen", side_effect=urlopen), \
             mock.patch.object(client_module, "UNREACHABLE_RETRY_DELAYS", ()), \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            completed = judge_eval.run_hard_complex_benchmark(
                "http://server/v1", "model", "", out_file=out,
                judge_endpoint="http://judge/v1", judge_model="judge")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertFalse(completed)
        self.assertEqual(len(requests), 2)
        self.assertIn("Judge unreachable", report["incomplete_reason"])
        self.assertEqual(report["not_run"], [2])
        self.assertIsNotNone(report["leaderboard"][0]["tok_s"])


class TestJudgeScoreParsing(unittest.TestCase):
    def test_out_of_range_nonfinite_and_boolean_scores_are_unscored(self):
        examples = (
            ('{"score_10": 80}', "score"),
            ('{"score_10": NaN}', "score"),
            ('{"score_10": true}', "score"),
            ('{"model_a_scores": {"correctness": 11}}', "model_a"),
            ('{"model_a_scores": {"correctness": -1}}', "model_a"),
            ('{"model_a_scores": {"correctness": NaN}}', "model_a"),
            ("score_10: 8.5e99", "score"),
            ("score_10: 9.0.0", "score"),
        )
        for answer, key in examples:
            with self.subTest(answer=answer):
                self.assertIsNone(judge_eval.parse_judge_score_10(answer, key)[0])

    def test_valid_zero_legacy_fifty_point_score_and_error_text_in_rationale(self):
        self.assertEqual(judge_eval.parse_judge_score_10('{"score_10": 0}', "score")[0], 0)
        self.assertEqual(judge_eval.parse_judge_score_10('{"model_a_total": 45}', "model_a")[0], 9)
        self.assertEqual(judge_eval.parse_judge_score_10(
            '{"score_10": 7, "rationale": "Reports [ERROR] correctly"}', "score")[0], 7)


class TestIndependentJudge(unittest.TestCase):
    task = {"id": 1, "name": "task", "category": "code", "system": "Python",
            "prompt": "write Python code", "max_tokens": 10, "temperature": 0.0}

    def test_missing_or_self_judge_is_rejected_before_model_requests(self):
        with mock.patch.object(judge_eval, "send_chat_completion_robust") as send:
            with self.assertRaisesRegex(ValueError, "judge"):
                judge_eval.run_hard_complex_benchmark("http://one/v1", "one", "")
            for endpoint, model in (("http://one", "one"), ("http://two/v1", "two")):
                with self.subTest(endpoint=endpoint):
                    with self.assertRaisesRegex(ValueError, "judge"):
                        judge_eval.run_hard_complex_benchmark(
                            "http://one/v1", "one", "", "http://two/v1", "two", "",
                            judge_endpoint=endpoint, judge_model=model)
            send.assert_not_called()

    def test_single_model_is_scored_by_the_separate_judge(self):
        responses = [("x = 1", 1.0, 10, 10.0), ('{"score_10": 8}', 1.0, 10, 10.0)]
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [self.task]), \
             mock.patch.object(judge_eval, "send_chat_completion_robust", side_effect=responses) as send, \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            judge_eval.run_hard_complex_benchmark(
                "http://one/v1", "one", "one-key", out_file=out,
                judge_endpoint="http://judge/v1", judge_model="judge", judge_key="judge-key")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(send.call_args_list[0].args[:3], ("http://one/v1", "one", "one-key"))
        self.assertEqual(send.call_args_list[1].args[:3], ("http://judge/v1", "judge", "judge-key"))
        judge_messages = send.call_args_list[1].args[3]
        self.assertIn(self.task["prompt"], judge_messages[1]["content"])
        self.assertRegex(judge_messages[1]["content"], r"<RESPONSE-[0-9a-f]+>\nx = 1\n</RESPONSE-[0-9a-f]+>")
        self.assertEqual(report["leaderboard"][0]["score_10"], 8.0)
        self.assertEqual(report["leaderboard"][0]["rank"], 1)
        self.assertEqual(report["judge"], {"endpoint": "http://judge/v1", "model": "judge"})

    def test_pair_ranking_uses_the_separate_judge(self):
        responses = [("x = 1", 1.0, 10, 10.0), ("x = 2", 1.0, 10, 10.0)]
        scored = {"m1_score_10": 8.0, "m2_score_10": 6.0, "winner": "Server 1",
                  "rationale": "First answer is better."}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(judge_eval, "HARD_BENCHMARK_TASKS", [self.task]), \
             mock.patch.object(judge_eval, "send_chat_completion_robust", side_effect=responses), \
             mock.patch.object(judge_eval, "run_llm_judge_10_pair", return_value=scored) as pair, \
             mock.patch("builtins.print"):
            out = os.path.join(td, "report.md")
            judge_eval.run_hard_complex_benchmark(
                "http://one/v1", "one", "", "http://two/v1", "two", "", out,
                judge_endpoint="http://judge/v1", judge_model="judge")
            with open(os.path.join(td, "report.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(pair.call_args.args[:2], ("http://judge/v1", "judge"))
        self.assertEqual([(x["model"], x["rank"], x["score_10"]) for x in report["leaderboard"]],
                         [("one", 1, 8.0), ("two", 2, 6.0)])

    def test_cli_uses_flat_server_config_and_configured_judge(self):
        config = {"endpoint": "http://config/v1", "model": "config-model", "api_key": "config-key",
                  "judge": {"endpoint": "http://judge/v1", "model": "judge", "api_key": "judge-key"}}
        env = {"ENDPOINT1": "http://env/v1", "MODEL1": "env-model", "API_KEY1": "env-key"}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f)
            with mock.patch.dict(os.environ, env, clear=True), \
                 mock.patch.object(judge_eval.env_loader, "load_env_file"), \
                 mock.patch.object(sys, "argv", ["judge_eval.py", "--config", path, "--single"]), \
                 mock.patch.object(judge_eval, "run_hard_complex_benchmark") as run:
                judge_eval.main()
        self.assertEqual(run.call_args.args[:3], ("http://config/v1", "config-model", "config-key"))
        self.assertEqual(run.call_args.kwargs["judge_endpoint"], "http://judge/v1")
        self.assertEqual(run.call_args.kwargs["judge_model"], "judge")
        self.assertEqual(run.call_args.kwargs["judge_key"], "judge-key")

    def test_cli_exits_incomplete_when_scoring_cannot_finish(self):
        config = {"endpoint": "http://config/v1", "model": "model",
                  "judge": {"endpoint": "http://judge/v1", "model": "judge"}}
        with mock.patch.object(judge_eval.env_loader, "load_config", return_value=(config, None, False)), \
             mock.patch.object(sys, "argv", ["judge_eval.py", "--single"]), \
             mock.patch.object(judge_eval, "run_hard_complex_benchmark", return_value=False), \
             self.assertRaises(SystemExit) as exited:
            judge_eval.main()
        self.assertEqual(exited.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
