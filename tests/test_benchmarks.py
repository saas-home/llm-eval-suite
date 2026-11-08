#!/usr/bin/env python3
"""Regression tests for the standalone benchmark probes."""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import typing
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import client as client_module  # noqa: E402
from benchmarks import (  # noqa: E402
    adversarial,
    code_synthesis,
    common,
    concurrency,
    context,
    needle_recall,
    precision,
    streaming,
    vision,
)
from benchmarks import (  # noqa: E402
    reasoning as reasoning_benchmark,
)


def response(content, reasoning=""):
    return {
        "content": content, "reasoning": reasoning, "text": reasoning + "\n" + content,
        "prompt_tokens": 10, "completion_tokens": 20, "tok_per_sec": 10.0,
        "ttft_ms": 10.0, "total_time_s": 1.0,
    }


RING_BUFFER = """
import os
assert "BENCHMARK_TEST_API_KEY" not in os.environ
class ConcurrentMonotonicRingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = {}
        self.seq = 0
    def push(self, value):
        self.seq += 1
        self.items[self.seq] = value
        self.items.pop(self.seq - self.capacity, None)
        return self.seq
    def get_by_seq(self, seq):
        return self.items[seq]
    def get_latest(self):
        if not self.items:
            raise IndexError()
        return self.seq, self.items[self.seq]
    def snapshot_in_order(self):
        return sorted(self.items.items())
"""

LRU_CACHE = """
import os
assert "BENCHMARK_TEST_API_KEY" not in os.environ
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = {}
        self.order = []
    def get(self, key):
        if key not in self.items:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.items[key]
    def peek(self, key):
        return self.items.get(key, -1)
    def put(self, key, value):
        if key in self.items:
            self.order.remove(key)
        elif len(self.order) == self.capacity:
            del self.items[self.order.pop(0)]
        self.items[key] = value
        self.order.append(key)
"""


class TestStandaloneClients(unittest.TestCase):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def test_adversarial_throughput_ignores_delayed_usage_trailer(self):
        chunks = (
            {"choices": [{"delta": {"content": "first"}}]},
            {"choices": [{"delta": {"content": "last"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 5}},
        )
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        times = iter((0.0, 1.0, 3.0, 13.0))
        with mock.patch.object(client_module.urllib.request, "urlopen",
                               return_value=self.Response(body.encode())), \
             mock.patch.object(client_module.time, "perf_counter", side_effect=lambda: next(times)):
            result = adversarial.BenchmarkClient("http://server/v1", "model").call([], stream=True)
        self.assertEqual(result["tok_per_sec"], 2.0)
        self.assertEqual(result["ttft_ms"], 1000.0)

    def test_request_failures_are_reported_separately_from_model_failures(self):
        error = urllib.error.URLError(ConnectionRefusedError(111, "connection refused"))
        with mock.patch.object(client_module.urllib.request, "urlopen", side_effect=error), \
             mock.patch.object(client_module.time, "sleep"), \
             mock.patch("builtins.print"):
            first = adversarial.test_multihop_graph_distractors(
                adversarial.BenchmarkClient("http://server/v1", "model"))
            second = code_synthesis.test_executable_code("http://server/v1")
        for result in (first, second):
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "ERROR")
            self.assertEqual(result["failure_kind"], "infra")
            self.assertTrue(result["endpoint_unreachable"])

    def test_malformed_stream_is_an_error_not_an_answer(self):
        body = (
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            b'data: {malformed}\n\n'
            b'data: [DONE]\n\n'
        )
        for benchmark in (
            lambda: adversarial.test_multihop_graph_distractors(
                adversarial.BenchmarkClient("http://server/v1", "model")),
            lambda: code_synthesis.test_executable_code("http://server/v1"),
        ):
            with self.subTest(benchmark=benchmark), \
                 mock.patch.object(client_module.urllib.request, "urlopen",
                                   return_value=self.Response(body)), \
                 mock.patch("builtins.print"):
                result = benchmark()
            self.assertEqual(result["status"], "ERROR")
            self.assertEqual(result["failure_kind"], "infra")

    def test_unreachable_server_stops_both_suites_and_writes_incomplete_report(self):
        failure = {
            "name": "first", "passed": False, "status": "ERROR", "error": "unreachable",
            "failure_kind": "infra", "endpoint_unreachable": True, "tok_s": 0,
            "tok_per_sec": 0, "tokens": 0,
        }
        cases = (
            (adversarial, "test_multihop_graph_distractors", "test_novel_algorithmic_fuzz"),
            (code_synthesis, "test_executable_code", "test_multi_needle_in_haystack"),
        )
        for module, first_name, second_name in cases:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                report_path = str(Path(directory) / "report.json")
                with mock.patch.object(module, first_name, return_value=failure) as first, \
                     mock.patch.object(module, second_name) as second, \
                     mock.patch.object(sys, "argv", ["benchmark", "--out", report_path]), \
                     mock.patch("builtins.print"), self.assertRaises(SystemExit) as exited:
                    module.main()
                first.assert_called_once()
                second.assert_not_called()
                self.assertEqual(exited.exception.code, 3)
                report = json.loads(Path(report_path).read_text(encoding="utf-8"))
                self.assertTrue(report["incomplete"])
                self.assertEqual(len(report["not_run"]), 4)
                self.assertEqual(report["results"], [failure])

    def test_other_stream_probes_reject_malformed_or_error_events(self):
        body = (
            b'data: {"choices":[{"delta":{"content":"architecture"}}],'
            b'"usage":{"completion_tokens":50}}\n\n'
            b'data: {malformed}\n\n'
            b'data: [DONE]\n\n'
        )
        task = {"stream_id": 1, "name": "task", "prompt": "hello", "expected_topics": ["architecture"]}
        with mock.patch.object(client_module.urllib.request, "urlopen",
                               side_effect=lambda *args, **kwargs: self.Response(body)), \
             mock.patch("builtins.print"):
            self.assertIsNone(streaming.run_test("http://server/v1", "hi"))
            self.assertIn("error", context.run_benchmark_run("http://server/v1", "hi"))
            results = {}
            concurrency.client_task(1, "hi", "http://server/v1", "model", 64, 0,
                                    results, threading.Lock(), None)
            self.assertTrue(results[1]["error"])
            self.assertEqual(results[1]["failure_kind"], "infra")
            reasoning_result = reasoning_benchmark.execute_stream("http://server/v1", "model", task, 50, 0)
            self.assertEqual(reasoning_result["status"], "ERROR")
            self.assertEqual(reasoning_result["failure_kind"], "infra")

    def test_vision_stream_error_does_not_save_a_success_report(self):
        body = (
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            b'data: {"error":{"message":"generation failed"}}\n\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vision.json"
            args = ["vision", "--url", "http://server/v1", "--out", str(output)]
            with mock.patch.object(sys, "argv", args), \
                 mock.patch.object(client_module.urllib.request, "urlopen",
                                   return_value=self.Response(body)), \
                 mock.patch("builtins.print"), self.assertRaises(SystemExit) as exited:
                vision.main()
            self.assertNotEqual(exited.exception.code, 0)
            self.assertFalse(output.exists())

    def test_context_cli_accepts_output_and_records_estimated_tokens(self):
        measurement = {
            "prompt_tokens": 100, "prompt_tokens_estimated": True, "completion_tokens": 10,
            "cached_tokens": 0, "ttft_s": 1.0, "cold_prefill_tok_s": 100.0,
            "effective_prefill_tok_s": 100.0, "decode_time_s": 1.0, "decode_tok_s": 9.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "context.json"
            args = ["context", "--tokens", "100", "--out", str(output)]
            with mock.patch.object(sys, "argv", args), \
                 mock.patch.object(context, "run_benchmark_run", return_value=dict(measurement)), \
                 mock.patch.object(context.time, "sleep"), mock.patch("builtins.print"):
                context.main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["results"][0]["prompt_tokens_estimated"])
            self.assertFalse(report["incomplete"])

    def test_fully_cached_context_has_zero_cold_prefill(self):
        result = {
            "prompt_tokens": 100, "cached_tokens": 100, "completion_tokens": 10,
            "ttft": 2.0, "decode_speed": 5.0,
        }
        with mock.patch.object(context, "LLMClient") as factory:
            factory.return_value.call.return_value = result
            measurement = context.run_benchmark_run("http://server/v1", "hi")
        self.assertEqual(measurement["cold_prefill_tok_s"], 0)
        self.assertEqual(measurement["effective_prefill_tok_s"], 50)

    def test_reasoning_topic_match_is_partial_not_verified_quality(self):
        body = (
            b'data: {"choices":[{"delta":{"content":"architecture"}}],'
            b'"usage":{"completion_tokens":50}}\n\n'
            b'data: [DONE]\n\n'
        )
        task = {"stream_id": 1, "name": "task", "prompt": "hello", "expected_topics": ["architecture"]}
        with mock.patch.object(client_module.urllib.request, "urlopen", return_value=self.Response(body)):
            result = reasoning_benchmark.execute_stream("http://server/v1", "model", task, 50, 0)
        self.assertEqual(result["status"], "PARTIAL")

    def test_parallel_decode_rate_uses_the_generated_chunk_span(self):
        results = [
            {"_decode_start": 1.0, "_decode_end": 3.0, "completion_tokens": 5},
            {"_decode_start": 2.0, "_decode_end": 4.0, "completion_tokens": 7},
            {"_decode_start": 0.0, "_decode_end": 100.0, "completion_tokens": 500, "error": "timeout"},
        ]
        self.assertAlmostEqual(reasoning_benchmark.aggregate_decode_rate(results), 10 / 3)

    def test_reasoning_task_annotation_supports_python38(self):
        annotation = typing.get_type_hints(reasoning_benchmark.build_parallel_tasks)["return"]
        self.assertEqual(annotation, typing.List[dict])

    def test_concurrency_matrix_fails_when_no_completion_was_generated(self):
        def no_output(client_id, prompt, api_url, model, max_tokens, temperature, results, lock, barrier, api_key):
            results[client_id] = {"prompt_tokens": 10, "completion_tokens": 0 if client_id == 1 else 10, "ttft_s": 0,
                                  "decode_tok_s": 0, "error": None}

        with mock.patch.object(concurrency, "client_task", side_effect=no_output), \
             mock.patch("builtins.print"):
            for clients in (1, 2):
                with self.subTest(clients=clients):
                    self.assertFalse(concurrency.run_matrix_benchmark([clients], [10], "http://server/v1", "model", 64, 0))

    def test_concurrency_reports_observed_aggregate_and_mean_client_rates(self):
        def measured(client_id, prompt, api_url, model, max_tokens, temperature, results, lock, barrier, api_key):
            results[client_id] = {"prompt_tokens": 100, "completion_tokens": 11, "ttft_s": float(client_id),
                                  "prefill_tok_s": 100.0 / client_id, "decode_tok_s": 10.0,
                                  "t_first": 2.0 * client_id - 1, "t_last": 2.0 * client_id,
                                  "error": None}

        with mock.patch.object(concurrency, "client_task", side_effect=measured), \
             mock.patch("builtins.print") as output:
            self.assertTrue(concurrency.run_matrix_benchmark([2], [100], "http://server/v1", "model", 64, 0))
        row = next(call.args[0] for call in output.call_args_list
                   if isinstance(call.args[0], str) and call.args[0].strip().startswith("100 |"))
        columns = [column.strip() for column in row.split("|")]
        self.assertEqual(float(columns[4]), 75.0)
        self.assertEqual(float(columns[5]), 100.0)
        self.assertEqual(float(columns[6]), 10.0)
        self.assertEqual(float(columns[7]), 6.7)

    def test_concurrency_stops_requests_and_health_checks_when_server_is_down(self):
        def refused(client_id, prompt, api_url, model, max_tokens, temperature, results, lock, barrier, api_key):
            results[client_id] = {"prompt_tokens": 0, "completion_tokens": 0, "ttft_s": 0,
                                  "decode_tok_s": 0, "error": "refused", "endpoint_unreachable": True}

        args = ["concurrency", "--clients", "1", "--contexts", "10", "20"]
        with mock.patch.object(sys, "argv", args), \
             mock.patch.object(concurrency, "client_task", side_effect=refused) as task, \
             mock.patch.object(concurrency, "check_health", return_value={"status": "ok"}) as health, \
             mock.patch("builtins.print"), self.assertRaises(SystemExit) as exited:
            concurrency.main()
        self.assertEqual(exited.exception.code, 3)
        task.assert_called_once()
        health.assert_called_once()

    def test_context_stops_after_unreachable_warmup(self):
        error = {"error": "unreachable", "endpoint_unreachable": True}
        with mock.patch.object(sys, "argv", ["context", "--tokens", "10", "20"]), \
             mock.patch.object(context, "run_benchmark_run", return_value=error) as run, \
             mock.patch("builtins.print"), self.assertRaises(SystemExit) as exited:
            context.main()
        self.assertEqual(exited.exception.code, 3)
        run.assert_called_once()


class TestLongContextGrading(unittest.TestCase):
    def test_chat_template_fallback_is_limited_to_a_template_rejection(self):
        success = {"content": "answer", "text": "answer", "total_time": 1.0}
        cases = (
            ({"error": "chat_template_kwargs is unsupported", "status_code": 400}, 2),
            ({"error": "context length exceeded", "status_code": 400}, 1),
            ({"error": "chat_template_kwargs server error", "status_code": 500}, 1),
        )
        for error, expected_calls in cases:
            with self.subTest(error=error), mock.patch.object(common, "LLMClient") as factory, \
                 mock.patch("builtins.print"):
                factory.return_value.call.side_effect = [error, success]
                result = precision.call_model("http://server/v1", [])
            self.assertEqual(factory.return_value.call.call_count, expected_calls)
            self.assertEqual("error" in result, expected_calls == 1)

    def test_numeric_graders_check_final_declared_values_not_reasoning(self):
        cases = (
            (precision, precision.test_14_stage_dependency, "FINAL_STAGE_14", "101276"),
            (precision, precision.test_financial_ledger_reconciliation, "ENDING_BALANCE", "$11565.22"),
            (needle_recall, needle_recall.test_multi_hop_variable_tracking, "FINAL_VALUE", "72898"),
        )
        for module, check, label, expected in cases:
            with self.subTest(label=label), mock.patch("builtins.print"):
                with mock.patch.object(module, "call_model",
                                       return_value=response(f"{label}: 0", reasoning=expected)):
                    self.assertFalse(check("http://server/v1")["passed"])
                with mock.patch.object(module, "call_model",
                                       return_value=response(f"{label}: {expected}")):
                    self.assertTrue(check("http://server/v1")["passed"])

    def test_key_value_recall_checks_the_actual_json_mapping(self):
        expected = {
            "KEY_B83C": "VAL_4m$Zk8*v", "KEY_E50F": "VAL_8t&Kx9$d",
            "KEY_H27I": "VAL_1k$Mf2&c", "KEY_K94L": "VAL_7z&Wb6$f",
            "KEY_N61O": "VAL_8b$Dj5&p", "KEY_Q38R": "VAL_9f&Nq8$y",
            "KEY_T05U": "VAL_5m$Pd9&k", "KEY_W72X": "VAL_4t&Kx1$d",
        }
        swapped = dict(expected)
        swapped["KEY_B83C"], swapped["KEY_E50F"] = swapped["KEY_E50F"], swapped["KEY_B83C"]
        with mock.patch("builtins.print"):
            for answer, passed in ((expected, True), (swapped, False)):
                with self.subTest(answer=answer), \
                     mock.patch.object(needle_recall, "call_model",
                                       return_value=response(json.dumps(answer))):
                    self.assertEqual(needle_recall.test_high_entropy_kv_recall("http://server/v1")["passed"],
                                     passed)

    def test_long_context_request_errors_are_not_graded_as_model_failures(self):
        error = {"error": "connection refused", "failure_kind": "infra",
                 "endpoint_unreachable": True, **response("")}
        cases = (
            (precision, precision.test_14_stage_dependency),
            (precision, precision.test_financial_ledger_reconciliation),
            (needle_recall, needle_recall.test_high_entropy_kv_recall),
            (needle_recall, needle_recall.test_multi_hop_variable_tracking),
        )
        for module, check in cases:
            with self.subTest(check=check.__name__), \
                 mock.patch.object(module, "call_model", return_value=error), \
                 mock.patch("builtins.print"):
                result = check("http://server/v1")
            self.assertEqual(result["status"], "ERROR")
            self.assertEqual(result["failure_kind"], "infra")

    def test_long_context_suites_stop_after_server_unreachable(self):
        failure = {"name": "first", "passed": False, "status": "ERROR",
                   "error": "unreachable", "endpoint_unreachable": True}
        cases = (
            (precision, "test_14_stage_dependency", "test_financial_ledger_reconciliation"),
            (needle_recall, "test_high_entropy_kv_recall", "test_multi_hop_variable_tracking"),
        )
        for module, first_name, second_name in cases:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                report_path = str(Path(directory) / "report.json")
                with mock.patch.object(module, first_name, return_value=failure), \
                     mock.patch.object(module, second_name) as second, \
                     mock.patch.object(sys, "argv", ["benchmark", "--out", report_path]), \
                     mock.patch("builtins.print"), self.assertRaises(SystemExit) as exited:
                    module.main()
                second.assert_not_called()
                self.assertEqual(exited.exception.code, 3)
                report = json.loads(Path(report_path).read_text(encoding="utf-8"))
                self.assertEqual(report[0], failure)
                self.assertEqual(report[1]["status"], "SKIPPED")


class TestAdversarialGrading(unittest.TestCase):
    def test_multihop_checks_final_declared_code_and_ordered_route(self):
        route = ("Ingress Node Alpha -> Transit Hub Tokyo -> Secure Bridge Omega\n"
                 "Chicago and Frankfurt routes are revoked.\n")
        correct = route + "ACTIVE_FAILOVER_CODE: AUTH_ROUTING_KEY_9921_TOK"
        answers = (
            (correct, "", True),
            (correct + "\nACTIVE_FAILOVER_CODE: WRONG", "", False),
            ("Tokyo and Bridge Omega.\nChicago and Frankfurt revoked.\n"
             "ACTIVE_FAILOVER_CODE: AUTH_ROUTING_KEY_9921_TOK", "", False),
            ("ACTIVE_FAILOVER_CODE: WRONG", correct, False),
            ("", correct, False),
        )
        for content, reasoning, expected in answers:
            with self.subTest(content=content, reasoning=reasoning), \
                 mock.patch.object(adversarial, "log"):
                client = mock.Mock()
                client.call.return_value = response(content, reasoning)
                self.assertEqual(adversarial.test_multihop_graph_distractors(client)["passed"], expected)

    def test_algebra_uses_the_declared_final_solutions(self):
        answers = (
            ("The inverse is 8.\nFINAL_SOLUTIONS_FOR_X: [8, 17]", "", True),
            ("FINAL_SOLUTIONS_FOR_X: [17, 8]", "", True),
            ("FINAL_SOLUTIONS_FOR_X: [18, 170]", "", False),
            ("FINAL_SOLUTIONS_FOR_X: [8, 17, 18]", "", False),
            ("FINAL_SOLUTIONS_FOR_X: [8, 17]\nFINAL_SOLUTIONS_FOR_X: [1, 2]", "", False),
            ("FINAL_SOLUTIONS_FOR_X: [1, 2]", "Earlier I thought 8 and 17.", False),
            ("", "FINAL_SOLUTIONS_FOR_X: [8, 17]", False),
        )
        for content, reasoning, expected in answers:
            with self.subTest(content=content, reasoning=reasoning), \
                 mock.patch.object(adversarial, "log"):
                client = mock.Mock()
                client.call.return_value = response(content, reasoning)
                self.assertEqual(adversarial.test_counterfactual_axiomatic_math(client)["passed"], expected)

    def test_frontier_checks_register_assignments_and_last_checksum(self):
        correct = "ALPHA_REG = 4821\nBETA_REG = 7392\nGAMMA_REG = 3105\nCOMPOSITE_CHECKSUM = 534"
        answers = (
            (correct, "", True),
            (correct + "\nCOMPOSITE_CHECKSUM = 535", "", False),
            ("ALPHA_REG = 14821\nBETA_REG = 7392\nGAMMA_REG = 3105\nCOMPOSITE_CHECKSUM = 534", "", False),
            ("ALPHA_REG = 4821\nBETA_REG = 7392\nGAMMA_REG = 3105\nCOMPOSITE_CHECKSUM = 1534", "", False),
            ("COMPOSITE_CHECKSUM = 535", correct, False),
            ("", correct, False),
        )
        for content, reasoning, expected in answers:
            with self.subTest(content=content, reasoning=reasoning), \
                 mock.patch.object(adversarial, "log"), \
                 mock.patch.object(adversarial.random, "randint", return_value=4821):
                client = mock.Mock()
                client.call.return_value = response(content, reasoning)
                self.assertEqual(adversarial.test_frontier_depth_precision(client, 100)["passed"], expected)

    @staticmethod
    def briefing(words=100, yaml_block=None, ending="[SECURITY_STAMP_2026_VERIFIED]"):
        yaml_block = yaml_block or (
            "enclave_id: vault\npage_isolation: true\nring_buffer_mb: 256\nattestation_pki: enabled")
        return (
            "### Section I: Foundation\nRulers support this architecture.\n"
            "### Section II: Isolation Proof\nA thin wall blocks all attacks in a virtual vault.\n"
            "### Section III: Hardware Enclaves\n" + "word " * words + "\n"
            "### Section IV: Infrastructure Configuration\n```yaml\n" + yaml_block + "\n```\n" + ending
        )

    def grade_briefing(self, text, reasoning=""):
        client = mock.Mock()
        client.call.return_value = response(text, reasoning)
        with mock.patch.object(adversarial, "log"):
            return adversarial.test_ifeval_combinatorial_constraints(client)

    def test_combinatorial_constraints_enforce_exact_sections_and_bounds(self):
        self.assertTrue(self.grade_briefing(self.briefing())["passed"])
        for count in (90, 130):
            with self.subTest(count=count):
                result = self.grade_briefing(self.briefing(count))
                self.assertFalse(result["passed"])
                self.assertFalse(result["constraint_details"]["word_count_90_130"])
        for count in (91, 129):
            self.assertTrue(self.grade_briefing(self.briefing(count))["passed"])
        extra_section = self.briefing().replace(
            "### Section IV: Infrastructure Configuration", "### Extra section\nx\n"
            "### Section IV: Infrastructure Configuration")
        self.assertFalse(self.grade_briefing(extra_section)["constraint_details"]["4_sections"])

    def test_combinatorial_constraints_validate_yaml_and_final_stamp(self):
        extra_key = ("enclave_id: vault\npage_isolation: true\nring_buffer_mb: 256\n"
                     "attestation_pki: enabled\nunexpected: value")
        self.assertFalse(self.grade_briefing(self.briefing(yaml_block=extra_key))["passed"])
        invalid_yaml = ("enclave_id: [unterminated\npage_isolation: true\n"
                        "ring_buffer_mb: 256\nattestation_pki: enabled")
        unverified = self.grade_briefing(self.briefing(yaml_block=invalid_yaml))
        self.assertFalse(unverified["passed"])
        self.assertEqual(unverified["status"], "PARTIAL")
        self.assertFalse(self.grade_briefing(self.briefing(
            ending="[SECURITY_STAMP_2026_VERIFIED]\nMore text."))["passed"])
        self.assertFalse(self.grade_briefing("", self.briefing())["passed"])


class TestNeedleJson(unittest.TestCase):
    def test_needle_values_are_checked_as_json_fields_not_substrings(self):
        correct = {
            "port": 9182, "salt": "k7$vB9#zL1", "node": "us-east-cluster-node-42",
            "overflow_bytes": 8388608, "signature": "sig_990184_verified",
            "sum_port_and_overflow": 8397790,
        }
        cases = (
            (json.dumps(correct), "", True),
            (json.dumps(dict(correct, port=19182)), "", False),
            (json.dumps(dict(correct, sum_port_and_overflow=8397791, note="8397790")), "", False),
            ("Not JSON: " + json.dumps(correct), "", False),
            ("", json.dumps(correct), False),
        )
        for content, reasoning, expected in cases:
            with self.subTest(content=content, reasoning=reasoning), \
                 mock.patch.object(code_synthesis, "call_model", return_value=response(content, reasoning)), \
                 mock.patch("builtins.print"):
                self.assertEqual(code_synthesis.test_multi_needle_in_haystack("http://server/v1")["passed"],
                                 expected)

    def test_strict_schema_requires_final_json_and_exact_value_types(self):
        nodes = [
            {"node_id": 1000 + i, "region": region, "is_active": True, "ip_address": f"192.0.2.{i+1}",
             "tags": ["a", "b", "c"], "load_avg": 0.5}
            for i, region in enumerate(("us-east", "us-west", "eu-central"))
        ]
        valid = json.dumps(nodes)
        bad_load = [dict(node) for node in nodes]
        bad_load[0]["load_avg"] = True
        bad_ip = [dict(node) for node in nodes]
        bad_ip[0]["ip_address"] = "192.000.2.1"
        cases = ((valid, "", True), ("Here is the JSON:\n" + valid, "", False),
                 ("```json\n" + valid + "\n```", "", False),
                 (json.dumps(bad_load), "", False), (json.dumps(bad_ip), "", False),
                 ("", valid, False))
        for content, reasoning, expected in cases:
            with self.subTest(content=content, reasoning=reasoning), \
                 mock.patch.object(code_synthesis, "call_model", return_value=response(content, reasoning)), \
                 mock.patch("builtins.print"):
                self.assertEqual(code_synthesis.test_strict_schema_validation("http://server/v1")["passed"],
                                 expected)

    def test_schema_validation_is_not_disabled_by_python_optimization(self):
        nodes = [
            {"node_id": i, "region": region, "is_active": True, "ip_address": f"192.0.2.{i+1}",
             "tags": ["a", "b", "c"], "load_avg": 0.5}
            for i, region in enumerate(("us-east", "us-west", "eu-central"))
        ]
        answer = json.dumps(nodes)
        script = (
            "from benchmarks import code_synthesis as b\n"
            f"b.call_model = lambda *a, **k: {{'content': {answer!r}, 'text': {answer!r}, "
            "'completion_tokens': 1, 'total_time_s': 1, 'tok_per_sec': 1, 'ttft_ms': 1}\n"
            "print('RESULT:', b.test_strict_schema_validation('http://server/v1')['passed'])\n"
        )
        result = subprocess.run([sys.executable, "-O", "-c", script], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT: False", result.stdout)

    def test_math_grading_uses_the_last_final_fraction(self):
        good = "C(12, 3) = 220; blue outcomes 164; red and blue 130.\nFinal: 65/82"
        examples = (
            (good, "", True),
            ("220 total, 164 blue, 130 both. Final: \\frac{65}{82}", "", True),
            (good + "\nCorrection: 1/2", "", False),
            ("220 total, 164 blue, 130 both. Final: 65/820", "", False),
            ("220 total, 164 blue, 130 both. Final: 1/2", good, False),
            ("", good, False),
        )
        for content, reasoning, expected in examples:
            with self.subTest(content=content, reasoning=reasoning), \
                 mock.patch.object(code_synthesis, "call_model", return_value=response(content, reasoning)), \
                 mock.patch("builtins.print"):
                self.assertEqual(code_synthesis.test_mathematical_reasoning("http://server/v1")["passed"],
                                 expected)


class TestStandaloneCodeExecution(unittest.TestCase):
    def run_ring(self, code, safe=""):
        client = mock.Mock()
        client.call.return_value = response("```python\n" + code + "\n```")
        with mock.patch.dict(os.environ, {"EVAL_SAFE": safe, "BENCHMARK_TEST_API_KEY": "secret"}), \
             mock.patch.object(adversarial, "log"):
            return adversarial.test_novel_algorithmic_fuzz(client)

    def run_lru(self, code, safe=""):
        with mock.patch.dict(os.environ, {"EVAL_SAFE": safe, "BENCHMARK_TEST_API_KEY": "secret"}), \
             mock.patch.object(code_synthesis, "call_model", return_value=response("```python\n" + code + "\n```")), \
             mock.patch("builtins.print"):
            return code_synthesis.test_executable_code("http://server/v1")

    def test_valid_code_passes_without_inheriting_credentials(self):
        self.assertTrue(self.run_ring(RING_BUFFER)["passed"])
        self.assertTrue(self.run_lru(LRU_CACHE)["passed"])

    def test_bare_success_markers_cannot_forge_a_pass(self):
        ring = ("import sys\nprint('ALL_5000_FUZZ_TESTS_PASSED')\nsys.exit(0)\n"
                "class ConcurrentMonotonicRingBuffer: pass")
        lru = "import sys\nprint('ALL_UNIT_TESTS_PASSED')\nsys.exit(0)"
        self.assertFalse(self.run_ring(ring)["passed"])
        self.assertFalse(self.run_lru(lru)["passed"])

    def test_execution_disabled_is_partial_not_pass(self):
        for result in (self.run_ring(RING_BUFFER, safe="1"), self.run_lru(LRU_CACHE, safe="1")):
            with self.subTest(result=result["name"]):
                self.assertFalse(result["passed"])
                self.assertEqual(result["status"], "PARTIAL")

    def test_false_safe_setting_does_not_disable_execution(self):
        self.assertTrue(self.run_ring(RING_BUFFER, safe="0")["passed"])
        self.assertTrue(self.run_lru(LRU_CACHE, safe="0")["passed"])
        self.assertFalse(self.run_ring("class ConcurrentMonotonicRingBuffer: pass", safe="0")["passed"])
        self.assertFalse(self.run_lru("class LRUCache: pass", safe="0")["passed"])


if __name__ == "__main__":
    unittest.main()
