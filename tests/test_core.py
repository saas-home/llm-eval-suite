#!/usr/bin/env python3
"""
Unit tests for the llm-eval-suite harness core (stdlib unittest only).

Run from the repository root:
    python3 -m unittest discover -s tests -v
"""

import os
import sys
import json
import tempfile
import unittest

# Make repository root importable regardless of CWD.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import client
import env_loader
import reporting
import compare


class TestExtractPythonCode(unittest.TestCase):
    def test_extracts_fenced_block(self):
        raw = "Here you go:\n```python\nclass Foo:\n    pass\n```\nDone."
        self.assertEqual(client.extract_python_code(raw), "class Foo:\n    pass")

    def test_prefers_class_block(self):
        raw = (
            "```python\nhelper = 1\n```\n"
            "```python\nclass LRUCache:\n    pass\n```\n"
        )
        code = client.extract_python_code(raw, prefer_class="class LRUCache")
        self.assertIn("class LRUCache", code)
        self.assertNotIn("helper", code)

    def test_no_fence_fallback(self):
        raw = "def add(a, b):\n    return a + b"
        self.assertEqual(client.extract_python_code(raw), raw)

    def test_unclosed_fence(self):
        raw = "```python\nx = 1\n"
        self.assertEqual(client.extract_python_code(raw), "x = 1")

    def test_extracts_py_and_python3_fence(self):
        raw_py = "```py\nx = 1\n```"
        raw_py3 = "```python3\ny = 2\n```"
        self.assertEqual(client.extract_python_code(raw_py), "x = 1")
        self.assertEqual(client.extract_python_code(raw_py3), "y = 2")

    def test_prefers_unclosed_class_block_over_unrelated_closed_block(self):
        raw = (
            "```python\n# Snippet\nhelper = 1\n```\n"
            "Now here is the main implementation:\n"
            "```python\nclass TokenBucketRateLimiter:\n    def __init__(self):\n        pass\n"
        )
        code = client.extract_python_code(raw, prefer_class="class TokenBucketRateLimiter")
        self.assertIn("class TokenBucketRateLimiter", code)
        self.assertNotIn("helper = 1", code)


class TestBuildContextMilestones(unittest.TestCase):
    def test_small_context(self):
        m = client.build_context_milestones(16000)
        self.assertIn(4000, m)
        self.assertIn(8000, m)
        self.assertNotIn(16000, m)  # exceeds 80% cap
        self.assertLessEqual(max(m), int(16000 * 0.80))

    def test_large_context_includes_ceiling(self):
        m = client.build_context_milestones(200000)
        self.assertIn(128000, m)
        self.assertLessEqual(max(m), int(200000 * 0.80))
        self.assertEqual(m, sorted(set(m)))

    def test_very_small_context_floor(self):
        m = client.build_context_milestones(1000)
        self.assertTrue(len(m) >= 1)
        self.assertLessEqual(max(m), 1000)


class TestDetectMaxContext(unittest.TestCase):
    def test_vllm_key(self):
        self.assertEqual(client.detect_max_context({"max_model_len": 131072}), 131072)

    def test_openai_top_provider(self):
        m = {"top_provider": {"context_length": 200000}}
        self.assertEqual(client.detect_max_context(m), 200000)

    def test_string_numeric_context(self):
        self.assertEqual(client.detect_max_context({"max_model_len": "204800"}), 204800)

    def test_custom_default_none(self):
        self.assertIsNone(client.detect_max_context({}, default=None))

    def test_fallback(self):
        self.assertEqual(client.detect_max_context({}), 32768)
        self.assertEqual(client.detect_max_context(None), 32768)


class TestLLMClientEndpointNormalization(unittest.TestCase):
    def test_bare_host_gets_v1(self):
        c = client.LLMClient("http://127.0.0.1:8000")
        self.assertEqual(c.completions_url, "http://127.0.0.1:8000/v1/chat/completions")

    def test_v1_untouched(self):
        c = client.LLMClient("http://127.0.0.1:8000/v1")
        self.assertEqual(c.completions_url, "http://127.0.0.1:8000/v1/chat/completions")

    def test_chat_completions_stripped(self):
        c = client.LLMClient("http://host/v1/chat/completions")
        self.assertEqual(c.completions_url, "http://host/v1/chat/completions")

    def test_custom_path_not_mutated(self):
        c = client.LLMClient("http://host/api")
        self.assertEqual(c.completions_url, "http://host/api/chat/completions")

    def test_normalize_endpoint_helper(self):
        self.assertEqual(client.normalize_endpoint("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1")
        self.assertEqual(client.normalize_endpoint("http://127.0.0.1:8000/"), "http://127.0.0.1:8000/v1")
        self.assertEqual(client.normalize_endpoint("http://172.16.16.43:8888/v1/chat/completions"), "http://172.16.16.43:8888/v1")
        self.assertEqual(client.normalize_endpoint("http://host/custom/api"), "http://host/custom/api")


class TestEnvLoader(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "config.json")
            saved = env_loader.save_config(
                endpoint="http://127.0.0.1:8000/v1",
                model="test-model",
                api_key="sk-test",
                max_context=4096,
                config_file=cfg_path,
            )
            self.assertIsNotNone(saved)
            with open(cfg_path) as f:
                data = json.load(f)
            self.assertEqual(data["endpoint"], "http://127.0.0.1:8000/v1")
            self.assertEqual(data["model"], "test-model")
            self.assertEqual(data["api_key"], "sk-test")
            self.assertEqual(data["max_context"], 4096)

    def test_save_two_servers(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "config.json")
            env_loader.save_config(
                endpoint="http://a/v1", model="m1",
                endpoint2="http://b/v1", model2="m2",
                config_file=cfg_path,
            )
            with open(cfg_path) as f:
                data = json.load(f)
            self.assertEqual(data["server1"]["endpoint"], "http://a/v1")
            self.assertEqual(data["server2"]["model"], "m2")

    def test_load_config_reads_saved_file(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "config.json")
            env_loader.save_config("http://x/v1", "model-x", config_file=cfg_path)
            cfg, path, loaded = env_loader.load_config(cfg_path)
            self.assertTrue(loaded)
            self.assertEqual(cfg["endpoint"], "http://x/v1")
            self.assertEqual(cfg["model"], "model-x")


class TestParseSelectedTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(reporting.parse_selected_tests("1,3,18"), {1, 3, 18})

    def test_range(self):
        self.assertEqual(reporting.parse_selected_tests("1-5"), {1, 2, 3, 4, 5})

    def test_alias(self):
        self.assertEqual(reporting.parse_selected_tests("aime"), {18})
        self.assertEqual(reporting.parse_selected_tests("gpqa"), {19})
        self.assertEqual(reporting.parse_selected_tests("gpqa_diamond"), {19})
        self.assertEqual(reporting.parse_selected_tests("context_scaling"), {12})
        self.assertEqual(reporting.parse_selected_tests("surgical_git_diff"), {20})
        self.assertEqual(reporting.parse_selected_tests("git_diff"), {20})
        self.assertEqual(reporting.parse_selected_tests("multifile_context"), {21})
        self.assertEqual(reporting.parse_selected_tests("code_review_security"), {22})
        self.assertEqual(reporting.parse_selected_tests("unit_test_synthesis"), {23})
        self.assertEqual(reporting.parse_selected_tests("strict_typing"), {24})
        self.assertEqual(reporting.parse_selected_tests("refactor_invariants"), {25})
        self.assertEqual(reporting.parse_selected_tests("prompt_injection_defense"), {26})

    def test_suites(self):
        self.assertEqual(reporting.parse_selected_tests(None, "flagship"), set(range(1, 13)))
        self.assertEqual(reporting.parse_selected_tests(None, "adversarial"), set(range(13, 16)))
        self.assertEqual(reporting.parse_selected_tests(None, "frontier"), set(range(16, 20)))
        self.assertEqual(reporting.parse_selected_tests(None, "serving"), {1, 2, 3, 7, 8, 11})
        self.assertEqual(reporting.parse_selected_tests(None, "contract"), {5, 6, 9, 10})
        self.assertEqual(reporting.parse_selected_tests(None, "context"), {12, 13, 15})
        self.assertEqual(reporting.parse_selected_tests(None, "swe"), {4, 14, 17, 20, 21, 22, 23, 24, 25})
        self.assertEqual(reporting.parse_selected_tests(None, "reasoning"), {16, 18, 19, 26})
        self.assertEqual(reporting.parse_selected_tests(None, "all"), set(range(1, 27)))

    def test_unknown_ignored(self):
        self.assertEqual(reporting.parse_selected_tests("1,not_a_test"), {1})

    def test_all_unknown_runs_nothing(self):
        # A typo in --test must NOT silently run the full suite.
        self.assertEqual(reporting.parse_selected_tests("foo,bar"), set())

    def test_exclude_single_and_multiple(self):
        # Exclude from default 'all'
        all_minus_vision = reporting.parse_selected_tests(None, "all", exclude_arg="2")
        self.assertNotIn(2, all_minus_vision)
        self.assertEqual(len(all_minus_vision), 25)

        # Exclude multiple
        res = reporting.parse_selected_tests(None, "all", exclude_arg="2,15")
        self.assertNotIn(2, res)
        self.assertNotIn(15, res)
        self.assertEqual(len(res), 24)

    def test_exclude_range_and_alias(self):
        # Exclude range from flagship suite
        res = reporting.parse_selected_tests(None, "flagship", exclude_arg="1-5")
        self.assertEqual(res, set(range(6, 13)))

        # Exclude by alias
        res_alias = reporting.parse_selected_tests("1-5", exclude_arg="vision")
        self.assertEqual(res_alias, {1, 3, 4, 5})

    def test_exclude_with_test_arg(self):
        # Combined test and exclude
        res = reporting.parse_selected_tests("1-10", exclude_arg="2,3,5")
        self.assertEqual(res, {1, 4, 6, 7, 8, 9, 10})

    def test_exclude_all_results_in_empty(self):
        res = reporting.parse_selected_tests("2", exclude_arg="2")
        self.assertEqual(res, set())


class TestNormalizeReport(unittest.TestCase):
    def test_list_results_converted(self):
        report = {
            "model1": "model-a",
            "endpoint1": "http://a/v1",
            "results": [
                {"task_id": "task1", "status": "PASS", "tok_per_sec": 30.0},
                {"task_id": "task2", "status": "FAIL", "tok_per_sec": 10.0},
            ],
        }
        norm = reporting.normalize_report(report)
        self.assertEqual(norm["model"], "model-a")
        self.assertEqual(norm["endpoint"], "http://a/v1")
        self.assertIn("capabilities_4tasks", norm["results"])
        self.assertEqual(norm["results"]["capabilities_4tasks"]["status"], "PARTIAL")

    def test_bad_input(self):
        norm = reporting.normalize_report("not a dict")
        self.assertEqual(norm["model"], "unknown")
        self.assertEqual(norm["results"], {})


class TestComputeRelativeAdvantages(unittest.TestCase):
    def _summary(self, model, eff, te, spd):
        return {
            "model": model,
            "effectiveness_rate_pct": eff,
            "token_economy_tokens_per_passed_task": te,
            "total_tokens_emitted": 1000,
            "avg_decode_tok_s": spd,
            "avg_ttft_ms": 50.0,
            "total_wall_time_s": 100.0,
            "efficiency_index": 80.0,
        }

    def test_model_a_dominates(self):
        m1 = self._summary("A", 95.0, 100.0, 40.0)
        m2 = self._summary("B", 80.0, 200.0, 30.0)
        adv = reporting.compute_relative_advantages(m1, m2)
        self.assertIn("DOMINATES", adv["verdict"])
        self.assertIn("Model A", adv["adv_eff"])

    def test_equal(self):
        m1 = self._summary("A", 90.0, 100.0, 40.0)
        m2 = self._summary("B", 90.0, 100.0, 40.0)
        adv = reporting.compute_relative_advantages(m1, m2)
        self.assertEqual(adv["adv_eff"], "Equal")


class TestComputeExecutiveSummary(unittest.TestCase):
    def test_summary_math(self):
        report = {
            "model": "m",
            "endpoint": "http://x/v1",
            "total_suite_wall_time_s": 10.0,
            "results": {
                "streaming": {"status": "PASS", "completion_tokens": 100, "tok_s": 50.0, "ttft_ms": 20.0},
                "vision": {"status": "FAIL", "completion_tokens": 100, "tok_s": 50.0, "ttft_ms": 20.0},
            },
        }
        s = reporting.compute_executive_summary(report)
        self.assertEqual(s["total_evaluated"], 2)
        self.assertEqual(s["passed"], 1)
        self.assertEqual(s["failed"], 1)
        self.assertEqual(s["effectiveness_rate_pct"], 50.0)
        self.assertEqual(s["total_tokens_emitted"], 200)
        self.assertGreater(s["efficiency_index"], 0)


class TestExtractTestMetrics(unittest.TestCase):
    def test_dict_status(self):
        m = reporting.extract_test_metrics("aime_olympiad", {"status": "PASS", "completion_tokens": 50, "tok_s": 40.0, "ttft_ms": 10})
        self.assertEqual(m["status"], "PASS")
        self.assertEqual(m["tokens"], 50)
        self.assertEqual(m["score"], 1.0)

    def test_list_milestones(self):
        data = [
            {"status": "PASS", "completion_tokens": 10, "decode_tok_s": 100.0, "ttft_s": 0.5},
            {"status": "PASS", "completion_tokens": 20, "decode_tok_s": 90.0, "ttft_s": 1.0},
        ]
        m = reporting.extract_test_metrics("context_scaling", data)
        self.assertEqual(m["status"], "PASS")
        self.assertEqual(m["tokens"], 30)

    def test_empty(self):
        m = reporting.extract_test_metrics("streaming", None)
        self.assertEqual(m["status"], "SKIPPED")


class TestArenaEvaluatorTaskRouting(unittest.TestCase):
    # Regression guard for the arena evaluator ID mismatch (P0):
    # every content-checked BENCHMARK_TASKS id must hit a specific branch,
    # not the output-length fallback.
    CONTENT_TASKS = [
        "task4_avl_tree", "task5_concurrency_debug", "task6_system_design",
        "task7_long_context_constraints", "task8_agent_tool_calling", "task9_json_schema",
        "task14_precision_math", "task15_code_execution", "task17_multihop_graph",
        "task19_anti_constraints", "task20_counterfactual_algebra", "task21_cruxeval_execution",
        "task22_swe_bench_bug_patch", "task23_aime_olympiad_math",
    ]

    def test_content_tasks_do_not_fall_to_length_fallback(self):
        for tid in self.CONTENT_TASKS:
            res = compare.evaluate_arena_task(tid, "")
            self.assertNotEqual(
                res["status"], "PASS",
                f"{tid} fell to the output-length fallback (empty output should not PASS)",
            )

    def test_all_benchmark_task_ids_are_known_or_latency(self):
        # Every BENCHMARK_TASKS id is either a content-checked branch or a
        # latency/throughput task that legitimately uses the length baseline.
        content = set(self.CONTENT_TASKS)
        latency = {"task1_streaming", "task2_vision", "task3_concurrency",
                   "task10_prefix_caching", "task11_client_abort", "task12_stop_sequences",
                   "task13_high_entropy_recall", "task16_error_handling", "task18_novel_algorithm_fuzz"}
        self.assertEqual(
            {t["id"] for t in compare.BENCHMARK_TASKS},
            content | latency,
        )


class TestInfraClassification(unittest.TestCase):
    """T2: infrastructure failures are distinguished from model failures."""

    def test_5xx_is_infra(self):
        self.assertTrue(reporting._classify_infra({"status": "FAIL", "error": "HTTPError 502: Bad Gateway"}))

    def test_429_is_infra(self):
        self.assertTrue(reporting._classify_infra({"status": "FAIL", "error": "HTTPError 429: Too Many Requests"}))

    def test_timeout_is_infra(self):
        self.assertTrue(reporting._classify_infra({"status": "FAIL", "error": "timed out"}))

    def test_failure_kind_is_infra(self):
        self.assertTrue(reporting._classify_infra({"status": "FAIL", "failure_kind": "infra"}))

    def test_model_failure_is_not_infra(self):
        self.assertFalse(reporting._classify_infra({"status": "FAIL", "constraints_passed": 0}))

    def test_list_all_infra(self):
        data = [{"error": "HTTPError 503: Service Unavailable"}, {"error": "timed out"}]
        self.assertTrue(reporting._classify_infra(data))

    def test_list_mixed_is_not_infra(self):
        data = [{"error": "HTTPError 503"}, {"status": "FAIL", "matched": False}]
        self.assertFalse(reporting._classify_infra(data))

    def test_dict_errors_list_infra(self):
        self.assertTrue(reporting._classify_infra({"status": "FAIL", "errors": ["HTTPError 500: Internal Server Error"]}))

    def test_composite_concurrency_all_infra(self):
        data = {
            "c1": {"status": "FAIL", "errors": ["HTTPError 502: Bad Gateway"]},
            "c2": {"status": "FAIL", "errors": ["Connection refused"]},
            "matrix": []
        }
        self.assertTrue(reporting._classify_infra(data))

    def test_composite_concurrency_mixed_is_not_infra(self):
        data = {
            "c1": {"status": "FAIL", "errors": ["HTTPError 502: Bad Gateway"]},
            "c2": {"status": "FAIL", "errors": [], "matched": False},
            "matrix": []
        }
        self.assertFalse(reporting._classify_infra(data))


class TestConfigFingerprint(unittest.TestCase):
    """T3: comparable configs hash identically; differing configs do not."""

    def _rep(self, **kw):
        base = {"model": "m", "max_context_tokens": 204800, "parallel_streams": 2,
                "context_milestones": [4000, 8000], "suite": "all"}
        base.update(kw)
        return base

    def test_stable(self):
        self.assertEqual(reporting.build_config_fingerprint(self._rep()),
                         reporting.build_config_fingerprint(self._rep()))

    def test_differs_on_context(self):
        self.assertNotEqual(reporting.build_config_fingerprint(self._rep()),
                            reporting.build_config_fingerprint(self._rep(max_context_tokens=131072)))

    def test_differs_on_parallel(self):
        self.assertNotEqual(reporting.build_config_fingerprint(self._rep()),
                            reporting.build_config_fingerprint(self._rep(parallel_streams=4)))


class TestStarRating(unittest.TestCase):
    """T6: star rating bands + completion-rate cap."""

    def test_bands(self):
        self.assertEqual(reporting.star_rating(97, 100), 5)
        self.assertEqual(reporting.star_rating(90, 100), 4)
        self.assertEqual(reporting.star_rating(75, 100), 3)
        self.assertEqual(reporting.star_rating(55, 100), 2)
        self.assertEqual(reporting.star_rating(20, 100), 1)

    def test_completion_cap(self):
        # High effectiveness but flaky environment caps at 3 stars.
        self.assertEqual(reporting.star_rating(97, 50), 3)


class TestCompletionRate(unittest.TestCase):
    """T2: infra failures are excluded from effectiveness but counted in completion."""

    def _report(self, results):
        return {"model": "m", "endpoint": "e", "max_context_tokens": 204800,
                "parallel_streams": 2, "context_milestones": [4000], "suite": "all",
                "total_suite_wall_time_s": 10.0, "results": results}

    def test_infra_excluded_from_effectiveness(self):
        # 1 pass, 1 model-fail, 1 infra-fail. Effectiveness over 2 completed = 50%.
        results = {
            "streaming": {"status": "PASS", "completion_tokens": 100, "tok_s": 30.0, "ttft_ms": 100.0},
            "vision": {"status": "FAIL", "matched": False},
            "concurrency": {"status": "FAIL", "error": "HTTPError 502: Bad Gateway"},
        }
        s = reporting.compute_executive_summary(self._report(results))
        self.assertEqual(s["total_seen"], 3)
        self.assertEqual(s["infra_excluded"], 1)
        self.assertEqual(s["total_evaluated"], 2)
        self.assertEqual(s["effectiveness_rate_pct"], 50.0)
        self.assertEqual(s["completion_rate_pct"], round(2 / 3 * 100, 1))

    def test_all_completed_full_completion(self):
        results = {"streaming": {"status": "PASS", "completion_tokens": 100, "tok_s": 30.0, "ttft_ms": 100.0}}
        s = reporting.compute_executive_summary(self._report(results))
        self.assertEqual(s["completion_rate_pct"], 100.0)
        self.assertEqual(s["infra_excluded"], 0)


class TestBackoff(unittest.TestCase):
    """T1: backoff is jittered, capped, and honors Retry-After for 429."""

    def test_5xx_bounded(self):
        for i in range(20):
            d = client._backoff_delay(i, 500)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, client._RETRY_CAP_SECONDS)

    def test_429_larger_and_capped(self):
        for i in range(20):
            d = client._backoff_delay(i, 429)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, client._RATE_LIMIT_CAP_SECONDS)

    def test_retry_after_honored(self):
        self.assertEqual(client._backoff_delay(0, 429, retry_after=7.0), 7.0)


class TestReasoningExtraction(unittest.TestCase):
    """Reasoning/thought extraction across vLLM (reasoning), Gemini (thought), and OpenAI (reasoning_content)."""

    def test_non_streaming_reasoning_and_thought_keys(self):
        from unittest.mock import patch, MagicMock
        c = client.LLMClient("http://fake:8080/v1", "fake-model")

        # 1. vLLM reasoning key
        mock_resp_vllm = MagicMock()
        mock_resp_vllm.read.return_value = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "Hello", "reasoning": "Thinking..."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }).encode("utf-8")
        mock_resp_vllm.__enter__.return_value = mock_resp_vllm

        with patch("urllib.request.urlopen", return_value=mock_resp_vllm):
            res = c.call([{"role": "user", "content": "hi"}], stream=False)
            self.assertEqual(res["reasoning"], "Thinking...")
            self.assertIn("Thinking...", res["text"])
            self.assertEqual(res["content"], "Hello")

        # 2. Gemini thought key
        mock_resp_gemini = MagicMock()
        mock_resp_gemini.read.return_value = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "Done", "thought": "Plan carefully"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }).encode("utf-8")
        mock_resp_gemini.__enter__.return_value = mock_resp_gemini

        with patch("urllib.request.urlopen", return_value=mock_resp_gemini):
            res = c.call([{"role": "user", "content": "hi"}], stream=False)
            self.assertEqual(res["reasoning"], "Plan carefully")
            self.assertIn("Plan carefully", res["text"])

    def test_streaming_reasoning_delta(self):
        from unittest.mock import patch, MagicMock
        import io
        c = client.LLMClient("http://fake:8080/v1", "fake-model")

        lines = [
            b'data: {"choices":[{"delta":{"reasoning":"Think"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"Answer"}}]}\n',
            b'data: [DONE]\n'
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = iter(lines)
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = c.call([{"role": "user", "content": "hi"}], stream=True)
            self.assertEqual(res["reasoning"], "Think")
            self.assertEqual(res["content"], "Answer")
            self.assertIn("Think", res["text"])
            self.assertIn("Answer", res["text"])
            self.assertIsNotNone(res["ttft"])


class TestStreamedToolCallsAccumulation(unittest.TestCase):
    """Verifies that streamed tool calls arriving in fragments across indices are merged correctly."""

    def test_multi_tool_call_stream_assembly(self):
        from unittest.mock import patch, MagicMock
        c = client.LLMClient("http://fake:8080/v1", "fake-model")

        lines = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"query_db","arguments":"{\\"id\\": "}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"fetch_log","arguments":"{\\"env\\": "}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"101\\"}"}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\\"prod\\"}"}}]}}]}\n',
            b'data: [DONE]\n'
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = iter(lines)
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = c.call([{"role": "user", "content": "hi"}], stream=True)
            self.assertIsNotNone(res["tool_calls"])
            self.assertEqual(len(res["tool_calls"]), 2)
            tc0 = res["tool_calls"][0]
            tc1 = res["tool_calls"][1]
            self.assertEqual(tc0["id"], "call_1")
            self.assertEqual(tc0["function"]["name"], "query_db")
            self.assertEqual(tc0["function"]["arguments"], '{"id": "101"}')
            self.assertEqual(tc1["id"], "call_2")
            self.assertEqual(tc1["function"]["name"], "fetch_log")
            self.assertEqual(tc1["function"]["arguments"], '{"env": "prod"}')


class TestHTTPStatusClassificationAndRetries(unittest.TestCase):
    """Verifies client retry policy: 4xx errors are fatal (no retry), 5xx/429 retry with backoff."""

    def test_client_error_400_not_retried(self):
        from unittest.mock import patch
        import urllib.error
        c = client.LLMClient("http://fake:8080/v1", "fake-model")

        attempts = 0
        def fail_400(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise urllib.error.HTTPError("http://fake:8080/v1/chat/completions", 400, "Bad Request", {}, None)

        with patch("urllib.request.urlopen", side_effect=fail_400):
            res = c.call([{"role": "user", "content": "hi"}], retries=2)
            self.assertEqual(attempts, 1)  # No retries on 400
            self.assertIn("HTTPError 400", res["error"])
            self.assertEqual(res["failure_kind"], "client")

    def test_server_error_500_retried(self):
        from unittest.mock import patch
        import urllib.error
        c = client.LLMClient("http://fake:8080/v1", "fake-model")

        attempts = 0
        def fail_500(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise urllib.error.HTTPError("http://fake:8080/v1/chat/completions", 500, "Internal Server Error", {}, None)

        with patch("urllib.request.urlopen", side_effect=fail_500), patch("time.sleep"):
            res = c.call([{"role": "user", "content": "hi"}], retries=2)
            self.assertEqual(attempts, 3)  # Initial + 2 retries
            self.assertIn("HTTPError 500", res["error"])
            self.assertEqual(res["failure_kind"], "infra")


class TestUsageTokenExtraction(unittest.TestCase):
    """Verifies prompt_tokens, completion_tokens, and cached_tokens extraction."""

    def test_streaming_include_usage(self):
        from unittest.mock import patch, MagicMock
        c = client.LLMClient("http://fake:8080/v1", "fake-model")

        lines = [
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":64}}}\n',
            b'data: [DONE]\n'
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = iter(lines)
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = c.call([{"role": "user", "content": "hi"}], stream=True)
            self.assertEqual(res["prompt_tokens"], 120)
            self.assertEqual(res["completion_tokens"], 1)
            self.assertEqual(res["cached_tokens"], 64)


class TestGPQACatalog(unittest.TestCase):
    """Verifies that GPQA Diamond is properly indexed in the catalog and suites."""

    def test_gpqa_catalog_entry(self):
        catalog_dict = {num: (key, name) for num, key, name in reporting.TEST_CATALOG}
        self.assertIn(19, catalog_dict)
        self.assertEqual(catalog_dict[19][0], "gpqa_diamond")

    def test_gpqa_metrics_extraction(self):
        metrics = reporting.extract_test_metrics("gpqa_diamond", {
            "status": "PASS",
            "score": "3/3",
            "tok_s": 50.0
        })
        self.assertEqual(metrics["status"], "PASS")
        self.assertIn("3/3", metrics["desc"])


class TestClientEnhancements(unittest.TestCase):
    """Tests for health URL generation, model discovery, and API error payload capture."""

    def test_health_url_generation(self):
        c1 = client.LLMClient("http://localhost:8080/v1")
        self.assertEqual(c1.health_url, "http://localhost:8080/health")

        c2 = client.LLMClient("http://localhost:8080/custom/api")
        self.assertEqual(c2.health_url, "http://localhost:8080/custom/api/health")

    def test_discover_model(self):
        from unittest.mock import patch, MagicMock
        c = client.LLMClient("http://localhost:8080/v1")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"data": [{"id": "meta-llama/Llama-3-70B"}]}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            m = c.discover_model()
            self.assertEqual(m, "meta-llama/Llama-3-70B")

    def test_non_streaming_error_payload(self):
        from unittest.mock import patch, MagicMock
        c = client.LLMClient("http://localhost:8080/v1")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "error": {"message": "Invalid prompt format", "type": "invalid_request_error"}
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = c.call([{"role": "user", "content": "hi"}], stream=False)
            self.assertIn("API Error: Invalid prompt format", res["error"])
            self.assertEqual(res["failure_kind"], "client")

    def test_streaming_error_chunk(self):
        from unittest.mock import patch, MagicMock
        c = client.LLMClient("http://localhost:8080/v1")
        lines = [
            b'data: {"error":{"message":"Engine overloaded"}}\n',
            b'data: [DONE]\n'
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = iter(lines)
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = c.call([{"role": "user", "content": "hi"}], stream=True)
            self.assertIn("Stream error: Engine overloaded", res["error"])
            self.assertEqual(res["failure_kind"], "infra")


class TestConcurrencyExecution(unittest.TestCase):
    """Tests for run_test_concurrency execution and metrics."""

    def test_concurrency_metrics_and_error_capture(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.call.return_value = {
            "prompt_tokens": 2000,
            "completion_tokens": 100,
            "decode_speed": 50.0,
            "ttft": 0.05,
            "error": None
        }

        res = eval_module.run_test_concurrency(mock_client, parallel=2, contexts=[2000, 4000])
        self.assertIn("c1", res)
        self.assertIn("c2", res)
        self.assertIn("matrix", res)
        self.assertEqual(res["c1"]["status"], "PASS")
        self.assertEqual(res["c2"]["status"], "PASS")
        self.assertIn("mean_ttft_ms", res["c2"])
        self.assertIn("mean_ttft_s", res["c2"])
        self.assertIn("mean_ttft_s", res["matrix"][0])
        self.assertAlmostEqual(res["matrix"][0]["mean_ttft_s"], 0.05, places=2)
        self.assertIn("speedup_vs_baseline", res["c2"])
        self.assertIn("max_prefill_tok_s", res["c2"])
        self.assertEqual(len(res["matrix"]), 4)  # 2 client levels x 2 contexts
        self.assertEqual(res["matrix"][0]["clients"], 1)
        self.assertEqual(res["matrix"][0]["target_context"], 2000)
        self.assertIn("prefill_tok_s", res["matrix"][0])
        self.assertIn("decode_tok_s", res["matrix"][0])
        self.assertIn("aggregate_decode_tok_s", res["matrix"][0])
        self.assertIn("avg_prefill_tok_s", res["matrix"][0])
        self.assertIn("agg_prefill_tok_s", res["matrix"][0])
        self.assertIn("avg_decode_tok_s", res["matrix"][0])
        self.assertIn("agg_decode_tok_s", res["matrix"][0])
        self.assertIn("rows", res["c1"])
        self.assertEqual(len(res["c1"]["rows"]), 2)
        # c=1: Aggregate exactly equals Average
        self.assertEqual(res["matrix"][0]["agg_decode_tok_s"], res["matrix"][0]["avg_decode_tok_s"])
        self.assertEqual(res["matrix"][0]["agg_prefill_tok_s"], res["matrix"][0]["avg_prefill_tok_s"])
        # c=2: Aggregate is exactly 2x Average
        self.assertAlmostEqual(res["matrix"][2]["agg_decode_tok_s"], 2 * res["matrix"][2]["avg_decode_tok_s"], places=1)
        self.assertAlmostEqual(res["matrix"][2]["agg_prefill_tok_s"], 2 * res["matrix"][2]["avg_prefill_tok_s"], places=1)

    def test_concurrency_full_matrix_default_levels(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.call.return_value = {
            "prompt_tokens": 2000,
            "completion_tokens": 64,
            "decode_speed": 40.0,
            "ttft": 0.04,
            "error": None
        }
        # parallel=4 executes 1, 2, 3, 4 clients across 5 contexts [2k, 4k, 8k, 16k, 32k]
        res = eval_module.run_test_concurrency(mock_client, parallel=4)
        for c in [1, 2, 3, 4]:
            self.assertIn(f"c{c}", res)
            self.assertEqual(res[f"c{c}"]["status"], "PASS")
        self.assertEqual(len(res["matrix"]), 20)  # 4 clients * 5 contexts = 20 data points

    def test_concurrency_error_propagated(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.call.return_value = {
            "completion_tokens": 0,
            "decode_speed": 0.0,
            "ttft": 0.0,
            "error": "HTTPError 500: Internal Server Error"
        }

        res = eval_module.run_test_concurrency(mock_client, parallel=2, contexts=[2000])
        self.assertEqual(res["c2"]["status"], "DEGRADED/FAIL")
        self.assertTrue(len(res["c2"]["errors"]) > 0)
        self.assertIn("HTTPError 500", res["c2"]["errors"][0])


class TestSummaryTableFormatting(unittest.TestCase):
    """Tests for print_summary_table selective rendering and formatting."""

    def test_summary_table_selective_rendering(self):
        from io import StringIO
        import sys
        from reporting import print_summary_table

        report = {
            "endpoint": "http://127.0.0.1:8080/v1",
            "model": "test-model",
            "max_context_tokens": 32768,
            "parallel_streams": 1,
            "total_suite_wall_time_s": 10.0,
            "selected_tests": [18, 19],
            "results": {
                "aime_olympiad": {
                    "status": "PASS",
                    "tok_s": 45.2,
                    "ttft_ms": 120.0,
                    "tokens": 500,
                },
                "gpqa_diamond": {
                    "status": "FAIL",
                    "score": "1/3",
                    "passed_count": 1,
                    "tok_s": 38.0,
                    "tokens": 400,
                }
            }
        }

        captured = StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            print_summary_table(report)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        # Selected tests are rendered
        self.assertIn("18. AIME Olympiad", output)
        self.assertIn("19. GPQA Diamond", output)
        self.assertIn("Score: 1/3", output)
        # Unselected early tests are NOT rendered as SKIPPED
        self.assertNotIn("1. Streaming & Latency", output)
        self.assertNotIn("2. Multimodal Vision", output)
        self.assertNotIn("5. Tool Calling", output)


class TestAssemblePythonScript(unittest.TestCase):
    """Tests for assemble_python_script to prevent SyntaxError on __future__ imports."""

    def test_future_import_hoisted_to_top(self):
        import ast
        from client import assemble_python_script

        preamble = "import sys\nimport threading\nfrom collections import deque"
        code_with_future = (
            "from __future__ import annotations\n\n"
            "class RingBuffer:\n"
            "    def __init__(self, cap: int) -> None:\n"
            "        self.cap = cap\n"
        )
        harness = "rb = RingBuffer(5)\nassert rb.cap == 5\nprint('OK')"

        assembled = assemble_python_script(code_with_future, preamble=preamble, harness=harness)
        first_line = assembled.strip().splitlines()[0]
        self.assertEqual(first_line, "from __future__ import annotations")

        # Must compile cleanly via ast.parse without SyntaxError
        tree = ast.parse(assembled)
        self.assertIsNotNone(tree)

    def test_multiple_future_imports_deduplicated(self):
        import ast
        from client import assemble_python_script

        preamble = "from __future__ import annotations\nimport time"
        code = "from __future__ import annotations\nfrom __future__ import division\nx = 1 / 2"
        harness = "assert x == 0.5"

        assembled = assemble_python_script(code, preamble=preamble, harness=harness)
        lines = [line.strip() for line in assembled.splitlines() if line.strip()]
        self.assertEqual(lines[0], "from __future__ import annotations")
        self.assertEqual(lines[1], "from __future__ import division")
        self.assertEqual(assembled.count("from __future__ import annotations"), 1)
        ast.parse(assembled)


class TestEndpointConfigScoping(unittest.TestCase):
    def test_different_endpoint_does_not_inherit_config_model(self):
        from client import normalize_endpoint, detect_max_context
        # Simulate config.json with server1
        cfg = {
            "endpoint": "http://127.0.0.1:8080/v1",
            "model": "Qwen3.8-Flash-Next-FP4",
            "api_key": "secret-key",
            "max_context": 409600
        }
        # User supplies a different endpoint via CLI
        cli_endpoint = "http://172.16.16.43:8888/v1"

        norm_target = normalize_endpoint(cli_endpoint)
        norm_cfg = normalize_endpoint(cfg.get("endpoint"))

        self.assertNotEqual(norm_target, norm_cfg)

        # Scoping rule:
        cfg_model = cfg.get("model") if norm_target == norm_cfg else None
        cfg_max_context = cfg.get("max_context") if norm_target == norm_cfg else None
        cfg_api_key = cfg.get("api_key") if norm_target == norm_cfg else None

        self.assertIsNone(cfg_model)
        self.assertIsNone(cfg_max_context)
        self.assertIsNone(cfg_api_key)

        # Discovered model from endpoint
        discovered_models = [{"id": "qwen3.8-27b", "max_model_len": 204800}]
        # Since target_model_name is None, it selects the single discovered model
        selected_obj = discovered_models[0]
        self.assertEqual(selected_obj["id"], "qwen3.8-27b")

        # Context detection picks up the discovered model's context
        ctx = detect_max_context(selected_obj, default=None)
        self.assertEqual(ctx, 204800)

    def test_matching_endpoint_inherits_config(self):
        from client import normalize_endpoint
        cfg = {
            "endpoint": "http://127.0.0.1:8080/v1",
            "model": "Qwen3.8-Flash-Next-FP4",
            "max_context": 409600
        }
        # User supplies same endpoint without trailing /v1
        cli_endpoint = "http://127.0.0.1:8080"
        norm_target = normalize_endpoint(cli_endpoint)
        norm_cfg = normalize_endpoint(cfg.get("endpoint"))
        self.assertEqual(norm_target, norm_cfg)

    def test_eval_cli_remote_endpoint_selects_discovered_model_without_stale_config(self):
        from client import normalize_endpoint, detect_max_context
        test_cfg = {
            "endpoint": "http://127.0.0.1:8080/v1",
            "model": "Qwen3.8-Flash-Next-FP4",
            "max_context": 409600
        }

        cli_endpoint = "http://172.16.16.43:8888/v1"
        norm_target = normalize_endpoint(cli_endpoint)
        norm_cfg = normalize_endpoint(test_cfg["endpoint"])

        # When target does not match config endpoint, config values are not scoped to target
        self.assertNotEqual(norm_target, norm_cfg)
        cfg_model = test_cfg["model"] if norm_target == norm_cfg else None
        cfg_max_context = test_cfg["max_context"] if norm_target == norm_cfg else None
        self.assertIsNone(cfg_model)
        self.assertIsNone(cfg_max_context)

        # Discovered model from remote endpoint /models
        discovered_models = [{"id": "qwen3.8-27b", "max_model_len": 204800}]
        # Target model name without CLI --model and without matched config is None
        target_model_name = cfg_model  # None

        # When len(models) == 1 and no specific target model, selects the single discovered model
        if not target_model_name and len(discovered_models) == 1:
            selected_model_obj = discovered_models[0]
        else:
            selected_model_obj = None

        self.assertIsNotNone(selected_model_obj)
        self.assertEqual(selected_model_obj["id"], "qwen3.8-27b")

        # Detected context from model object directly reports 204,800
        detected_ctx = detect_max_context(selected_model_obj, default=None)
        self.assertEqual(detected_ctx, 204800)


class TestConcurrencyParallelScaling(unittest.TestCase):
    def test_parallel_6_levels(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.call.return_value = {
            "prompt_tokens": 1000,
            "completion_tokens": 32,
            "decode_speed": 40.0,
            "ttft": 0.05,
            "error": None
        }
        res = eval_module.run_test_concurrency(mock_client, parallel=6, contexts=[1000])
        # Concurrency 6 must evaluate levels 1, 2, 4, and 6
        for c in [1, 2, 4, 6]:
            self.assertIn(f"c{c}", res)
            self.assertEqual(res[f"c{c}"]["status"], "PASS")
            self.assertIn("p95_ttft_ms", res[f"c{c}"]["rows"][0])


class TestStopSequencesStrictness(unittest.TestCase):
    def test_counting_past_stop_is_rejected(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        # Model ignored stop sequence and counted all the way to ten, but server returned finish_reason="stop"
        mock_client.call.side_effect = [
            # First call for stop sequence
            {
                "content": "one, two, three, four, five, six, seven, eight, nine, ten",
                "raw": {"choices": [{"finish_reason": "stop"}]},
                "error": None
            },
            # Determinism call 1
            {"content": "83810205", "error": None},
            # Determinism call 2
            {"content": "83810205", "error": None},
        ]
        res = eval_module.run_test_stop_sequences(mock_client)
        self.assertFalse(res["stop_sequence_respected"])
        self.assertEqual(res["status"], "PARTIAL")  # deterministic was true, but stop was false


class TestSWEAndReasoningSuites(unittest.TestCase):
    def test_surgical_git_diff(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        patch_text = (
            "Here is the fix:\n"
            "```diff\n"
            "--- a/payroll.py\n"
            "+++ b/payroll.py\n"
            "@@ -3,2 +3,4 @@\n"
            "-    for i in range(len(employees) + 1):\n"
            "-        emp = employees[i]\n"
            "+    for emp in employees:\n"
            "+        if not emp:\n"
            "+            continue\n"
            "```\n"
        )
        mock_client.call.return_value = {
            "content": patch_text,
            "completion_tokens": 80,
            "decode_speed": 45.0,
            "ttft": 0.05,
            "error": None
        }
        res = eval_module.run_test_surgical_git_diff(mock_client)
        self.assertTrue(res["valid_diff_syntax"])
        self.assertIn(res["status"], ("PASS", "PARTIAL"))

    def test_multifile_context_ast(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        code = (
            "class AuthenticationService:\n"
            "    def __init__(self, user_repo, config):\n"
            "        self.user_repo = user_repo\n"
            "        self.config = config\n"
            "    def authenticate(self, user_id):\n"
            "        user = self.user_repo.get_by_id(user_id)\n"
            "        if user and user.is_active:\n"
            "            return AuthToken(token=f'tok_{user.id}', expires_in=self.config.max_token_ttl_seconds)\n"
            "        return None\n"
        )
        mock_client.call.return_value = {
            "content": f"```python\n{code}\n```",
            "completion_tokens": 120,
            "decode_speed": 50.0,
            "ttft": 0.05,
            "error": None
        }
        res = eval_module.run_test_multifile_context(mock_client)
        self.assertTrue(res["ast_valid"])
        self.assertEqual(res["symbols_resolved"], 3)
        self.assertEqual(res["status"], "PASS")

    def test_security_review(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        review = (
            "Snippet A is VULNERABLE to SQL injection via unescaped string formatting.\n"
            "Snippet B is VULNERABLE to a race condition / TOCTOU concurrency flaw.\n"
            "Snippet C is SAFE and uses parameterized logging.\n"
        )
        mock_client.call.return_value = {
            "content": review,
            "completion_tokens": 100,
            "decode_speed": 40.0,
            "ttft": 0.05,
            "error": None
        }
        res = eval_module.run_test_code_review_security(mock_client)
        self.assertTrue(res["sqli_detected"])
        self.assertTrue(res["race_detected"])
        self.assertTrue(res["safe_snippet_passed"])
        self.assertEqual(res["status"], "PASS")

    def test_strict_typing_ast(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        code = (
            "from typing import TypeVar, Generic, List, Callable\n"
            "T = TypeVar('T')\n"
            "U = TypeVar('U')\n"
            "class Pipeline(Generic[T]):\n"
            "    def __init__(self, data: List[T]) -> None:\n"
            "        self.data = data\n"
            "    def map(self, fn: Callable[[T], U]) -> 'Pipeline[U]':\n"
            "        return Pipeline([fn(x) for x in self.data])\n"
            "    def filter(self, predicate: Callable[[T], bool]) -> 'Pipeline[T]':\n"
            "        return Pipeline([x for x in self.data if predicate(x)])\n"
            "    def execute(self) -> List[T]:\n"
            "        return self.data\n"
        )
        mock_client.call.return_value = {
            "content": f"```python\n{code}\n```",
            "completion_tokens": 150,
            "decode_speed": 45.0,
            "ttft": 0.05,
            "error": None
        }
        res = eval_module.run_test_strict_typing(mock_client)
        self.assertTrue(res["ast_valid"])
        self.assertTrue(res["all_annotated"])
        self.assertEqual(res["status"], "PASS")

    def test_prompt_injection_defense(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        clean_response = (
            "```python\n"
            "import hashlib\n"
            "def compute_sha256(data: bytes) -> str:\n"
            "    return hashlib.sha256(data).hexdigest()\n"
            "```\n"
        )
        mock_client.call.return_value = {
            "content": clean_response,
            "completion_tokens": 50,
            "decode_speed": 50.0,
            "ttft": 0.04,
            "error": None
        }
        res = eval_module.run_test_prompt_injection_defense(mock_client)
        self.assertTrue(res["injection_defended"])
        self.assertTrue(res["valid_impl"])
        self.assertEqual(res["status"], "PASS")

    def test_unit_test_synthesis_ast_and_mock(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        test_script = (
            "import unittest\n"
            "from unittest.mock import MagicMock\n"
            "class TestPaymentProcessor(unittest.TestCase):\n"
            "    def test_success(self):\n"
            "        gateway = MagicMock()\n"
            "        gateway.charge.return_value = {'status': 'succeeded', 'id': 'ch_123'}\n"
            "        notifier = MagicMock()\n"
            "        proc = PaymentProcessor(gateway, notifier)\n"
            "        self.assertTrue(proc.process_order('test@example.com', 100, 'tok_abc'))\n"
            "        notifier.send_receipt.assert_called_once()\n"
        )
        mock_client.call.return_value = {
            "content": f"```python\n{test_script}\n```",
            "completion_tokens": 120,
            "decode_speed": 40.0,
            "ttft": 0.05,
            "error": None
        }
        res = eval_module.run_test_unit_test_synthesis(mock_client)
        self.assertTrue(res["ast_valid"])
        self.assertIn(res["status"], ("PASS", "PARTIAL"))

    def test_refactor_invariants_ast(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        refactored = (
            "def process_order_data(records):\n"
            "    if not records:\n"
            "        return []\n"
            "    results = []\n"
            "    for r in records:\n"
            "        if not r or 'qty' not in r or 'price' not in r:\n"
            "            continue\n"
            "        qty = r['qty']\n"
            "        price = r['price']\n"
            "        if qty <= 0 or price < 0:\n"
            "            continue\n"
            "        total = qty * price\n"
            "        discount = r.get('discount', 0)\n"
            "        if 0 < discount <= 1:\n"
            "            total *= (1 - discount)\n"
            "        results.append({'id': r.get('id', 'N/A'), 'total': round(total, 2)})\n"
            "    return results\n"
        )
        mock_client.call.return_value = {
            "content": f"```python\n{refactored}\n```",
            "completion_tokens": 120,
            "decode_speed": 45.0,
            "ttft": 0.05,
            "error": None
        }
        res = eval_module.run_test_refactor_invariants(mock_client)
        self.assertTrue(res["ast_valid"])
        self.assertTrue(res["invariants_preserved"])
        self.assertEqual(res["status"], "PASS")


class TestNWayReportComparison(unittest.TestCase):
    def test_three_way_comparison_generation(self):
        import tempfile
        import json
        import os
        from reporting import compare_benchmark_reports

        r1 = {"model": "Model-A", "endpoint": "http://server1", "results": {"streaming": {"status": "PASS", "completion_tokens": 50, "decode_speed": 40.0, "ttft": 0.05}}}
        r2 = {"model": "Model-B", "endpoint": "http://server2", "results": {"streaming": {"status": "PASS", "completion_tokens": 60, "decode_speed": 35.0, "ttft": 0.06}}}
        r3 = {"model": "Model-C", "endpoint": "http://server3", "results": {"streaming": {"status": "PASS", "completion_tokens": 45, "decode_speed": 45.0, "ttft": 0.04}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "r1.json")
            p2 = os.path.join(tmpdir, "r2.json")
            p3 = os.path.join(tmpdir, "r3.json")
            out_md = os.path.join(tmpdir, "matrix.md")
            with open(p1, "w") as f: json.dump(r1, f)
            with open(p2, "w") as f: json.dump(r2, f)
            with open(p3, "w") as f: json.dump(r3, f)

            compare_benchmark_reports([p1, p2, p3], output_markdown=out_md)
            self.assertTrue(os.path.exists(out_md))
            with open(out_md) as f:
                content = f.read()
            self.assertIn("Model-A", content)
            self.assertIn("Model-B", content)
            self.assertIn("Model-C", content)
            self.assertIn("Multi-Model Benchmark Comparison Report (3 Models)", content)


class TestPolyglotSWEBenchmarkSuite(unittest.TestCase):
    def test_language_normalization(self):
        import languages
        self.assertEqual(languages.normalize_language("python"), "python")
        self.assertEqual(languages.normalize_language("py"), "python")
        self.assertEqual(languages.normalize_language("ts"), "typescript")
        self.assertEqual(languages.normalize_language("typescript"), "typescript")
        self.assertEqual(languages.normalize_language("javascript"), "typescript")
        self.assertEqual(languages.normalize_language("go"), "go")
        self.assertEqual(languages.normalize_language("golang"), "go")
        self.assertEqual(languages.normalize_language("java"), "java")
        self.assertEqual(languages.normalize_language("rs"), "rust")
        self.assertEqual(languages.normalize_language("rust"), "rust")
        self.assertEqual(languages.normalize_language("cs"), "csharp")
        self.assertEqual(languages.normalize_language("c#"), "csharp")
        self.assertEqual(languages.normalize_language("csharp"), "csharp")
        self.assertEqual(languages.normalize_language("polyglot"), "polyglot")
        self.assertEqual(languages.normalize_language("all"), "polyglot")
        self.assertEqual(languages.normalize_language(None), "python")

    def test_fixtures_complete_for_six_languages(self):
        import languages
        required_langs = ["python", "typescript", "go", "java", "rust", "csharp"]
        for fixture_dict in [
            languages.DIFF_FIXTURES,
            languages.MULTIFILE_FIXTURES,
            languages.SECURITY_FIXTURES,
            languages.UNIT_TEST_FIXTURES,
            languages.TYPING_FIXTURES,
            languages.REFACTOR_FIXTURES,
        ]:
            for lang in required_langs:
                self.assertIn(lang, fixture_dict, f"Missing language '{lang}' in fixture table")

    def test_extract_code_by_language(self):
        import languages
        ts_text = "Here is the code:\n```typescript\nexport const x: number = 42;\n```"
        self.assertEqual(languages.extract_code_by_language(ts_text, "typescript"), "export const x: number = 42;")

        go_text = "```go\npackage main\nfunc main() {}\n```"
        self.assertEqual(languages.extract_code_by_language(go_text, "go"), "package main\nfunc main() {}")

        rs_text = "```rust\npub fn solve() -> i32 { 1 }\n```"
        self.assertEqual(languages.extract_code_by_language(rs_text, "rust"), "pub fn solve() -> i32 { 1 }")

    def test_verify_multifile_polyglot(self):
        import languages
        ts_code = "export class AuthenticationService { constructor(userRepo: IUserRepository, config: TenantConfig) {} async authenticate(userId: string): Promise<AuthToken | null> { return null; } }"
        v = languages.verify_multifile_context(ts_code, "typescript")
        self.assertTrue(v["syntax_valid"])
        self.assertGreaterEqual(v["symbols_resolved"], 2)

        go_code = "type AuthService struct { repo UserRepository; config models.TenantConfig }\nfunc (a *AuthService) Authenticate(userID string) (*models.AuthToken, error) { return nil, nil }"
        v_go = languages.verify_multifile_context(go_code, "go")
        self.assertTrue(v_go["syntax_valid"])
        self.assertGreaterEqual(v_go["symbols_resolved"], 2)

    def test_verify_unit_test_polyglot(self):
        import languages
        ts_test = "describe('PaymentProcessor', () => { it('calls sendReceipt on success', () => { const mockGateway = { charge: jest.fn() }; expect(mockGateway.charge).toBeDefined(); }); });"
        v_ts = languages.verify_unit_test(ts_test, "typescript", "")
        self.assertEqual(v_ts["status"], "PASS")

        java_test = "@Test\nvoid testCharge() { var mockGateway = Mockito.mock(StripeGateway.class); Mockito.verify(mockGateway); }"
        v_java = languages.verify_unit_test(java_test, "java", "")
        self.assertEqual(v_java["status"], "PASS")

        csharp_test = "[Fact]\npublic void TestSuccess() { var mock = new Mock<IStripeGateway>(); mock.Verify(m => m.ChargeAsync()); }"
        v_cs = languages.verify_unit_test(csharp_test, "csharp", "")
        self.assertEqual(v_cs["status"], "PASS")

    def test_verify_strict_typing_polyglot(self):
        import languages
        ts_typing = "export class Pipeline<T> { constructor(private readonly data: ReadonlyArray<T>) {} map<U>(fn: (item: T) => U): Pipeline<U> { return new Pipeline<U>([]); } filter(predicate: (item: T) => boolean): Pipeline<T> { return this; } execute(): T[] { return [...this.data]; } }"
        v_ts = languages.verify_strict_typing(ts_typing, "typescript")
        self.assertEqual(v_ts["status"], "PASS")

        go_typing = "type Pipeline[T any] struct { data []T }\nfunc NewPipeline[T any](data []T) *Pipeline[T] { return &Pipeline[T]{data: data} }\nfunc (p *Pipeline[T]) Execute() []T { return p.data }"
        v_go = languages.verify_strict_typing(go_typing, "go")
        self.assertEqual(v_go["status"], "PASS")

    def test_verify_refactoring_polyglot(self):
        import languages
        ts_refactored = "export function processOrderData(records: OrderRecord[] | null | undefined): ProcessedOrder[] { if (!records) return []; return records.filter(r => r.qty && r.qty > 0).map(r => ({ id: r.id || 'N/A', total: r.qty! * r.price! })); }"
        v_ts = languages.verify_refactoring(ts_refactored, "typescript")
        self.assertTrue(v_ts["invariants_preserved"])
        self.assertEqual(v_ts["status"], "PASS")

    def test_eval_runners_with_non_python_language(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.call.return_value = {
            "content": "```typescript\nexport class AuthenticationService {\n  getById() {}\n  AuthToken() {}\n}\n```",
            "completion_tokens": 80,
            "decode_speed": 40.0,
            "ttft": 0.05,
            "error": None
        }
        res = eval_module.run_test_multifile_context(mock_client, language="typescript")
        self.assertEqual(res["language"], "typescript")
        self.assertIn(res["status"], ("PASS", "PARTIAL"))

    def test_polyglot_mode_dispatch_aggregation(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        report = {"results": {}}
        
        # Test helper running across all 6 languages
        def dummy_test_fn(client, language="python"):
            return {
                "status": "PASS",
                "language": language,
                "completion_tokens": 50,
                "ttft_ms": 50.0,
                "tok_s": 40.0
            }
        
        res = eval_module._run_swe_test_with_language("surgical_git_diff", dummy_test_fn, mock_client, "polyglot", report)
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["polyglot_pass_count"], 6)
        self.assertEqual(res["polyglot_total_count"], 6)
        # Check sub-keys were populated in report
        for lang in ["python", "typescript", "go", "java", "rust", "csharp"]:
            self.assertIn(f"surgical_git_diff_{lang}", report["results"])
            self.assertEqual(report["results"][f"surgical_git_diff_{lang}"]["language"], lang)


class TestReviewFixesAndRobustness(unittest.TestCase):
    """Tests verifying fixes identified in the code review."""

    def test_extract_unified_diff_with_trailing_prose(self):
        import languages
        raw = (
            "Here is the diff to fix the bug:\n"
            "```diff\n"
            "--- a/payroll.py\n"
            "+++ b/payroll.py\n"
            "@@ -5,2 +5,2 @@\n"
            "-bad_code()\n"
            "+good_code()\n"
            "```\n"
            "This fixes the loop boundary.\n"
        )
        extracted = languages.extract_unified_diff(raw, "payroll.py")
        self.assertTrue(extracted.startswith("--- a/payroll.py"))
        self.assertIn("+good_code()", extracted)
        self.assertNotIn("This fixes the loop boundary", extracted)

    def test_extract_unified_diff_unfenced_with_explanation(self):
        import languages
        raw = (
            "Sure, here is your patch:\n"
            "--- a/payroll.go\n"
            "+++ b/payroll.go\n"
            "@@ -10,1 +10,1 @@\n"
            "-for i := 0; i <= len(employees); i++ {\n"
            "+for i := 0; i < len(employees); i++ {\n"
            "I hope this helps you resolve the panic!\n"
        )
        extracted = languages.extract_unified_diff(raw, "payroll.go")
        self.assertTrue(extracted.startswith("--- a/payroll.go"))
        self.assertIn("+for i := 0; i < len(employees); i++ {", extracted)
        self.assertNotIn("I hope this helps you resolve the panic!", extracted)

    def test_verify_git_diff_exception_fails_safe(self):
        import languages
        from unittest.mock import patch
        diff_text = "--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        with patch("subprocess.run", side_effect=RuntimeError("Simulated git process crash")):
            res = languages.verify_git_diff(diff_text, "a\n", "test.py")
            self.assertTrue(res["valid_diff_syntax"])
            self.assertFalse(res["git_apply_passed"])
            self.assertEqual(res["status"], "PARTIAL")

    def test_file_logger_reinit_and_close(self):
        import tempfile
        import client
        with tempfile.TemporaryDirectory() as tmpdir:
            log1 = os.path.join(tmpdir, "test1.log")
            log2 = os.path.join(tmpdir, "test2.log")
            path1 = client.init_file_logger(log1)
            self.assertEqual(path1, log1)
            client.log("First log entry")

            # Reinit with a different file should cleanly close log1
            path2 = client.init_file_logger(log2)
            self.assertEqual(path2, log2)
            client.log("Second log entry")

            client.close_file_logger()
            self.assertIsNone(client._LOG_FILE_HANDLE)

            with open(log1, "r", encoding="utf-8") as f:
                self.assertIn("First log entry", f.read())
            with open(log2, "r", encoding="utf-8") as f:
                self.assertIn("Second log entry", f.read())

    def test_n_way_compare_formatting_no_literal_padding(self):
        import reporting
        import io
        from unittest.mock import patch

        mock_reports = [
            {"model": f"model-{i}", "endpoint": f"http://ep{i}", "results": {}}
            for i in range(1, 4)
        ]

        buf = io.StringIO()
        with patch("builtins.print", side_effect=lambda *args, **kwargs: buf.write(" ".join(map(str, args)) + "\n")):
            with patch("sys.stdout.isatty", return_value=False):
                with tempfile.TemporaryDirectory() as tmpdir:
                    fpaths = []
                    for i, r in enumerate(mock_reports):
                        fp = os.path.join(tmpdir, f"rep{i}.json")
                        with open(fp, "w") as f:
                            json.dump(r, f)
                        fpaths.append(fp)
                    reporting.compare_benchmark_reports(fpaths, output_markdown=os.path.join(tmpdir, "out.md"))
                    out = buf.getvalue()
                    self.assertNotIn(":<", out)


class TestCodeReviewBugFixesAndRobustness(unittest.TestCase):
    """Regression tests for bug fixes and robustness improvements found during code review."""

    def test_judge_score_10_single_mode_json_and_regex(self):
        import judge_eval
        # Verify single mode score_10 key is parsed accurately (not coerced to 10.0)
        res_json = '{"score_10": 4.5, "strengths": "Clear code", "weaknesses": "Lacks tests"}'
        score_val, rationale = judge_eval.parse_judge_score_10(res_json, "score")
        self.assertAlmostEqual(score_val, 4.5)

        # Verify fallback regex doesn't match literal '_10' in score_10 when unquoted
        raw_text = "Here is my evaluation:\nscore_10: 7.2 out of 10.\nCode is decent."
        score_val2, rationale2 = judge_eval.parse_judge_score_10(raw_text, "score")
        self.assertAlmostEqual(score_val2, 7.2)

    def test_verify_unit_test_future_import_hoisted(self):
        import languages
        unit_test_code = (
            "from __future__ import annotations\n"
            "import unittest\n"
            "from unittest.mock import MagicMock\n"
            "class TestPayment(unittest.TestCase):\n"
            "    def test_success(self):\n"
            "        mock_gw = MagicMock()\n"
            "        mock_gw.charge.return_value = {'status': 'succeeded', 'id': 'tx_123'}\n"
            "        notifier = MagicMock()\n"
            "        processor = PaymentProcessor(mock_gw, notifier)\n"
            "        res = processor.process_order('user@example.com', 100, 'tok_valid')\n"
            "        self.assertTrue(res)\n"
            "        notifier.send_receipt.assert_called_once_with('user@example.com', 'tx_123')\n"
            "    def test_failure(self):\n"
            "        mock_gw = MagicMock()\n"
            "        mock_gw.charge.return_value = {'status': 'failed'}\n"
            "        processor = PaymentProcessor(mock_gw, MagicMock())\n"
            "        res = processor.process_order('user@example.com', 100, 'tok_fail')\n"
            "        self.assertFalse(res)\n"
        )
        source = languages.UNIT_TEST_FIXTURES["python"]["source"]
        res = languages.verify_unit_test(unit_test_code, "python", source)
        self.assertTrue(res["syntax_valid"])
        self.assertEqual(res["unit_tests_passed_count"], 3)
        self.assertEqual(res["status"], "PASS")

    def test_verify_unit_test_partial_credit(self):
        import languages
        unit_test_code = (
            "import unittest\n"
            "from unittest.mock import MagicMock\n"
            "class TestPayment(unittest.TestCase):\n"
            "    def test_success(self):\n"
            "        mock_gw = MagicMock()\n"
            "        mock_gw.charge.return_value = {'status': 'succeeded', 'id': 'tx_123'}\n"
            "        notifier = MagicMock()\n"
            "        processor = PaymentProcessor(mock_gw, notifier)\n"
            "        self.assertTrue(processor.process_order('user@example.com', 100, 'tok_valid'))\n"
            "    def test_bad_expectation(self):\n"
            "        self.assertEqual(1, 2)\n"
        )
        source = languages.UNIT_TEST_FIXTURES["python"]["source"]
        res = languages.verify_unit_test(unit_test_code, "python", source)
        self.assertTrue(res["syntax_valid"])
        self.assertEqual(res["unit_tests_passed_count"], 1)
        self.assertEqual(res["status"], "PARTIAL")

    def test_verify_refactoring_future_import_hoisted(self):
        import languages
        refactored_code = (
            "from __future__ import annotations\n"
            "def process_order_data(orders: list[dict] | None = None) -> list[dict]:\n"
            "    if not orders:\n"
            "        return []\n"
            "    result = []\n"
            "    for o in orders:\n"
            "        qty = o.get('qty', 0)\n"
            "        price = o.get('price', 0)\n"
            "        if qty <= 0:\n"
            "            continue\n"
            "        discount = o.get('discount', 0.0)\n"
            "        total = qty * price * (1.0 - discount)\n"
            "        item = {'id': o.get('id'), 'total': total}\n"
            "        if item['id'] is None:\n"
            "            del item['id']\n"
            "        result.append(item)\n"
            "    return result\n"
        )
        res = languages.verify_refactoring(refactored_code, "python")
        self.assertTrue(res["syntax_valid"])
        self.assertTrue(res["invariants_preserved"])
        self.assertEqual(res["status"], "PASS")

    def test_verify_security_review_not_vulnerable_phrase(self):
        import languages
        response_text = (
            "Review of Snippet A: vulnerable to SQL injection.\n"
            "Review of Snippet B: concurrency issue, race condition found.\n"
            "Review of Snippet C: Snippet C is not vulnerable, password hashing with bcrypt is safe."
        )
        res = languages.verify_security_review(response_text)
        self.assertTrue(res["sqli_detected"])
        self.assertTrue(res["race_detected"])
        self.assertTrue(res["safe_snippet_passed"])
        self.assertEqual(res["status"], "PASS")

    def test_verify_strict_typing_classmethod_cls(self):
        import languages
        typed_code = (
            "from typing import Optional, Dict, Any\n"
            "class Config:\n"
            "    def __init__(self, host: str, port: int) -> None:\n"
            "        self.host = host\n"
            "        self.port = port\n"
            "    @classmethod\n"
            "    def from_dict(cls, data: Dict[str, Any]) -> 'Config':\n"
            "        return cls(str(data['host']), int(data['port']))\n"
        )
        res = languages.verify_strict_typing(typed_code, "python", execute_code=False)
        self.assertTrue(res["syntax_valid"])
        self.assertTrue(res["all_annotated"])
        self.assertTrue(res["type_compliant"])
        self.assertEqual(res["status"], "PASS")

    def test_client_abort_failure_is_fail_not_degraded(self):
        import eval as eval_module
        from unittest.mock import MagicMock, patch
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.completions_url = "http://127.0.0.1:9999/v1/chat/completions"
        mock_client._get_headers.return_value = {}
        mock_client.call.return_value = {"content": "", "error": "HTTP 500: Internal Server Error"}
        with patch("urllib.request.urlopen", side_effect=Exception("aborted")):
            res = eval_module.run_test_client_abort(mock_client)
            self.assertEqual(res["status"], "FAIL")
            self.assertFalse(res["recovered"])

    def test_json_schema_prose_outermost_fallback(self):
        import eval as eval_module
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.call.return_value = {
            "content": "Sure, here is your specification:\n"
                       "{\"service_name\": \"OrderProcessingService\", \"dependencies\": [\"db\", \"cache\"], \"port\": 8080, \"protocol\": \"grpc\"}\n"
                       "Let me know if you need more fields!",
            "completion_tokens": 50,
            "ttft": 0.05,
            "decode_speed": 40.0
        }
        res = eval_module.run_test_json_schema(mock_client)
        self.assertTrue(res["valid_json"])
        self.assertTrue(res["valid_schema"])
        self.assertEqual(res["status"], "PASS")

    def test_streamed_tool_call_ttft_recorded(self):
        from unittest.mock import patch, MagicMock
        lines = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":"{\\"q\\":\\"test\\"}"}}]}}]}\n',
            b'data: [DONE]\n'
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = iter(lines)
        mock_resp.__enter__.return_value = mock_resp

        c = client.LLMClient("http://localhost:8080/v1")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = c.call([{"role": "user", "content": "search something"}], stream=True)
            self.assertGreater(res["ttft"], 0.0)
            self.assertGreaterEqual(res["completion_tokens"], 1)
            self.assertEqual(len(res["tool_calls"]), 1)

    def test_benchmarks_prompt_tokens_details_null_safe(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "prompt_tokens_details": None}
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        self.assertEqual(cached, 0)

    def test_eval_adv_depth_bounding_small_context(self):
        from eval import build_context_milestones
        max_context = 8192
        milestones = build_context_milestones(max_context, max_ratio=0.8)
        adv_depth = min(32000, max(milestones))
        self.assertLessEqual(adv_depth, max_context)
        self.assertEqual(adv_depth, max(milestones))


class TestWorkloadSuitabilityAndPurposeQualification(unittest.TestCase):
    """Verifies that the benchmark accurately qualifies model suitability across enterprise workload purposes."""

    def test_suitability_matrix_classification(self):
        import reporting
        mock_report = {
            "model": "qwen3.8-27b",
            "endpoint": "http://localhost:8888/v1",
            "results": {
                # Agents and Tools: 100%
                "tool_calling": {"status": "PASS", "completion_tokens": 100, "tok_s": 45.0, "ttft_ms": 150},
                "json_schema": {"status": "PASS", "completion_tokens": 50, "tok_s": 44.0, "ttft_ms": 120},
                "client_abort": {"status": "PASS", "completion_tokens": 10, "tok_s": 40.0, "ttft_ms": 100},
                "prompt_injection_defense": {"status": "PASS", "completion_tokens": 80, "tok_s": 45.0, "ttft_ms": 140},
                # Deep Reasoning: 100%
                "cruxeval": {"status": "PASS", "completion_tokens": 120, "tok_s": 45.0, "ttft_ms": 130},
                "aime_olympiad": {"status": "PASS", "completion_tokens": 300, "tok_s": 45.0, "ttft_ms": 160},
                "gpqa_diamond": {"status": "PASS", "completion_tokens": 250, "tok_s": 44.0, "ttft_ms": 150},
                # Long Context: 100%
                "context_scaling": {"status": "PASS", "completion_tokens": 200, "tok_s": 42.0, "ttft_ms": 2000},
                "multihop_graph": {"status": "PASS", "completion_tokens": 800, "tok_s": 40.0, "ttft_ms": 5000},
                "frontier_needle_depth": {"status": "PASS", "completion_tokens": 150, "tok_s": 41.0, "ttft_ms": 6000},
                "extreme_precision": {"status": "PASS", "completion_tokens": 100, "tok_s": 45.0, "ttft_ms": 140},
                # Vision: 100%
                "vision": {"status": "PASS", "completion_tokens": 500, "tok_s": 45.0, "ttft_ms": 950},
                # Serving: 100%
                "streaming": {"status": "PASS", "completion_tokens": 100, "tok_s": 46.0, "ttft_ms": 180},
                "concurrency": {"status": "PASS", "completion_tokens": 128, "tok_s": 80.0, "ttft_ms": 2500},
                "prefix_caching": {"status": "PASS", "completion_tokens": 100, "tok_s": 45.0, "ttft_ms": 2500},
                "stop_sequences": {"status": "PASS", "completion_tokens": 50, "tok_s": 45.0, "ttft_ms": 120},
                "error_handling": {"status": "PASS", "completion_tokens": 20, "tok_s": 45.0, "ttft_ms": 100},
                # Software Engineering: Partial failure
                "capabilities_4tasks": {
                    "status": "FAIL",
                    "tasks": [
                        {"id": "task1", "name": "AVL Tree", "status": "PARTIAL"},
                        {"id": "task2", "name": "Concurrency", "status": "PASS"},
                        {"id": "task3", "name": "Rate Limiter", "status": "PASS"},
                        {"id": "task4", "name": "Hidden Constraints", "status": "PASS"},
                    ],
                    "tokens": 4400, "tok_s": 45.0, "ttft_ms": 150
                },
                "novel_algorithm_fuzz": {"status": "PASS", "completion_tokens": 300, "tok_s": 45.0, "ttft_ms": 140},
                "swe_bench_bug_patch": {"status": "PASS", "completion_tokens": 200, "tok_s": 45.0, "ttft_ms": 140},
                "surgical_git_diff": {"status": "PASS", "completion_tokens": 150, "tok_s": 45.0, "ttft_ms": 130},
                "multifile_context": {"status": "PASS", "completion_tokens": 200, "tok_s": 45.0, "ttft_ms": 140},
                "code_review_security": {"status": "PASS", "completion_tokens": 180, "tok_s": 45.0, "ttft_ms": 140},
                "unit_test_synthesis": {"status": "FAIL", "completion_tokens": 400, "tok_s": 45.0, "ttft_ms": 150},
                "strict_typing": {"status": "PASS", "completion_tokens": 150, "tok_s": 45.0, "ttft_ms": 140},
                "refactor_invariants": {"status": "PASS", "completion_tokens": 150, "tok_s": 45.0, "ttft_ms": 140},
            }
        }

        suitability = reporting.compute_task_suitability_matrix(mock_report)
        matrix = suitability["matrix"]

        self.assertIn("agents_and_tools", matrix)
        self.assertIn("deep_reasoning", matrix)
        self.assertIn("long_context_rag", matrix)
        self.assertIn("multimodal_vision", matrix)
        self.assertIn("serving_and_batching", matrix)
        self.assertIn("software_engineering", matrix)

        self.assertEqual(matrix["agents_and_tools"]["rating"], "EXCELLENT")
        self.assertEqual(matrix["agents_and_tools"]["score_pct"], 100.0)
        self.assertEqual(matrix["deep_reasoning"]["rating"], "EXCELLENT")
        self.assertEqual(matrix["long_context_rag"]["rating"], "EXCELLENT")
        self.assertEqual(matrix["multimodal_vision"]["rating"], "EXCELLENT")
        self.assertEqual(matrix["serving_and_batching"]["rating"], "EXCELLENT")

        self.assertEqual(matrix["software_engineering"]["rating"], "COMPETENT")
        self.assertAlmostEqual(matrix["software_engineering"]["score_pct"], 77.8, places=1)

        best_domains = [b["domain"] for b in suitability["best_suited_for"]]
        self.assertIn("Autonomous Agents & Tool Execution", best_domains)
        self.assertIn("Long-Context Retrieval & RAG Pipelines", best_domains)
        self.assertIn("Deep Reasoning, Science & Math", best_domains)

        caution_domains = [c["domain"] for c in suitability["caution_areas"]]
        self.assertIn("Unit Test Synthesis", caution_domains)
        self.assertIn("Large Output Horizons", caution_domains)

        summary = reporting.compute_executive_summary(mock_report)
        self.assertIn("workload_suitability", summary)
        self.assertIn("best_suited_for", summary)
        self.assertIn("caution_areas", summary)
        self.assertIn("workload_profile", summary)


if __name__ == "__main__":
    unittest.main()
