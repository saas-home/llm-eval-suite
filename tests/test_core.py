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
        self.assertEqual(reporting.parse_selected_tests("1,3,22"), {1, 3, 22})

    def test_range(self):
        self.assertEqual(reporting.parse_selected_tests("1-5"), {1, 2, 3, 4, 5})

    def test_alias(self):
        self.assertEqual(reporting.parse_selected_tests("aime"), {22})
        self.assertEqual(reporting.parse_selected_tests("context_scaling"), {14})

    def test_suites(self):
        self.assertEqual(reporting.parse_selected_tests(None, "flagship"), set(range(1, 15)))
        self.assertEqual(reporting.parse_selected_tests(None, "adversarial"), set(range(15, 20)))
        self.assertEqual(reporting.parse_selected_tests(None, "frontier"), set(range(20, 23)))
        self.assertEqual(reporting.parse_selected_tests(None, "all"), set(range(1, 23)))

    def test_unknown_ignored(self):
        self.assertEqual(reporting.parse_selected_tests("1,not_a_test"), {1})

    def test_all_unknown_runs_nothing(self):
        # A typo in --test must NOT silently run the full suite.
        self.assertEqual(reporting.parse_selected_tests("foo,bar"), set())


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


if __name__ == "__main__":
    unittest.main()
