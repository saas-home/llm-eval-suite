#!/usr/bin/env python3
"""Tests for the decision report: sampling settings, reliability (pass^k),
contract review and summarisation graders (Tests 47-48), the workload test
(Test 46), use-case verdicts and the Markdown report."""

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import client as client_module
import eval as eval_module
import judge as judge_module
import judge_validation
import report_md
import suites
import usecases
import workload
from client import LLMClient


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return _Resp(body.encode())


# ----------------------------------------------------------------------------
# client.py
# ----------------------------------------------------------------------------
class TestClientSampling(unittest.TestCase):
    def _payload(self, **kw):
        sent = []

        def fake_urlopen(req, timeout=None):
            sent.append(json.loads(req.data.decode()))
            return _Resp(json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())

        with mock.patch.object(client_module.urllib.request, "urlopen", side_effect=fake_urlopen):
            LLMClient("http://srv/v1", model="m").call([{"role": "user", "content": "hi"}], **kw)
        return sent[0]

    def test_sampling_overrides_temperature_and_adds_fields(self):
        p = self._payload(temperature=0.0, sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertEqual((p["temperature"], p["top_p"], p["top_k"]), (1.0, 0.95, 20))

    def test_unknown_sampling_keys_are_not_sent(self):
        p = self._payload(sampling={"top_p": 0.9, "model": "other", "messages": []})
        self.assertEqual(p["model"], "m")
        self.assertEqual(p["messages"][0]["content"], "hi")
        self.assertNotIn("source", self._payload(sampling={"source": "card"}))


class TestAnswerTiming(unittest.TestCase):
    def test_answer_start_is_after_reasoning(self):
        chunks = [{"choices": [{"delta": {"reasoning_content": "think"}}]},
                  {"choices": [{"delta": {"reasoning_content": " more"}}]},
                  {"choices": [{"delta": {"content": "Answer"}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 3}}]
        times = iter([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0])
        with mock.patch.object(client_module.urllib.request, "urlopen", return_value=_sse(chunks)), \
                mock.patch.object(client_module.time, "perf_counter", side_effect=lambda: next(times)):
            res = LLMClient("http://srv/v1", model="m").call([{"role": "user", "content": "q"}], stream=True)
        self.assertEqual(res["ttft"], 1.0)
        self.assertEqual(res["answer_ttft"], 5.0)
        self.assertEqual((res["reasoning_chunks"], res["content_chunks"]), (2, 1))

    def test_decode_speed_ends_at_last_token_not_delayed_usage(self):
        chunks = [{"choices": [{"delta": {"content": "first"}}]},
                  {"choices": [{"delta": {"content": "last"}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 5}}]
        times = iter([0.0, 1.0, 3.0, 13.0])
        with mock.patch.object(client_module.urllib.request, "urlopen", return_value=_sse(chunks)), \
                mock.patch.object(client_module.time, "perf_counter", side_effect=lambda: next(times)):
            res = LLMClient("http://srv/v1", model="m").call([{"role": "user", "content": "q"}], stream=True)
        self.assertEqual(res["total_time"], 13.0)
        self.assertEqual(res["decode_speed"], 2.0)

    def test_single_chunk_does_not_use_delayed_trailer_as_decode_time(self):
        chunks = [{"choices": [{"delta": {"content": "all tokens"}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 5}}]
        times = iter([0.0, 1.0, 13.0])
        with mock.patch.object(client_module.urllib.request, "urlopen", return_value=_sse(chunks)), \
                mock.patch.object(client_module.time, "perf_counter", side_effect=lambda: next(times)):
            res = LLMClient("http://srv/v1", model="m").call([{"role": "user", "content": "q"}], stream=True)
        self.assertEqual(res["total_time"], 13.0)
        self.assertEqual(res["decode_speed"], 0.0)


class TestStreamIntegrity(unittest.TestCase):
    def call_with_body(self, body):
        with mock.patch.object(client_module.urllib.request, "urlopen", return_value=_Resp(body)):
            return LLMClient("http://srv/v1", model="m").call(
                [{"role": "user", "content": "q"}], stream=True, retries=0)

    def test_unterminated_stream_is_an_infrastructure_failure(self):
        body = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        res = self.call_with_body(body)
        self.assertEqual(res["failure_kind"], "infra")
        self.assertIn("ended", res["error"].lower())
        self.assertEqual(res["content"], "partial")

    def test_malformed_stream_event_is_not_silently_discarded(self):
        body = (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                b'data: {invalid json}\n\n'
                b'data: [DONE]\n\n')
        res = self.call_with_body(body)
        self.assertEqual(res["failure_kind"], "infra")
        self.assertIn("malformed", res["error"].lower())
        self.assertEqual(res["content"], "partial")

    def test_timeout_after_partial_stream_keeps_content_and_does_not_retry(self):
        class Interrupted(_Resp):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                raise TimeoutError("stream timed out")

        with mock.patch.object(client_module.urllib.request, "urlopen",
                               side_effect=lambda *args, **kwargs: Interrupted(b"")) as urlopen, \
             mock.patch.object(client_module.time, "sleep") as sleep:
            res = LLMClient("http://srv/v1", model="m").call(
                [{"role": "user", "content": "q"}], stream=True)
        self.assertEqual(res["failure_kind"], "infra")
        self.assertEqual(res["content"], "partial")
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_finish_reason_without_done_is_accepted(self):
        body = (b'data: {"choices":[{"delta":{"content":"complete"}}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        res = self.call_with_body(body)
        self.assertNotIn("error", res)
        self.assertEqual((res["content"], res["finish_reason"]), ("complete", "stop"))

    def test_multiline_sse_event_and_keepalive_are_accepted(self):
        body = (b': keepalive\n\n'
                b'data: {"choices": [\n'
                b'data: {"delta": {"content": "joined"}}]}\n\n'
                b'data: [DONE]\n\n')
        res = self.call_with_body(body)
        self.assertNotIn("error", res)
        self.assertEqual(res["content"], "joined")

    def test_non_object_event_is_an_explicit_stream_failure(self):
        res = self.call_with_body(b'data: null\n\ndata: [DONE]\n\n')
        self.assertEqual(res["failure_kind"], "infra")
        self.assertIn("malformed", res["error"].lower())

    def test_done_without_any_completion_choice_is_a_stream_failure(self):
        body = b'data: {"choices":[],"usage":{"completion_tokens":0}}\n\ndata: [DONE]\n\n'
        res = self.call_with_body(body)
        self.assertEqual(res["failure_kind"], "infra")
        self.assertIn("choice", res["error"].lower())

    def test_invalid_utf8_is_not_graded_as_model_output(self):
        body = (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"\xff"}}]}\n\n'
                b'data: [DONE]\n\n')
        res = self.call_with_body(body)
        self.assertEqual(res["failure_kind"], "infra")
        self.assertEqual(res["content"], "partial")

    def test_empty_but_well_formed_completion_is_accepted(self):
        body = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        res = self.call_with_body(body)
        self.assertNotIn("error", res)
        self.assertEqual(res["content"], "")


class TestResponseIntegrity(unittest.TestCase):
    def test_nonstream_response_without_a_completion_is_an_infra_failure(self):
        for response in ({}, {"choices": []}, {"choices": [{}]}, []):
            with self.subTest(response=response), \
                 mock.patch.object(client_module.urllib.request, "urlopen",
                                   return_value=_Resp(json.dumps(response).encode())) as urlopen:
                res = LLMClient("http://srv/v1", model="m").call(
                    [{"role": "user", "content": "q"}], stream=False, retries=0)
                self.assertEqual(res["failure_kind"], "infra")
                self.assertIn("malformed", res["error"].lower())
                self.assertEqual(urlopen.call_count, 1)

    def test_empty_but_well_formed_nonstream_completion_is_accepted(self):
        response = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        with mock.patch.object(client_module.urllib.request, "urlopen",
                               return_value=_Resp(json.dumps(response).encode())):
            res = LLMClient("http://srv/v1", model="m").call(
                [{"role": "user", "content": "q"}], stream=False, retries=0)
        self.assertNotIn("error", res)
        self.assertEqual(res["content"], "")


class TestCountPromptTokens(unittest.TestCase):
    def test_counts_with_the_chat_template(self):
        sent = []

        def fake(req, timeout=None):
            sent.append((req.full_url, json.loads(req.data.decode())))
            return _Resp(b'{"count": 11, "max_model_len": 524288}')

        with mock.patch.object(client_module.urllib.request, "urlopen", side_effect=fake):
            n = LLMClient("http://srv:8080/v1", model="m").count_prompt_tokens([{"role": "user", "content": "hi"}])
        self.assertEqual(n, 11)
        self.assertEqual(sent[0][0], "http://srv:8080/tokenize")
        self.assertEqual(sent[0][1]["messages"][0]["content"], "hi")

    def test_missing_endpoint_is_not_asked_again(self):
        import urllib.error
        err = urllib.error.HTTPError("http://srv:8080/tokenize", 404, "Not Found", {}, io.BytesIO(b""))
        c = LLMClient("http://srv:8080/v1", model="m")
        with mock.patch.object(client_module.urllib.request, "urlopen", side_effect=err) as u:
            self.assertIsNone(c.count_prompt_tokens([{"role": "user", "content": "hi"}]))
            self.assertIsNone(c.count_prompt_tokens([{"role": "user", "content": "hi"}]))
        self.assertEqual(u.call_count, 1)


class TestServerInfo(unittest.TestCase):
    METRICS = ('vllm:cache_config_info{cache_dtype="bfloat16",enable_prefix_caching="True",'
               'gpu_memory_utilization="0.88",kv_cache_memory_bytes="37000000000",kv_cache_size_tokens="1119425",'
               'hash_block_size="None"} 1.0\n')

    def test_parse_cache_config_types(self):
        self.assertEqual(client_module.parse_cache_config(self.METRICS), {
            "kv_cache_tokens": 1119425, "kv_cache_dtype": "bfloat16", "prefix_caching": True,
            "gpu_memory_utilization": 0.88, "kv_cache_memory_bytes": 37000000000})
        self.assertEqual(client_module.parse_cache_config("other_metric 1"), {})

    def test_weights_repo_from_cache_path(self):
        self.assertEqual(client_module.weights_repo_from_path(
            "/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9"),
            "nvidia/Qwen3.8-Flash-Next-FP4-Local")
        self.assertIsNone(client_module.weights_repo_from_path("/models/local"))

    def test_vllm_server(self):
        c = LLMClient("http://srv:8080/v1", model="M")
        jsons = {"http://srv:8080/version": {"version": "0.33.0"}}
        with mock.patch.object(c, "_get_json", side_effect=lambda url, timeout=5: jsons.get(url)), \
                mock.patch.object(c, "_get_text", return_value=self.METRICS), \
                mock.patch.object(c, "fetch_models", return_value=[
                    {"id": "other"}, {"id": "M", "owned_by": "vllm", "max_model_len": 524288,
                                      "root": "/hub/models--org--name/snapshots/x"}]):
            info = c.fetch_server_info()
        self.assertEqual((info["engine"], info["version"], info["weights"], info["max_model_len"]),
                         ("vLLM", "0.33.0", "org/name", 524288))
        self.assertEqual(info["kv_cache_tokens"], 1119425)

    def test_llama_cpp_server(self):
        c = LLMClient("http://srv:8080/v1", model="M")
        jsons = {"http://srv:8080/props": {"build_info": "b5000-abc"}}
        with mock.patch.object(c, "_get_json", side_effect=lambda url, timeout=5: jsons.get(url)), \
                mock.patch.object(c, "_get_text", return_value=""), \
                mock.patch.object(c, "fetch_models", return_value=[{"id": "M", "owned_by": "llamacpp"}]):
            info = c.fetch_server_info()
        self.assertEqual((info["engine"], info["version"]), ("llama.cpp", "b5000-abc"))


# ----------------------------------------------------------------------------
# suites.py
# ----------------------------------------------------------------------------
class TestSamplingSettings(unittest.TestCase):
    def tearDown(self):
        suites.SAMPLING = None

    def test_default(self):
        self.assertEqual(suites.sampling_for(1), {"temperature": 0.0})
        self.assertEqual(suites.sampling_for(3), {"temperature": 0.6})

    def test_configured_settings_apply_to_every_repeat_count(self):
        suites.SAMPLING = {"temperature": 1.0, "top_p": 0.95, "source": "model card"}
        self.assertEqual(suites.sampling_for(1), {"temperature": 1.0, "top_p": 0.95})

    def test_parse_sampling_flag(self):
        self.assertEqual(eval_module.parse_sampling("temperature=1.0, top_p=0.95,top_k=20"),
                         {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        with self.assertRaises(ValueError):
            eval_module.parse_sampling("temprature=1.0")
        with self.assertRaises(ValueError):
            eval_module.parse_sampling("top_p")


class TestReliability(unittest.TestCase):
    def test_pass_all_vs_pass_any(self):
        r = suites.reliability([[1, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1]], None)
        self.assertEqual((r["pass_all_repeats_pct"], r["pass_any_repeat_pct"], r["inconsistent_items"]),
                         (50.0, 75.0, 1))

    def test_checklist_tasks_pass_at_08(self):
        r = suites.reliability([[0.8, 0.9, 1.0], [0.8, 0.6, 1.0]], "writing_tasks")
        self.assertEqual(r["pass_all_repeats_pct"], 50.0)
        self.assertEqual(suites.reliability([[0.8]], None)["pass_all_repeats_pct"], 0.0)

    def test_run_benchmark_records_attempt_scores(self):
        answers = iter(["Answer: A", "Answer: B", "Answer: A", "Answer: A"])
        lock = threading.Lock()

        class C:
            def call(self, *a, **kw):
                with lock:
                    return {"content": next(answers), "completion_tokens": 1, "sampling_seen": kw.get("sampling")}

        items = [{"id": "q1", "question": "?", "options": {"A": "a", "B": "b"}, "answer": "A"},
                 {"id": "q2", "question": "?", "options": {"A": "a", "B": "b"}, "answer": "A"}]
        res = suites.run_benchmark(C(), "T", items, suites.mc_messages("x"), suites.grade_mc, 2, 2, workers=1)
        self.assertEqual([d["attempt_scores"] for d in res["details"]], [[1.0, 0.0], [1.0, 1.0]])
        self.assertEqual(res["reliability"]["pass_all_repeats_pct"], 50.0)
        self.assertEqual(res["sampling"], {"temperature": 0.6})

    def test_unscored_repeats_beyond_the_tolerance_prevent_pass_and_recommendation(self):
        class C:
            max_context = None

            def __init__(self, empty):
                self.calls, self.empty = 0, empty

            def call(self, *args, **kwargs):
                self.calls += 1
                content = "" if self.calls <= self.empty else "Here is the answer."
                return {"content": content, "completion_tokens": 1}

        judge = mock.Mock()
        judge.checklist.return_value = {"judged_by": "judge", "verdicts": [False]}
        items = [{"id": str(i), "prompt": "A safe question", "should_refuse": False} for i in range(40)]
        # 1 of 120 attempts (0.8%) is within the tolerance; 7 (5.8%) is not.
        for empty, status in ((1, "PASS"), (7, "PARTIAL")):
            with self.subTest(empty=empty), mock.patch.object(suites, "log"):
                result = suites.run_benchmark(
                    C(empty), "XSTest", items, suites.xstest_messages, suites.make_grade_xstest(judge),
                    samples=40, repeats=3, workers=1, key="xstest")
                self.assertEqual((result["unscored_attempts"], result["total_attempts"]), (empty, 120))
                self.assertEqual(result["status"], status)
                verdict = next(v for v in usecases.rank_use_cases({"xstest": result}) if v["key"] == "assistant")
                self.assertTrue(any("xstest" in reason and "unscored attempt" in reason
                                    for reason in verdict["reasons"]))
                if status == "PARTIAL":
                    self.assertEqual(verdict["verdict"], usecases.REVIEW)


class TestContractReviewGrader(unittest.TestCase):
    GOV = ("This Agreement shall be governed by and construed in accordance with the laws of the State of "
           "New York, without regard to its conflict of laws principles.")

    def test_absent_clause(self):
        self.assertEqual(suites.grade_clause_answer([], "NOT FOUND")[0], 1.0)
        score, d = suites.grade_clause_answer([], "QUOTE: This Agreement is governed by Delaware law in all respects.")
        self.assertEqual((score, d["outcome"]), (0.0, "invented a clause"))

    def test_partial_quote_of_long_clause(self):
        self.assertEqual(suites.grade_clause_answer(
            [self.GOV], "QUOTE: governed by and construed in accordance with the laws of the State of New York")[0], 1.0)

    def test_wrong_clause_and_miss(self):
        self.assertEqual(suites.grade_clause_answer(
            [self.GOV], "QUOTE: Either party may terminate this Agreement upon thirty days written notice.")[0], 0.0)
        self.assertEqual(suites.grade_clause_answer([self.GOV], "NOT FOUND")[0], 0.0)

    def test_quote_inside_reasoning_free_answer(self):
        """No QUOTE: marker: the whole answer is read as the quote."""
        self.assertEqual(suites.grade_clause_answer([self.GOV], self.GOV)[0], 1.0)

    def test_data_files_join(self):
        contracts, _ = suites.load_contracts()
        items, _ = suites.load_items("legal", "cuad_review")
        self.assertTrue(items)
        self.assertTrue(all(it["contract_id"] in contracts for it in items))
        self.assertTrue(any(not it["answers"] for it in items))  # absent clauses are tested


class _Judge:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = []

    def checklist(self, task, response, criteria, context=""):
        self.calls.append({"criteria": criteria, "context": context})
        return {"verdicts": self.verdicts, "passed": sum(v is True for v in self.verdicts), "total": len(criteria),
                "error": None, "failure_kind": None, "judged_by": "judge"}


class TestSummaryGrader(unittest.TestCase):
    ITEM = {"title": "A bill", "text": "The bill text.", "summary": " ".join(["word"] * 150),
            "points": ["p1", "p2", "p3", "p4"]}

    def grade(self, verdicts, words=100):
        judge = _Judge(verdicts)
        score, detail = suites.make_grade_summary(judge)(self.ITEM, {"content": " ".join(["w"] * words)})
        return score, detail, judge

    def test_coverage_and_judge_sees_the_bill(self):
        score, detail, judge = self.grade([True, True, True, False, True])
        self.assertEqual(score, 0.75)
        self.assertEqual(judge.calls[0]["context"], "The bill text.")
        self.assertEqual(len(judge.calls[0]["criteria"]), 5)

    def test_invented_statement_scores_zero(self):
        self.assertEqual(self.grade([True, True, True, True, False])[0], 0.0)

    def test_unanswered_faithfulness_is_not_a_pass(self):
        self.assertEqual(self.grade([True, True, True, True, None])[0], 0.0)

    def test_over_length_capped(self):
        self.assertEqual(self.grade([True, True, True, True, True], words=200)[0], 0.5)
        self.assertEqual(self.grade([True, True, True, True, True], words=160)[0], 1.0)  # within 10% slack


# ----------------------------------------------------------------------------
# workload.py
# ----------------------------------------------------------------------------
class TestWorkloadSettings(unittest.TestCase):
    def test_defaults_are_the_target_scenario(self):
        w = workload.resolve_workload(None, 524288)
        self.assertEqual((w["prompt_tokens"], w["users"], w["answer_tokens"]), (200000, 4, 2000))
        self.assertEqual(w["gates"], {"first_token_s": 90, "complete_s": 300, "per_user_tok_s": 15})

    def test_config_overrides_and_gate_merge(self):
        w = workload.resolve_workload({"users": 8, "gates": {"first_token_s": 30}, "unknown": 1}, None)
        self.assertEqual((w["users"], w["gates"]["first_token_s"], w["gates"]["complete_s"]), (8, 30, 300))
        self.assertNotIn("unknown", w)

    def test_cli_flags_workload_ttft_and_complete(self):
        import eval as ev
        parser = ev.build_parser()
        args = parser.parse_args(["--workload-ttft", "250.5", "--workload-complete", "500"])
        self.assertEqual(args.workload_ttft, 250.5)
        self.assertEqual(args.workload_complete, 500.0)

        for invalid in ["nan", "inf", "-5", "0"]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--workload-ttft", invalid])
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--workload-complete", invalid])

    def test_workload_line_reports_accuracy_and_gates(self):
        import report_md
        report = {
            "results": {
                "workload": {
                    "workload": {"prompt_tokens": 200000, "answer_tokens": 2000, "stagger_s": [0.0, 1.0],
                                 "gates": {"first_token_s": 90, "complete_s": 300, "per_user_tok_s": 15}},
                    "users": [{"first_token_s": 240.0}, {"first_token_s": 260.0}],
                    "goodput_users": 0,
                    "correct_users": 2,
                    "solo": {"meets_gates": True},
                    "servers_for_users": 2,
                    "basis": "1 user(s) per server"
                }
            }
        }
        line = report_md._workload_line(report)
        self.assertIn("2 of 2 clauses found correctly", line)
        self.assertIn("0 of 2 users within the gates", line)

    def test_workload_sizing_sla_context(self):
        import report_md
        report = {
            "results": {
                "workload": {
                    "workload": {"prompt_tokens": 200000, "answer_tokens": 2000, "stagger_s": [0.0, 1.0],
                                 "gates": {"first_token_s": 90, "complete_s": 300, "per_user_tok_s": 15}},
                    "users": [
                        {"start_offset_s": 0.0, "prompt_tokens": 200000, "first_token_s": 240.0,
                         "answer_start_s": 245.0, "complete_s": 320.0, "decode_tok_s": 25.0, "score": 1.0,
                         "meets_gates": False, "gate_checks": {"first_token": False, "complete": False}},
                        {"start_offset_s": 1.0, "prompt_tokens": 200000, "first_token_s": 260.0,
                         "answer_start_s": 265.0, "complete_s": 330.0, "decode_tok_s": 25.0, "score": 1.0,
                         "meets_gates": False, "gate_checks": {"first_token": False, "complete": False}},
                    ],
                    "goodput_users": 0,
                    "correct_users": 2,
                    "solo": {"start_offset_s": 0.0, "prompt_tokens": 200000, "first_token_s": 70.0,
                             "answer_start_s": 75.0, "complete_s": 120.0, "decode_tok_s": 30.0, "score": 1.0,
                             "meets_gates": True},
                    "servers_for_users": 2,
                    "sizing": {"users_per_server": 1, "servers_for_users": 2, "basis": "1 user alone met the gates"},
                }
            }
        }
        lines = report_md._sizing(report)
        text = "\n".join(lines)
        self.assertIn("SLA context:", text)
        self.assertIn("enterprise document review SLA", text)
        self.assertIn("2 and 3 concurrent users were not directly measured", text)

    def test_workload_sizing_truncated_response_excluded_from_sla(self):
        import report_md
        report = {
            "results": {
                "workload": {
                    "workload": {"prompt_tokens": 200000, "answer_tokens": 2000, "stagger_s": [0.0, 1.0],
                                 "gates": {"first_token_s": 90, "complete_s": 300, "per_user_tok_s": 15}},
                    "users": [
                        {"start_offset_s": 0.0, "prompt_tokens": 200000, "first_token_s": 240.0,
                         "answer_start_s": 245.0, "complete_s": 320.0, "decode_tok_s": 25.0, "score": 1.0,
                         "finish_reason": "length", "meets_gates": False},
                        {"start_offset_s": 1.0, "prompt_tokens": 200000, "first_token_s": 260.0,
                         "answer_start_s": 265.0, "complete_s": 330.0, "decode_tok_s": 25.0, "score": 1.0,
                         "finish_reason": "stop", "meets_gates": False},
                    ],
                    "goodput_users": 0,
                    "correct_users": 2,
                    "solo": {"first_token_s": 70.0, "meets_gates": True},
                    "servers_for_users": 2,
                    "sizing": {"users_per_server": 1, "servers_for_users": 2, "basis": "1 user alone met the gates"},
                }
            }
        }
        lines = report_md._sizing(report)
        text = "\n".join(lines)
        self.assertNotIn("all 2 users completed within an enterprise document review SLA", text)

    def test_workload_sizing_partial_goodput_under_relaxed_gate(self):
        import report_md
        report = {
            "results": {
                "workload": {
                    "workload": {"prompt_tokens": 200000, "answer_tokens": 2000, "stagger_s": [0.0, 1.0],
                                 "gates": {"first_token_s": 300, "complete_s": 600, "per_user_tok_s": 15}},
                    "users": [
                        {"start_offset_s": 0.0, "prompt_tokens": 200000, "first_token_s": 240.0,
                         "complete_s": 320.0, "decode_tok_s": 25.0, "score": 1.0,
                         "finish_reason": "stop", "meets_gates": True},
                        {"start_offset_s": 1.0, "prompt_tokens": 200000, "first_token_s": 350.0,
                         "complete_s": 650.0, "decode_tok_s": 25.0, "score": 1.0,
                         "finish_reason": "stop", "meets_gates": False},
                    ],
                    "goodput_users": 1,
                    "correct_users": 2,
                    "solo": {"first_token_s": 70.0, "meets_gates": True},
                    "servers_for_users": 2,
                    "sizing": {"users_per_server": 1, "servers_for_users": 2, "basis": "1 of 2 users met the gates"},
                }
            }
        }
        lines = report_md._sizing(report)
        text = "\n".join(lines)
        self.assertNotIn("while 1 server carried all", text)

    def test_prompt_shrinks_to_fit_context(self):
        w = workload.resolve_workload(None, 131072)
        self.assertEqual(w["prompt_tokens"], 131072 - 8192 - workload.PROMPT_MARGIN_TOKENS)
        self.assertEqual(w["requested_prompt_tokens"], 200000)

    def test_context_must_fit_the_output_and_minimum_prompt(self):
        with self.assertRaisesRegex(ValueError, "context"):
            workload.resolve_workload(None, 4096)
        minimum = 8192 + workload.PROMPT_MARGIN_TOKENS + 1000
        w = workload.resolve_workload(None, minimum)
        self.assertEqual(w["prompt_tokens"] + w["max_output_tokens"] + workload.PROMPT_MARGIN_TOKENS, minimum)

    def test_impossible_context_is_skipped_before_requesting_the_model(self):
        server = mock.MagicMock(max_context=4096)
        result = workload.run_workload(server)
        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("context", result["error"])
        server.call.assert_not_called()

    def test_stagger_is_seeded_and_bounded(self):
        a = workload.stagger_offsets(4, 3, 10, 7)
        self.assertEqual(a, workload.stagger_offsets(4, 3, 10, 7))
        gaps = [b - x for x, b in zip(a, a[1:])]
        self.assertEqual(a[0], 0.0)
        self.assertTrue(all(2.95 <= g <= 10.05 for g in gaps), gaps)


class TestWorkloadPrompt(unittest.TestCase):
    CONTRACTS = [{"id": f"c{i}", "title": f"Contract {i}", "text": "x" * 1000, "clauses": {}} for i in range(10)]

    def test_contains_focus_and_unique_session(self):
        focus = self.CONTRACTS[3]
        p1 = workload.build_prompt(self.CONTRACTS, 5000, 1, focus, "Governing Law: which law?", 1500)
        p2 = workload.build_prompt(self.CONTRACTS, 5000, 1, focus, "Governing Law: which law?", 1500)
        self.assertNotEqual(p1[:60], p2[:60])  # no shared cached prefix between users
        self.assertIn("Contract 3 ===", p1)
        k = int(p1.split("Question about DOCUMENT ")[1].split(" ")[0])
        self.assertIn(f"=== DOCUMENT {k}: Contract 3 ===", p1)
        self.assertIn("about 1500 words", p1)
        self.assertGreaterEqual(p1.count("=== DOCUMENT "), 5)

    def test_set_is_cut_to_the_target_size(self):
        """Whole contracts were added past the target: live prompts ran 4-11%
        over 200K tokens. The last filler contract is now cut."""
        focus = self.CONTRACTS[3]
        p = workload.build_prompt(self.CONTRACTS, 3500, 1, focus, "Q", 100)
        docs = p.split("=== END OF DOCUMENTS ===")[0]
        body = sum(len(chunk.split(" ===\n", 1)[1].rstrip("\n")) for chunk in docs.split("=== DOCUMENT ")[1:])
        self.assertLessEqual(body, 3500 + 40)  # the cut contract's continuation marker
        self.assertIn("x" * 1000, p)  # the focus contract is whole

    def test_fit_prompt_rescales_to_the_server_count(self):
        class Counter:
            calls = 0

            def count_prompt_tokens(self, messages):
                Counter.calls += 1
                return len(messages[0]["content"]) // 2  # 2 characters per token

        contracts = [{"id": f"c{i}", "title": f"C{i}", "text": "word " * 2000, "clauses": {}} for i in range(20)]
        p = workload.fit_prompt(Counter(), contracts, 40000, 10000, 1, contracts[0], "Q", 100)
        self.assertAlmostEqual(len(p) // 2, 10000, delta=100)
        self.assertGreater(Counter.calls, 1)

    def test_fit_prompt_without_a_counter_keeps_the_estimate(self):
        p = workload.fit_prompt(object(), self.CONTRACTS, 5000, 1000, 1, self.CONTRACTS[0], "Q", 100)
        self.assertIn("Contract 0 ===", p)

    def test_split_answer(self):
        self.assertEqual(workload.split_answer("QUOTE: x\nREVIEW: long text"), "QUOTE: x\n")

    def test_sizing_uses_the_solo_user_when_no_concurrent_user_meets_the_gates(self):
        """Live: alone 89 s (gate met), 4 at once 199-353 s (none met). The
        report said 'not estimated' although 1 user per server works."""
        res = {"users": [{"meets_gates": False}] * 4, "goodput_users": 0, "solo": {"meets_gates": True}}
        sz = workload.sizing(res)
        self.assertEqual((sz["users_per_server"], sz["servers_for_users"]), (1, 4))
        self.assertIn("2-3 were not tested", sz["basis"])
        self.assertEqual(workload.sizing(dict(res, solo={"meets_gates": False}))["servers_for_users"], None)
        self.assertEqual(workload.sizing(dict(res, goodput_users=2))["servers_for_users"], 2)
        self.assertEqual(workload.sizing(dict(res, goodput_users=4))["users_per_server"], 4)

    def test_servers_needed(self):
        self.assertEqual(workload.servers_needed(4, 1), 4)
        self.assertEqual(workload.servers_needed(10, 4), 3)
        self.assertIsNone(workload.servers_needed(4, 0))


class _WorkloadServer:
    """Prefill is serial (like a compute-bound GPU): each request's first
    token waits for every earlier prompt. Times are real but short: the
    test's gates are in tens of milliseconds."""

    def __init__(self, prefill_s, decode_tok_s=30.0):
        self.prefill_s, self.decode_tok_s = prefill_s, decode_tok_s
        self.lock = threading.Lock()
        self.max_context = 524288

    def fetch_kv_capacity(self):
        return {"kv_cache_tokens": 1119425, "max_concurrency_at_max_len": 2.1}

    def call(self, messages, max_tokens=None, stream=False, **kw):
        prompt = messages[0]["content"]
        if max_tokens == 1:  # tokenizer calibration
            return {"prompt_tokens": len(prompt) // 4, "completion_tokens": 1, "content": ""}
        t0 = time.perf_counter()
        with self.lock:
            time.sleep(self.prefill_s)
        ttft = time.perf_counter() - t0
        k = int(prompt.split("Question about DOCUMENT ")[1].split(" ")[0])
        doc = prompt.split(f"=== DOCUMENT {k}: ")[1]
        gold = doc.split("[GOLD] ")[1].split("\n")[0]
        return {"content": "QUOTE: " + gold + "\nREVIEW: fine", "ttft": ttft, "answer_ttft": ttft,
                "prompt_tokens": 200000, "cached_tokens": 0, "completion_tokens": 2000,
                "decode_speed": self.decode_tok_s, "finish_reason": "stop",
                "reasoning_chunks": 900, "content_chunks": 1100}


FAST_GATES = {"stagger_s": [0.003, 0.010], "gates": {"first_token_s": 0.09, "complete_s": 0.3, "per_user_tok_s": 15}}


class TestWorkloadRun(unittest.TestCase):
    def _contracts(self):
        cs = []
        for i in range(6):
            gold = f"governed by the laws of state number {i} in all respects and for all purposes"
            cs.append({"id": f"c{i}", "title": f"Contract {i}", "text": f"[GOLD] {gold}\n" + "y" * 100000,
                       "clauses": {"Governing Law": [gold]}, "questions": {"Governing Law": "Governing Law: which?"}})
        return {c["id"]: c for c in cs}, {"source": "test", "license": "CC-BY-4.0"}

    def run_with(self, server):
        with mock.patch.object(suites, "load_contracts", side_effect=self._contracts):
            return workload.run_workload(server, FAST_GATES)

    def test_full_focus_contract_that_exceeds_context_is_not_sent(self):
        gold = "governed by the laws of state number one"
        contract = {"id": "c1", "title": "Long contract", "text": gold + " x" * 30000,
                    "clauses": {"Governing Law": [gold]}, "questions": {"Governing Law": "Which law?"}}
        server = mock.MagicMock(max_context=12000)
        server.call.side_effect = lambda messages, **kwargs: {
            "prompt_tokens": len(messages[0]["content"]) // 3,
            "completion_tokens": 1,
            "content": "",
        }
        server.count_prompt_tokens.side_effect = lambda messages: len(messages[0]["content"]) // 3
        with mock.patch.object(suites, "load_contracts", return_value=({"c1": contract}, {})):
            result = workload.run_workload(server, {"users": 1, "stagger_s": [0, 0]})
        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("context", result["error"])
        self.assertEqual(server.call.call_count, 1)  # tokenizer calibration only
        self.assertEqual(server.call.call_args.kwargs["max_tokens"], 1)

    def test_dataset_without_labelled_clauses_is_skipped(self):
        contract = {"id": "c1", "title": "Contract", "text": "No labels.", "clauses": {}}
        server = mock.MagicMock(max_context=524288)
        with mock.patch.object(suites, "load_contracts", return_value=({"c1": contract}, {})):
            result = workload.run_workload(server, {"users": 1})
        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("no labelled clauses", result["error"])
        server.call.assert_not_called()

    def test_goodput_counts_users_within_gates(self):
        """Prefill takes 40 ms per request and requests queue: early users see
        their first token within the 90 ms gate, later ones wait for every
        prompt ahead of them and miss it."""
        res = self.run_with(_WorkloadServer(prefill_s=0.04))
        firsts = [u["first_token_s"] for u in res["users"]]
        self.assertEqual(firsts, sorted(firsts))
        self.assertTrue(1 <= res["goodput_users"] < 4, firsts)
        self.assertEqual(res["goodput_users"], sum(u["gate_checks"]["first_token"] for u in res["users"]))
        self.assertEqual(res["status"], "PARTIAL")
        self.assertEqual(res["correct_users"], 4)
        self.assertEqual(res["kv_sessions_at_workload"], 1119425 // (200000 + 8192))
        self.assertEqual(len(res["details"]), 5)  # alone + 4 users

    def test_all_users_in_gates_pass(self):
        res = self.run_with(_WorkloadServer(prefill_s=0.005))
        self.assertEqual((res["goodput_users"], res["status"], res["servers_for_users"]), (4, "PASS", 1))

    def test_slow_decode_misses_the_gate(self):
        res = self.run_with(_WorkloadServer(prefill_s=0.005, decode_tok_s=10))
        self.assertEqual(res["goodput_users"], 0)
        self.assertFalse(res["users"][0]["gate_checks"]["decode"])
        self.assertIsNone(res["servers_for_users"])
        self.assertEqual(res["status"], "FAIL")

    def test_fast_but_wrong_is_not_a_pass(self):
        """Seen in an end-to-end run: every user within the gates but every
        clause wrong was reported PASS."""
        server = _WorkloadServer(prefill_s=0.005)
        server_call = server.call

        def wrong(messages, **kw):
            res = server_call(messages, **kw)
            if kw.get("max_tokens") != 1:
                res = dict(res, content="NOT FOUND\nREVIEW: fine")
            return res
        server.call = wrong
        res = self.run_with(server)
        self.assertEqual((res["goodput_users"], res["correct_users"], res["status"]), (4, 0, "PARTIAL"))

    def test_missing_dataset_is_skipped(self):
        with mock.patch.object(suites, "load_contracts", return_value=({}, {})):
            self.assertEqual(workload.run_workload(_WorkloadServer(0.005))["status"], "SKIPPED")


# ----------------------------------------------------------------------------
# usecases.py
# ----------------------------------------------------------------------------
def _scored(acc, lo, hi, pass_all=90.0, items=50, repeats=3, passm=80.0, partial=50.0):
    return {"status": "PASS", "accuracy_pct": acc, "ci95_low_pct": lo, "ci95_high_pct": hi, "items": items,
            "repeats": repeats, "pass_marks": {"pass_pct": passm, "partial_pct": partial, "basis": "t"},
            "reliability": {"pass_all_repeats_pct": pass_all, "inconsistent_items": 2}, "details": []}


class TestUseCaseVerdicts(unittest.TestCase):
    def verdict(self, results, key):
        return next(r for r in usecases.rank_use_cases(results) if r["key"] == key)

    def test_recommended_needs_ci_and_reliability(self):
        res = {"legalbench": _scored(92, 85, 97), "contract_review": _scored(90, 82, 96)}
        self.assertEqual(self.verdict(res, "legal")["verdict"], usecases.RECOMMENDED)
        res["contract_review"] = _scored(90, 82, 96, pass_all=70.0)
        v = self.verdict(res, "legal")
        self.assertEqual(v["verdict"], usecases.REVIEW)
        self.assertIn("every attempt", v["reasons"][0])

    def test_weakest_scored_test_decides(self):
        res = {"legalbench": _scored(95, 90, 99), "contract_review": _scored(40, 30, 50)}
        self.assertEqual(self.verdict(res, "legal")["verdict"], usecases.NOT_RECOMMENDED)

    def test_ci_below_pass_mark_is_review(self):
        res = {"summarisation": _scored(80, 70, 88, passm=75.0)}
        self.assertEqual(self.verdict(res, "summarisation")["verdict"], usecases.REVIEW)

    def test_few_items_is_limited_evidence(self):
        res = {"summarisation": _scored(100, 95, 100, items=20)}
        v = self.verdict(res, "summarisation")
        self.assertEqual((v["verdict"], v["evidence"]), (usecases.REVIEW, "limited"))

    def test_unscored_items_beyond_the_tolerance_cap_a_strong_result_at_review(self):
        # 1 unscored of 50 items (2%) is reported but within the tolerance.
        res = {"legalbench": dict(_scored(92, 85, 97, items=49), unscored_items=1),
               "contract_review": _scored(90, 82, 96)}
        v = self.verdict(res, "legal")
        self.assertEqual(v["verdict"], usecases.RECOMMENDED)
        self.assertTrue(any("legalbench" in reason and "unscored" in reason for reason in v["reasons"]))
        # 4 of 53 (7.5%) is not.
        res["legalbench"] = dict(_scored(92, 85, 97, items=49), unscored_items=4)
        v = self.verdict(res, "legal")
        self.assertEqual((v["verdict"], v["evidence"]), (usecases.REVIEW, "strong"))
        self.assertTrue(any("legalbench" in reason and "unscored" in reason for reason in v["reasons"]))
        res["contract_review"] = _scored(40, 30, 50)
        self.assertEqual(self.verdict(res, "legal")["verdict"], usecases.NOT_RECOMMENDED)

    def test_older_reports_with_unscored_repeats_beyond_the_tolerance_are_not_recommended(self):
        # Older reports have no total_attempts: 50 items x 3 repeats = 150.
        for unscored, expected in ((1, usecases.RECOMMENDED), (10, usecases.REVIEW)):
            details = [{"attempt_scores": [1.0, None, 1.0]}] * unscored
            for key, use_case, extra in (
                ("xstest", "assistant", {"details": details}),
                ("multipl_e", "coding", {"per_language": {"python": {"details": details}}}),
            ):
                with self.subTest(key=key, unscored=unscored):
                    result = dict(_scored(100, 100, 100, pass_all=98.0), **extra)
                    verdict = self.verdict({key: result}, use_case)
                    self.assertEqual(verdict["verdict"], expected)
                    self.assertTrue(any("unscored attempt" in reason for reason in verdict["reasons"]))

    def test_skipped_multipl_e_languages_cannot_be_recommended(self):
        res = {"multipl_e": dict(_scored(95, 90, 99), languages_skipped=["go", "rust"])}
        v = self.verdict(res, "coding")
        self.assertEqual((v["verdict"], v["evidence"]), (usecases.REVIEW, "strong"))
        self.assertTrue(any("multipl_e" in reason and "go" in reason for reason in v["reasons"]))

    def test_older_multipl_e_reports_include_skipped_datasets_in_verdict(self):
        for reported, expected in (([], ["go", "rust"]), (["rust"], ["rust", "go"])):
            with self.subTest(reported=reported):
                result = dict(_scored(100, 100, 100), languages_skipped=reported,
                              per_language={"python": {"status": "PASS"},
                                            "go": {"status": "SKIPPED", "error": "dataset missing"},
                                            "rust": {"status": "SKIPPED", "error": "toolchain missing"}})
                verdict = self.verdict({"multipl_e": result}, "coding")
                self.assertEqual(verdict["verdict"], usecases.REVIEW)
                self.assertEqual(verdict["scored"][0]["languages_skipped"], expected)
                self.assertTrue(any("go" in reason and "rust" in reason for reason in verdict["reasons"]))

    def test_failed_check_caps_at_review(self):
        res = {"multipl_e": _scored(95, 90, 99), "strict_typing": {"status": "FAIL"}}
        v = self.verdict(res, "coding")
        self.assertEqual(v["verdict"], usecases.REVIEW)
        self.assertIn("strict_typing", v["reasons"][-1])

    def test_checks_only(self):
        self.assertEqual(self.verdict({"tool_calling": {"status": "PASS"}}, "agent")["verdict"], usecases.REVIEW)
        self.assertEqual(self.verdict({"tool_calling": {"status": "FAIL"}}, "agent")["verdict"],
                         usecases.NOT_RECOMMENDED)

    def test_infra_failures_and_safety_alone_are_not_measured(self):
        res = {"summarisation": dict(_scored(90, 85, 95), failure_kind="infra"),
               "prompt_injection_defense": {"status": "PASS"}}
        self.assertEqual(self.verdict(res, "summarisation")["verdict"], usecases.NOT_MEASURED)

    def test_ranking_order(self):
        res = {"legalbench": _scored(95, 90, 99), "contract_review": _scored(95, 90, 99),
               "summarisation": _scored(40, 30, 50), "tool_calling": {"status": "PASS"}}
        ranked = usecases.rank_use_cases(res)
        order = [r["key"] for r in ranked]
        self.assertLess(order.index("legal"), order.index("agent"))
        self.assertLess(order.index("agent"), order.index("summarisation"))
        self.assertEqual(ranked[-1]["verdict"], usecases.NOT_MEASURED)

    def test_original_tasks_weigh_more_in_ranking(self):
        """drafting: an original task with a small margin and a public one with
        a large margin; the weighted margin leans to the original task."""
        res = {"writing_tasks": _scored(90, 82, 95), "ifeval_full": _scored(99, 98, 100)}
        r = self.verdict(res, "drafting")
        self.assertAlmostEqual(r["weighted_margin"], (2 * 2 + 18) / 3, places=1)

    def test_workload_answers_count_for_long_documents(self):
        res = {"workload": {"status": "PARTIAL", "details": [{"score": 1.0}, {"score": 1.0}, {"score": 0.0}]}}
        s = self.verdict(res, "long_document")["scored"][0]
        self.assertAlmostEqual(s["accuracy_pct"], 66.7, places=1)
        self.assertEqual(s["items"], 3)

    def test_partial_pass_marks_use_the_benchmark_defaults(self):
        """A record with only pass_pct crashed the verdicts (KeyError)."""
        data = dict(_scored(90, 85, 95), pass_marks={"pass_pct": 80.0})
        s = usecases.scored_summary("writing_tasks", data)
        self.assertEqual((s["pass_pct"], s["partial_pct"]), (80.0, suites.pass_marks("writing_tasks")["partial"]))

    def test_failure_examples_show_the_answer(self):
        data = _scored(70, 60, 80)
        data["details"] = [{"id": "q1", "attempt_scores": [1.0, 0.0],
                            "attempts": [{"outcome": "ok", "response_tail": "a"},
                                         {"outcome": "missed", "response_tail": "the wrong answer"}]}]
        ex = usecases.failure_examples({"summarisation": data}, ["summarisation"])
        self.assertEqual((ex[0]["item"], ex[0]["detail"]["outcome"], ex[0]["answer_tail"]),
                         ("q1", "missed", "the wrong answer"))


# ----------------------------------------------------------------------------
# report_md.py
# ----------------------------------------------------------------------------
class TestReportHelpers(unittest.TestCase):
    def test_paired_difference(self):
        a = [{"id": i, "score": 1.0} for i in range(20)]
        b = [{"id": i, "score": 0.0 if i < 10 else 1.0} for i in range(20)] + [{"id": 99, "score": 0.0}]
        pd = report_md.paired_difference(a, b)
        self.assertEqual((pd["diff_pts"], pd["items"]), (50.0, 20))
        self.assertGreater(pd["ci_low_pts"], 0)
        self.assertIsNone(report_md.paired_difference(a, [{"id": "x", "score": 1}]))

    def test_reference_matches_model_family(self):
        ref = report_md.load_reference(None, {"model": "Qwen3.8-Flash-Next-FP4"})
        self.assertIsNotNone(ref)
        self.assertIn("Qwen3.8-27B", ref["models"])
        self.assertIsNone(report_md.load_reference(None, {"model": "llama-3"}))
        self.assertIsNotNone(report_md.load_reference(
            None, {"model": "local", "server_info": {"weights": "nvidia/Qwen3.8-Flash-Next-FP4-Local"}}))

    def test_judge_agreement(self):
        report = {"results": {"writing_tasks": {"details": [
            {"id": "w1", "attempts": [{"judge": {"c1": True, "c2": False}}]}]}}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"test": "writing_tasks", "item": "w1", "criterion": "c1", "human": True},
                       {"test": "writing_tasks", "item": "w1", "criterion": "c2", "human": True}], f)
        try:
            self.assertEqual(report_md.judge_agreement_text(report, f.name),
                             "The judge agreed with human labels on 1 of 2 criteria (50%).")
        finally:
            os.unlink(f.name)
        self.assertIsNone(report_md.judge_agreement_text(report, None))


class TestReportDocument(unittest.TestCase):
    REPORT = {
        "model": "Qwen3.8-Flash-Next-FP4", "endpoint": "http://srv/v1", "timestamp": "2026-11-07 01:00:00",
        "server_info": {"engine": "vLLM", "version": "0.33.0", "kv_cache_tokens": 1119425},
        "hardware": "1x GPU", "judge": {"model": "judge-m", "self_judged": False},
        "qualification": {"samples": 50, "repeats": 3}, "total_suite_wall_time_s": 3700,
        "results": {
            "legalbench": _scored(92, 85, 97), "contract_review": _scored(70, 60, 80),
            "concurrency": {"status": "PASS", "verified_max_parallel": 4, "peak_prefill_tok_s": 2696,
                            "single_stream_decode_tok_s": 33.9,
                            "levels": [{"clients": 4, "per_stream_decode_tok_s": 21.0, "aggregate_decode_tok_s": 82.4}]},
            "workload": {"status": "PARTIAL", "goodput_users": 1, "servers_for_users": 4,
                         "kv_sessions_at_workload": 5, "aggregate_prefill_tok_s": 2400, "aggregate_decode_tok_s": 70,
                         "workload": dict(workload.DEFAULT_WORKLOAD),
                         "solo": {"start_offset_s": 0, "prompt_tokens": 200000, "first_token_s": 84.0,
                                  "prefill_tok_s": 2381, "decode_tok_s": 30.0, "complete_s": 150, "score": 1.0,
                                  "meets_gates": True, "gate_checks": {}},
                         "users": [{"start_offset_s": 0, "prompt_tokens": 200000, "first_token_s": 85.0 + 80 * i,
                                    "answer_start_s": 140.0 + 80 * i, "complete_s": 200.0 + 80 * i,
                                    "decode_tok_s": 20.0, "score": 1.0, "meets_gates": i == 0,
                                    "gate_checks": {"first_token": i == 0, "complete": True, "decode": True}}
                                   for i in range(4)],
                         "details": [{"score": 1.0}] * 5},
        },
    }

    def test_sections_and_key_lines(self):
        md = report_md.build_report(self.REPORT, reference=report_md.load_reference(None, self.REPORT))
        for heading in ("## Summary", "## Choosing a model", "## Sizing hardware", "## Approving for a team",
                        "## Method and caveats", "## Appendix: every test"):
            self.assertIn(heading, md)
        self.assertIn("1 of 4 users within the gates", md)
        self.assertIn("4 users need about 4 server(s) like this one (1 of 4 users met the gates at once)", md)
        self.assertIn("vLLM 0.33.0", md)
        self.assertIn("Published by the vendor", md)
        self.assertIn("No other run was given", md)
        self.assertIn("missed: first_token", md)
        self.assertIn("| Decode per user (tok/s) | 33.9 | 20.0 | 21.0 |", md)

    def test_comparison_with_an_earlier_run(self):
        older = json.loads(json.dumps(self.REPORT))
        older["timestamp"] = "2026-11-01 01:00:00"
        older["results"]["legalbench"]["details"] = [{"id": i, "score": 0.0} for i in range(5)]
        newer = json.loads(json.dumps(self.REPORT))
        newer["results"]["legalbench"]["details"] = [{"id": i, "score": 1.0} for i in range(5)]
        md = report_md.build_report(newer, [older])
        self.assertIn("### Verdicts side by side", md)
        self.assertIn("(2026-11-01)", md)
        self.assertIn("+100.0", md)

    def test_write_report_next_to_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.json")
            out = report_md.write_report(self.REPORT, path)
            self.assertEqual(out, os.path.join(d, "run.md"))
            self.assertTrue(os.path.exists(out))

    def test_hardware_added_after_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.json")
            with open(path, "w") as f:
                json.dump(dict(self.REPORT, hardware=None), f)
            with mock.patch("builtins.print"):
                report_md.main([path, "--hardware", "Nvidia DGX Spark - 128 GB"])
            with open(path) as f:
                self.assertEqual(json.load(f)["hardware"], "Nvidia DGX Spark - 128 GB")
            with open(os.path.join(d, "run.md")) as f:
                self.assertIn("| Hardware | Nvidia DGX Spark - 128 GB |", f.read())

    def test_incomplete_run_warns(self):
        r = dict(self.REPORT, incomplete={"reason": "server down", "tests_not_run": ["a", "b"]})
        self.assertIn("[!WARNING]", report_md.build_report(r))


class _FixedJudge:
    """Answers each criterion from a function of (task, response)."""

    def __init__(self, decide, configured=True):
        self.decide, self.configured, self.validation = decide, configured, None

    def checklist(self, task, response, criteria, context="", count=True):
        v = self.decide(task, response)
        if v == "objective":
            return {"verdicts": [None] * len(criteria), "judged_by": "objective"}
        return {"verdicts": [v] * len(criteria), "judged_by": "judge"}


class TestJudgeValidation(unittest.TestCase):
    def test_summeval_agreement(self):
        items = [{"summary": "good", "source": "s", "consistent": True},
                 {"summary": "bad", "source": "s", "consistent": False},
                 {"summary": "bad", "source": "s", "consistent": True}]
        j = _FixedJudge(lambda task, resp: resp == "good")
        res = [judge_validation._summeval(j, it) for it in items]
        self.assertEqual(res, [True, True, False])

    def test_llmbar_pairs(self):
        pair = {"instruction": "i", "output_1": "A", "output_2": "B", "better": 1}
        self.assertEqual(judge_validation._llmbar(_FixedJudge(lambda t, r: r == "A"), pair), "separated")
        self.assertEqual(judge_validation._llmbar(_FixedJudge(lambda t, r: r == "B"), pair), "reversed")
        self.assertEqual(judge_validation._llmbar(_FixedJudge(lambda t, r: True), pair), "tie")
        self.assertEqual(judge_validation._llmbar(_FixedJudge(lambda t, r: "objective"), pair), "no verdict")

    def test_summary_and_text(self):
        v = judge_validation.summarise([True, True, False, None], ["separated", "tie", "reversed", "no verdict"])
        self.assertEqual((v["summeval"]["answered"], v["summeval"]["agreement_pct"]), (3, 66.7))
        self.assertEqual((v["llmbar"]["answered"], v["llmbar"]["separated"], v["llmbar"]["separated_pct"]),
                         (3, 1, 33.3))
        text = judge_validation.validation_text(v)
        self.assertIn("2 of 3 summaries (67%", text)
        self.assertIn("1 of 3 pairs (33%", text)
        self.assertIsNone(judge_validation.validation_text(None))

    def test_no_judge_no_check(self):
        self.assertIsNone(judge_validation.run_judge_validation(judge_module.Judge()))

    def test_data_bundled_and_balanced(self):
        summeval, _ = suites.load_items("judge_validation", "summeval")
        llmbar, _ = suites.load_items("judge_validation", "llmbar")
        self.assertEqual((sum(i["consistent"] for i in summeval), sum(not i["consistent"] for i in summeval)), (20, 20))
        self.assertEqual(len(llmbar), 40)

    def test_judged_tests_exist(self):
        import reporting
        self.assertTrue(eval_module.JUDGED_TEST_NUMBERS <= reporting.ALL_TESTS)


class TestNoJudgeVerdicts(unittest.TestCase):
    def test_objective_grading_cannot_be_recommended(self):
        res = {"summarisation": dict(_scored(95, 90, 99), objective_only=True)}
        r = next(x for x in usecases.rank_use_cases(res) if x["key"] == "summarisation")
        self.assertEqual(r["verdict"], usecases.REVIEW)
        self.assertEqual(r["no_judge"], ["summarisation"])
        self.assertIn("graded without a judge", r["reasons"][-1])
        md = report_md.build_report({"model": "m", "results": res, "judge": {"configured": False}})
        self.assertIn("Graded without a judge", md)
        self.assertIn("none: judged tests graded objectively", md)

    def test_validation_shown_for_judged_use_cases(self):
        res = {"summarisation": _scored(95, 90, 99)}
        judge = {"configured": True, "model": "judge-m", "validation": judge_validation.summarise(
            [True] * 9 + [False], ["separated"] * 8 + ["tie"] * 2)}
        md = report_md.build_report({"model": "m", "results": res, "judge": judge})
        self.assertIn("agreed with expert raters on 9 of 10 summaries (90%", md)


class TestSpread(unittest.TestCase):
    def test_low_average_high_against_own_marks(self):
        """SimpleQA at 12% (pass 10%) is above its mark; MedQA at 80%
        (pass 75%) is closer to it: low is MedQA, not SimpleQA."""
        scored = [usecases.scored_summary("simpleqa", _scored(12, 6, 18, passm=10.0, partial=5.0)),
                  usecases.scored_summary("personal_tasks", _scored(90, 85, 95, items=20)),
                  usecases.scored_summary("xstest", _scored(82, 75, 88, passm=90.0, partial=80.0))]
        sp = usecases.spread(scored)
        self.assertEqual((sp["low"]["key"], sp["high"]["key"]), ("xstest", "personal_tasks"))
        self.assertAlmostEqual(sp["average_pct"], (12 * 50 + 90 * 20 + 82 * 50) / 120, places=1)
        self.assertIsNone(usecases.spread([]))

    def test_report_columns(self):
        res = {"legalbench": _scored(92, 85, 97), "contract_review": _scored(70, 60, 80)}
        md = report_md.build_report({"model": "m", "results": res})
        self.assertIn("| # | Use case | Verdict | Evidence | Low | Average | High | Why |", md)
        self.assertIn("Contract Review (CUAD): 70% (pass 80%)", md)
        self.assertIn("| 81% (pass 80%) |", md)  # legal: (92*50 + 70*50) / 100 items, both marks 80%
        self.assertIn("| same | same |", md)  # long documents: one scored test


class TestCatalog(unittest.TestCase):
    def test_new_tests_are_registered(self):
        import reporting
        nums = {num: key for num, key, _ in reporting.TEST_CATALOG}
        self.assertEqual((nums[46], nums[47], nums[48]), ("workload", "contract_review", "summarisation"))
        self.assertEqual(reporting.parse_selected_tests(None, "workload"), {46})
        self.assertIn(47, reporting.parse_selected_tests(None, "legal"))
        self.assertIn(48, reporting.parse_selected_tests(None, "writing"))
        self.assertIn(46, eval_module.ISOLATED_TESTS)
        self.assertIn("contract_review", suites.RUNNERS)
        self.assertIn("summarisation", suites.RUNNERS)

    def test_every_use_case_test_exists(self):
        import reporting
        keys = {key for _n, key, _name in reporting.TEST_CATALOG}
        for uc in usecases.USE_CASES:
            for key in uc["scored"] + uc["checks"]:
                self.assertIn(key, keys, (uc["key"], key))


if __name__ == "__main__":
    unittest.main()


class TestRequestTimeout(unittest.TestCase):
    """A 32,768-token answer without streaming is one socket read; the old
    fixed 600 s timeout would have failed it at ~20 tok/s as an infra error."""

    def test_non_streaming_timeout_covers_the_whole_budget(self):
        import client
        self.assertGreaterEqual(client.request_timeout(32768, stream=False), 32768 / client.MIN_DECODE_TOK_S)
        self.assertEqual(client.request_timeout(1024, stream=False), client.DEFAULT_TIMEOUT_S)
        self.assertEqual(client.request_timeout(32768, stream=True), client.DEFAULT_TIMEOUT_S)

    def test_default_timeout_reaches_urlopen(self):
        import client
        from unittest import mock
        c = client.LLMClient("http://127.0.0.1:9/v1", model="m")
        with mock.patch.object(client.urllib.request, "urlopen", side_effect=OSError("stop")) as op:
            c._call([{"role": "user", "content": "x"}], max_tokens=32768, retries=0)
        self.assertGreaterEqual(op.call_args.kwargs["timeout"], 3276)


class TestOutputBudget(unittest.TestCase):
    """--max-output-tokens / config 'max_output_tokens': one budget for the
    coding and accuracy tests, capped so prompt + answer fit the context."""

    def test_cli_then_config_then_default(self):
        import eval as ev
        import suites
        self.assertEqual(ev.resolve_max_output_tokens(65536, 16384), 65536)
        self.assertEqual(ev.resolve_max_output_tokens(None, 16384), 16384)
        self.assertEqual(ev.resolve_max_output_tokens(None, None), suites.DEFAULT_MAX_OUTPUT_TOKENS)
        for bad in (100, "lots"):
            with self.assertRaises(ValueError):
                ev.resolve_max_output_tokens(bad, None)

    def test_budget_reaches_both_test_families(self):
        import eval as ev
        import suites
        old = (ev.SWE_MAX_TOKENS, suites.QUAL_MAX_TOKENS)
        try:
            ev.set_output_budget(65536)
            self.assertEqual((ev.SWE_MAX_TOKENS, suites.QUAL_MAX_TOKENS), (65536, 65536))
            c = mock.MagicMock()
            c.call.return_value = {"content": "A", "error": None}
            suites.run_benchmark(c, "t", [{"q": 1}], lambda it: [{"role": "user", "content": "q"}],
                                 lambda it, res: (1.0, {}), 1, 1, 1)
            self.assertEqual(c.call.call_args.kwargs["max_tokens"], 65536)
        finally:
            ev.SWE_MAX_TOKENS, suites.QUAL_MAX_TOKENS = old

    def test_budget_is_capped_to_the_context_window(self):
        import suites
        small = mock.MagicMock(max_context=32768)
        msgs = [{"role": "user", "content": "x" * 30000}]  # ~10,000 tokens estimated
        got = suites.output_budget(small, msgs, 32768)
        self.assertLess(got + len(str(msgs)) // 3, 32768)
        self.assertEqual(suites.output_budget(mock.MagicMock(max_context=524288), msgs, 32768), 32768)
        # Less room than MIN_OUTPUT_TOKENS: the prompt is too large for this
        # server. A cut-off answer would be scored as the model's failure.
        tight = [{"role": "user", "content": "x" * (3 * (32768 - 512 - 700))}]
        with self.assertRaises(suites.PromptTooLargeError):
            suites.output_budget(small, tight, 32768)
        fits = [{"role": "user", "content": "x" * (3 * (32768 - 512 - 2000))}]
        got = suites.output_budget(small, fits, 32768)
        self.assertGreaterEqual(got, suites.MIN_OUTPUT_TOKENS)
        self.assertLessEqual(got + len(json.dumps(fits)) // 3, 32768)
        huge = [{"role": "user", "content": "x" * 200000}]
        with self.assertRaises(suites.PromptTooLargeError):
            suites.output_budget(small, huge, 32768)

    def test_budget_never_exceeds_requested_or_remaining_space(self):
        import suites
        messages = [{"role": "user", "content": "x" * 1500}]
        prompt_tokens = len(json.dumps(messages)) // 3
        base = prompt_tokens + suites.CONTEXT_HEADROOM_TOKENS
        client = mock.MagicMock(max_context=base + 2000)
        self.assertEqual(suites.output_budget(client, messages, 4096), 2000)
        self.assertEqual(suites.output_budget(client, messages, 128), 128)
        # A small requested budget only needs its own size, not the minimum.
        client.max_context = base + 128
        self.assertEqual(suites.output_budget(client, messages, 128), 128)
        client.max_context = base + 127
        with self.assertRaises(suites.PromptTooLargeError):
            suites.output_budget(client, messages, 128)
        client.max_context = base + suites.MIN_OUTPUT_TOKENS - 1
        with self.assertRaises(suites.PromptTooLargeError):
            suites.output_budget(client, messages, 4096)

    def test_retry_that_outgrows_the_context_keeps_the_graded_attempt(self):
        import eval as ev
        import suites
        prompt = "x" * 300
        first = len(json.dumps([{"role": "user", "content": prompt}])) // 3
        c = mock.MagicMock(max_context=first + suites.CONTEXT_HEADROOM_TOKENS + 4096)
        c.call.return_value = {"content": "y" * 12000, "error": None, "finish_reason": "stop"}
        grade = lambda text, res: {"status": "FAIL", "feedback": "SyntaxError: bad"}  # noqa: E731
        with mock.patch.object(ev, "log"):
            out = ev.run_with_retries(c, prompt, grade, 4096, "Fix it.")
        self.assertEqual(c.call.call_count, 1)
        self.assertEqual(out["grade"]["status"], "FAIL")
        self.assertEqual(out["fields"]["score"], 0.0)

    def test_oversized_qualification_items_are_unscored_without_model_calls(self):
        import suites
        client = mock.MagicMock(max_context=2048)
        items = [{"id": "short", "content": "short"}, {"id": "long", "content": "x" * 6000}]
        client.call.return_value = {"content": "answer", "completion_tokens": 1}
        grader = mock.Mock(return_value=(1.0, {}))
        with mock.patch.object(suites, "log"):
            result = suites.run_benchmark(
                client, "test", items, lambda item: [{"role": "user", "content": item["content"]}],
                grader, samples=2, repeats=1, workers=1, max_tokens=128)
        self.assertEqual(client.call.call_count, 1)
        self.assertEqual(grader.call_count, 1)
        self.assertEqual(result["accuracy_pct"], 100.0)
        self.assertEqual(result["unscored_items"], 1)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["details"][1]["score"])
        self.assertIn("context", result["details"][1]["attempts"][0]["error"].lower())

    def test_simpleqa_summary_excludes_oversized_items(self):
        client = mock.MagicMock(max_context=2048)
        client.call.return_value = {"content": "Answer: Paris", "completion_tokens": 2}
        judge = mock.MagicMock()
        items = [{"id": "short", "question": "Capital of France?", "answer": "Paris"},
                 {"id": "long", "question": "x" * 6000, "answer": "Paris"}]
        with mock.patch.object(suites, "load_items", return_value=(items, {})):
            result = suites.run_simpleqa(client, judge, samples=2, repeats=1, workers=1)
        self.assertEqual(client.call.call_count, 1)
        self.assertEqual(result["unscored_items"], 1)
        self.assertEqual(result["correct_pct"], 100.0)
        self.assertEqual(result["not_attempted_pct"], 0.0)

    def test_ifeval_summary_excludes_oversized_items(self):
        client = mock.MagicMock(max_context=2048)
        client.call.return_value = {"content": "Paris", "completion_tokens": 1}
        item = {"instruction_ids": ["punctuation:no_comma"], "kwargs": [{}]}
        items = [dict(item, id="short", prompt="Say Paris"),
                 dict(item, id="long", prompt="x" * 6000)]
        with mock.patch.object(suites, "load_items", return_value=(items, {})):
            result = suites.run_ifeval_full(client, None, samples=2, repeats=1, workers=1)
        self.assertEqual(client.call.call_count, 1)
        self.assertEqual(result["unscored_items"], 1)
        self.assertEqual(result["prompt_strict_pct"], 100.0)
        self.assertEqual(result["instruction_strict_pct"], 100.0)

    def test_unscored_judge_errors_remain_infrastructure_failures(self):
        client = mock.MagicMock(max_context=2048)
        client.call.return_value = {"content": "answer", "completion_tokens": 1}
        with mock.patch.object(suites, "log"):
            result = suites.run_benchmark(
                client, "test", [{"id": "x"}], lambda _: [{"role": "user", "content": "short"}],
                lambda *_: (None, {"judge_error": "timeout"}), samples=1, repeats=1, workers=1)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["failure_kind"], "infra")
        self.assertIn("judge: timeout", result["error"])

    def test_multipl_e_aggregates_scored_and_oversized_items(self):
        client = mock.MagicMock(max_context=2048)
        client.call.return_value = {"content": "answer", "completion_tokens": 1}
        items = [{"id": "short", "prompt": "short"}, {"id": "long", "prompt": "x" * 6000}]
        with mock.patch.object(suites, "MULTIPL_E_LANGS", ["python"]), \
             mock.patch.object(suites, "load_items", return_value=(items, {})), \
             mock.patch.object(suites, "multipl_e_messages",
                               side_effect=lambda item: [{"role": "user", "content": item["prompt"]}]), \
             mock.patch.object(suites, "grade_multipl_e", return_value=(1.0, {"stage": "pass"})):
            result = suites.run_multipl_e(client, None, samples=2, repeats=1, workers=1)
        self.assertEqual(client.call.call_count, 1)
        self.assertEqual((result["items"], result["unscored_items"], result["accuracy_pct"]), (1, 1, 100.0))
        self.assertEqual(result["status"], "PARTIAL")

    def test_multipl_e_propagates_unscored_repeats(self):
        client = mock.MagicMock(max_context=2048)
        client.call.return_value = {"content": "answer", "completion_tokens": 1}
        items = [{"id": "short", "prompt": "short"}]
        with mock.patch.object(suites, "MULTIPL_E_LANGS", ["python"]), \
             mock.patch.object(suites, "load_items", return_value=(items, {})), \
             mock.patch.object(suites, "multipl_e_messages",
                               side_effect=lambda item: [{"role": "user", "content": item["prompt"]}]), \
             mock.patch.object(suites, "grade_multipl_e", side_effect=[(1.0, {}), (None, {})]):
            result = suites.run_multipl_e(client, None, samples=1, repeats=2, workers=1)
        self.assertEqual((result["items"], result["unscored_items"], result["unscored_attempts"]), (1, 0, 1))
        self.assertEqual(result["status"], "PARTIAL")

    def test_multipl_e_missing_language_dataset_is_not_a_pass(self):
        client = mock.MagicMock(max_context=2048)
        client.call.return_value = {"content": "answer", "completion_tokens": 1}
        items = [{"id": str(i), "prompt": "short"} for i in range(40)]
        with mock.patch.object(suites, "MULTIPL_E_LANGS", ["python", "go"]), \
             mock.patch("polyglot.toolchain_available", return_value=True), \
             mock.patch.object(suites, "load_items",
                               side_effect=lambda _, name: (items if name.endswith("_python") else [], {})), \
             mock.patch.object(suites, "multipl_e_messages",
                               side_effect=lambda item: [{"role": "user", "content": item["prompt"]}]), \
             mock.patch.object(suites, "grade_multipl_e", return_value=(1.0, {})):
            result = suites.run_multipl_e(client, None, samples=40, repeats=1, workers=1)
        self.assertEqual(result["per_language"]["go"]["status"], "SKIPPED")
        self.assertEqual(result["languages_skipped"], ["go"])
        self.assertEqual(result["status"], "PARTIAL")
        verdict = next(v for v in usecases.rank_use_cases({"multipl_e": result}) if v["key"] == "coding")
        self.assertEqual(verdict["verdict"], usecases.REVIEW)
        self.assertTrue(any("go" in reason and "unscored languages" in reason for reason in verdict["reasons"]))

    def test_multipl_e_all_oversized_reports_the_context_reason(self):
        client = mock.MagicMock(max_context=2048)
        items = [{"id": "long", "prompt": "x" * 6000}]
        with mock.patch.object(suites, "MULTIPL_E_LANGS", ["python"]), \
             mock.patch.object(suites, "load_items", return_value=(items, {})), \
             mock.patch.object(suites, "multipl_e_messages",
                               side_effect=lambda item: [{"role": "user", "content": item["prompt"]}]):
            result = suites.run_multipl_e(client, None, samples=1, repeats=2, workers=1)
        client.call.assert_not_called()
        self.assertEqual(result["status"], "SKIPPED")
        self.assertEqual(result["unscored_attempts"], 2)
        self.assertIn("context", result["error"].lower())

    def test_multipl_e_compiler_infrastructure_failure_is_not_dataset_skip(self):
        client = mock.MagicMock(max_context=2048)
        client.call.return_value = {"content": "answer", "completion_tokens": 1}
        items = [{"id": "short", "prompt": "short"}]
        with mock.patch.object(suites, "MULTIPL_E_LANGS", ["python"]), \
             mock.patch.object(suites, "load_items", return_value=(items, {})), \
             mock.patch.object(suites, "multipl_e_messages",
                               side_effect=lambda item: [{"role": "user", "content": item["prompt"]}]), \
             mock.patch.object(suites, "grade_multipl_e",
                               return_value=(None, {"stage": "infra", "detail": "compiler timed out"})):
            result = suites.run_multipl_e(client, None, samples=1, repeats=1, workers=1)
        self.assertEqual((result["status"], result["failure_kind"]), ("FAIL", "infra"))
        self.assertIn("compiler timed out", result["error"])

    def test_core_test_skips_when_its_prompt_cannot_fit(self):
        import eval as ev
        client = mock.MagicMock(max_context=300)
        with mock.patch.object(ev, "log"):
            result = ev.run_test_code_review_security(client)
        client.call.assert_not_called()
        self.assertEqual(result["status"], "SKIPPED")
        self.assertIn("context", result["error"].lower())

    def test_report_states_the_budget(self):
        import report_md
        self.assertIn("32,768 tokens", report_md._budget_text({"max_output_tokens": 32768}))
        self.assertIn("8,192", report_md._budget_text({}))
