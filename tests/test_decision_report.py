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

    def test_prompt_shrinks_to_fit_context(self):
        w = workload.resolve_workload(None, 131072)
        self.assertEqual(w["prompt_tokens"], 131072 - 8192 - workload.PROMPT_MARGIN_TOKENS)
        self.assertEqual(w["requested_prompt_tokens"], 200000)

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
        # Less room than MIN_OUTPUT_TOKENS: ask for what fits, never more than
        # the window (the old 1,024 floor overflowed it and the server
        # rejected the request).
        tight = [{"role": "user", "content": "x" * (3 * (32768 - 512 - 700))}]
        got = suites.output_budget(small, tight, 32768)
        self.assertLess(got, suites.MIN_OUTPUT_TOKENS)
        self.assertLessEqual(got + len(str(tight)) // 3, 32768)
        huge = [{"role": "user", "content": "x" * 200000}]
        self.assertEqual(suites.output_budget(small, huge, 32768), 256)

    def test_report_states_the_budget(self):
        import report_md
        self.assertIn("32,768 tokens", report_md._budget_text({"max_output_tokens": 32768}))
        self.assertIn("8,192", report_md._budget_text({}))
