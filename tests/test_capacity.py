#!/usr/bin/env python3
"""Tests for server-capacity detection and measurement (slot detection,
Test 3 concurrency, Test 12 context scaling)."""

import os
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import eval as eval_module
import reporting
from client import LLMClient


def _client_with(responses, endpoint="http://srv:8888/v1"):
    """An LLMClient whose GETs return `responses[url]` (None = unreachable)."""
    client = LLMClient(endpoint)
    patcher = mock.patch.object(client, "_get_json", side_effect=lambda url, timeout=5: responses.get(url))
    patcher.start()
    return client, patcher


class TestSlotDetection(unittest.TestCase):
    def detect(self, responses):
        client, patcher = _client_with(responses)
        try:
            return client.detect_server_slots()
        finally:
            patcher.stop()

    def test_health_parallel(self):
        """The local exl3 server reports {"parallel": 2} on /health."""
        self.assertEqual(self.detect({"http://srv:8888/health": {"ok": True, "parallel": 2}}),
                         (2, "/health parallel"))

    def test_health_max_slots_as_string(self):
        self.assertEqual(self.detect({"http://srv:8888/health": {"max_slots": "4"}}), (4, "/health max_slots"))

    def test_llamacpp_props(self):
        """llama.cpp's /health has no slot count; /props has total_slots."""
        self.assertEqual(self.detect({"http://srv:8888/health": {"status": "ok"},
                                      "http://srv:8888/props": {"total_slots": 8}}),
                         (8, "/props total_slots"))

    def test_llamacpp_slots_list(self):
        self.assertEqual(self.detect({"http://srv:8888/slots": [{"id": 0}, {"id": 1}, {"id": 2}]}), (3, "/slots"))

    def test_nonsense_values_are_ignored(self):
        """`true`, 0 and negative counts must not read as a slot count."""
        responses = {"http://srv:8888/health": {"parallel": True, "max_slots": 0},
                     "http://srv:8888/props": {"total_slots": -1}}
        self.assertEqual(self.detect(responses), (None, "not reported"))

    def test_not_reported(self):
        """vLLM-style servers expose no slot count."""
        self.assertEqual(self.detect({}), (None, "not reported"))

    def test_endpoints_are_off_the_api_root(self):
        """/v1 is the API prefix; /health, /props and /slots live at the root."""
        client = LLMClient("http://srv:8888/v1")
        self.assertEqual((client.root_url, client.health_url), ("http://srv:8888", "http://srv:8888/health"))
        self.assertEqual(LLMClient("http://srv:8888").health_url, "http://srv:8888/health")


class TestParallelPrecedence(unittest.TestCase):
    def test_cli_wins(self):
        self.assertEqual(eval_module.resolve_parallel(3, 6, 2, "/health parallel"), (3, "--parallel"))

    def test_config_beats_server(self):
        self.assertEqual(eval_module.resolve_parallel(None, "6", 2, "/health parallel"), (6, "config.json"))

    def test_server_reported(self):
        self.assertEqual(eval_module.resolve_parallel(None, None, 2, "/health parallel"),
                         (2, "server /health parallel"))

    def test_nothing_decides(self):
        self.assertEqual(eval_module.resolve_parallel(None, None, None, "not reported"), (None, None))


def _stream(t_first, t_finish, tokens, error=None):
    """A stream that produced `tokens` evenly between t_first and t_finish."""
    step = (t_finish - t_first) / tokens
    return {"t_first": None if error else t_first, "t_finish": t_finish, "completion_tokens": 0 if error else tokens,
            "ttft_s": t_first, "decode_speed": tokens / (t_finish - t_first), "error": error,
            "chunk_times": [] if error else [t_first + i * step for i in range(tokens)]}


class TestConcurrencyLevels(unittest.TestCase):
    def test_plans(self):
        self.assertEqual(eval_module.concurrency_levels(2), [1, 2, 3, 4])
        self.assertEqual(eval_module.concurrency_levels(4), list(range(1, 9)))
        self.assertEqual(eval_module.concurrency_levels(1), [1, 2])
        self.assertEqual(eval_module.concurrency_levels(16), list(range(1, 9)) + [16, 32])
        self.assertEqual(eval_module.concurrency_levels(40), list(range(1, 9)) + [16, 32, 40, 64])
        self.assertEqual(eval_module.concurrency_levels(None), [1, 2, 4, 8, 16, 32])


class TestAnalyzeLevel(unittest.TestCase):
    def test_truly_parallel_streams(self):
        lv = eval_module.analyze_level(2, [_stream(1.0, 11.0, 100), _stream(1.1, 11.1, 100)])
        self.assertEqual(lv["queued"], 0)
        self.assertTrue(lv["all_streams_overlapped"])
        self.assertAlmostEqual(lv["aggregate_decode_tok_s"], 20.0, delta=0.5)

    def test_one_at_a_time_is_queued_and_not_double_counted(self):
        """The old formula summed per-stream speeds (200 tok/s here) although
        the server served the two requests one after the other."""
        lv = eval_module.analyze_level(2, [_stream(1.0, 2.0, 100), _stream(2.5, 3.5, 100)])
        self.assertEqual(lv["queued"], 1)
        self.assertFalse(lv["all_streams_overlapped"])
        self.assertAlmostEqual(lv["aggregate_decode_tok_s"], 80.0, delta=0.5)

    def test_only_overlapping_tokens_count(self):
        """A alone for 5 s, then A and B together for 5 s at 10 tok/s each."""
        lv = eval_module.analyze_level(2, [_stream(1.0, 11.0, 100), _stream(6.0, 16.0, 100)])
        self.assertEqual(lv["window_s"], 5.0)
        self.assertAlmostEqual(lv["aggregate_decode_tok_s"], 20.0, delta=0.5)

    def test_errors_are_counted(self):
        lv = eval_module.analyze_level(2, [_stream(1.0, 2.0, 10), _stream(1.0, 1.5, 10, error="HTTPError 503")])
        self.assertEqual((lv["served"], len(lv["errors"])), (1, 1))


def _level(clients, agg, queued=0, errors=()):
    return {"clients": clients, "served": clients - len(errors), "errors": list(errors), "queued": queued,
            "aggregate_decode_tok_s": agg}


class TestCapacityVerdict(unittest.TestCase):
    def test_advertised_slots_verified(self):
        v = eval_module.capacity_verdict([_level(1, 40), _level(2, 75), _level(3, 76, queued=1)], 2, True)
        self.assertEqual((v["verified_max_parallel"], v["status"]), (2, "PASS"))

    def test_best_level_is_the_fewest_clients_at_peak(self):
        """Seen live: 80.3 tok/s at 4 clients, 80.4 at 8 (4 of them queued).
        The best level is 4; 8 only adds waiting."""
        v = eval_module.capacity_verdict([_level(1, 35.2), _level(2, 54.4), _level(4, 80.3),
                                          _level(8, 80.4, queued=4), _level(16, 79.2, queued=12)], None, True)
        self.assertEqual((v["best_throughput_level"], v["peak_aggregate_decode_tok_s"]), (4, 80.4))
        self.assertEqual(v["verified_max_parallel"], 4)
        self.assertFalse(v["verified_is_lower_bound"])

    def test_fewer_slots_than_advertised(self):
        v = eval_module.capacity_verdict([_level(1, 40), _level(2, 75), _level(3, 76, queued=1)], 4, True)
        self.assertEqual((v["verified_max_parallel"], v["status"]), (2, "PARTIAL"))

    def test_served_one_at_a_time(self):
        v = eval_module.capacity_verdict([_level(1, 40), _level(2, 41, queued=1)], 2, True)
        self.assertEqual((v["verified_max_parallel"], v["status"]), (1, "FAIL"))

    def test_concurrent_but_no_throughput_gain(self):
        """Admitted together but time-sliced: two streams at half speed each."""
        v = eval_module.capacity_verdict([_level(1, 40), _level(2, 41)], 2, False)
        self.assertEqual(v["status"], "FAIL")
        self.assertIn("no throughput gain", v["note"])

    def test_unknown_slots_reports_measurement(self):
        v = eval_module.capacity_verdict([_level(1, 40), _level(2, 75), _level(4, 140)], None, False)
        self.assertEqual((v["verified_max_parallel"], v["status"], v["verified_is_lower_bound"]), (4, "PASS", True))

    def test_single_request_failure(self):
        v = eval_module.capacity_verdict([_level(1, 0, errors=["HTTPError 500"])], 2, True)
        self.assertEqual((v["verified_max_parallel"], v["status"]), (0, "FAIL"))


class _TwoSlotServer:
    """Serves at most 2 requests at once; later requests wait for a slot."""
    max_context = None

    def __init__(self):
        self.slots = threading.Semaphore(2)

    def call(self, messages, max_tokens=None, temperature=None, stream=False, chunk_callback=None, **kw):
        t0 = time.perf_counter()
        if chunk_callback is None:  # prefill probe
            return {"prompt_tokens": 2000, "cached_tokens": 0, "ttft": 0.01, "completion_tokens": 1}
        with self.slots:
            time.sleep(0.02)
            t_first = time.perf_counter()
            for i in range(20):
                time.sleep(0.005)
                chunk_callback("x", t_first, time.perf_counter(), i + 1)
        return {"prompt_tokens": 500, "completion_tokens": 20, "ttft": t_first - t0,
                "decode_speed": 20 / (time.perf_counter() - t_first)}


class TestConcurrencyEndToEnd(unittest.TestCase):
    def test_two_slot_server_is_verified_at_two(self):
        res = eval_module.run_test_concurrency(_TwoSlotServer(), 2, "server /health parallel", quick=True)
        self.assertEqual(res["verified_max_parallel"], 2)
        self.assertEqual(res["status"], "PASS")
        self.assertGreater(res["levels"][2]["queued"], 0)   # 3 clients: one waits
        self.assertAlmostEqual(res["peak_prefill_tok_s"], 200000.0)

    def test_server_advertising_four_is_partial(self):
        res = eval_module.run_test_concurrency(_TwoSlotServer(), 4, "server /health parallel", quick=True)
        self.assertEqual((res["verified_max_parallel"], res["status"]), (2, "PARTIAL"))

    def test_report_reads_the_result(self):
        res = eval_module.run_test_concurrency(_TwoSlotServer(), 2, "server /health parallel", quick=True)
        m = reporting.extract_test_metrics("concurrency", res)
        self.assertEqual((m["status"], m["score"]), ("PASS", 1.0))
        self.assertIn("2 parallel", m["desc"])
        self.assertEqual(m["ttft_ms"], res["levels"][0]["mean_ttft_ms"])  # single request, no waiting


class TestQueueCounters(unittest.TestCase):
    VLLM = (
        "# HELP vllm:num_requests_running x\n"
        'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
        'vllm:num_requests_running{engine="1",model_name="m"} 1.0\n'
        'vllm:num_requests_waiting{engine="0",model_name="m"} 2.0\n'
        'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 2.0\n'
        'vllm:num_preemptions_total{engine="0",model_name="m"} 7.0\n'
        'vllm:cache_config_info{block_size="3456",kv_cache_max_concurrency="2.135135",'
        'kv_cache_size_tokens="1119425",num_gpu_blocks="395"} 1.0\n')

    def test_vllm_sums_engines(self):
        import client
        self.assertEqual(client.parse_queue_metrics(self.VLLM), {"running": 4, "waiting": 2, "preemptions": 7})

    def test_vllm_kv_capacity(self):
        import client
        self.assertEqual(client.parse_kv_capacity(self.VLLM),
                         {"kv_cache_tokens": 1119425, "max_concurrency_at_max_len": 2.14})

    def test_sglang_and_absent(self):
        import client
        self.assertEqual(client.parse_queue_metrics("sglang:num_running_reqs 2\nsglang:num_queue_reqs 5\n"),
                         {"running": 2, "waiting": 5, "preemptions": None})
        self.assertIsNone(client.parse_queue_metrics("process_resident_memory_bytes 1e9\n"))

    def test_exl3_health_jobs(self):
        """The local exl3 server has no /metrics but reports job counts on /health."""
        c, patcher = _client_with({"http://srv:8888/health": {"gpu_active_jobs": 2, "gpu_pending_jobs": 1}})
        try:
            with mock.patch.object(c, "_get_text", return_value=None):
                self.assertEqual(c.fetch_queue_state(),
                                 {"running": 2, "waiting": 1, "preemptions": None, "source": "/health jobs"})
        finally:
            patcher.stop()

    def test_sampler_disabled_without_counters(self):
        with eval_module.QueueSampler(mock.MagicMock(), interval=0.001) as sampler:
            time.sleep(0.01)
        self.assertIsNone(sampler.result())


class TestContextSteps(unittest.TestCase):
    def test_final_step_is_98_percent(self):
        self.assertEqual(eval_module.context_scaling_steps(262144),
                         [4000, 8000, 16000, 32000, 64000, 128000, 256900])
        self.assertEqual(eval_module.context_scaling_steps(524288)[-2:], [256000, 513800])

    def test_small_context_keeps_room_for_the_answer(self):
        """98% of 8k leaves too little room for a 2,048-token answer."""
        self.assertEqual(eval_module.context_scaling_steps(8192), [4000, 5700])
        self.assertEqual(eval_module.context_scaling_steps(4096), [1600])


class TestContextErrors(unittest.TestCase):
    def test_classification(self):
        c = eval_module.classify_context_error
        self.assertEqual(c("HTTPError 400: Bad Request - This model's maximum context length is 524288 tokens"),
                         "context_limit_rejected")
        self.assertEqual(c("HTTPError 413: Payload Too Large - "), "context_limit_rejected")
        self.assertEqual(c("The read operation timed out"), "timeout")
        self.assertEqual(c("HTTPError 500: Internal Server Error - CUDA out of memory"), "server_error_or_oom")
        self.assertEqual(c("HTTPError 502: Bad Gateway - no room in queue"), "server_error_or_oom")
        self.assertEqual(c("HTTPError 400: Bad Request - zoom level"), "request_error")


class TestNeedleGrading(unittest.TestCase):
    def test_final_answer_only(self):
        """The value in the reasoning alone does not count (it did before)."""
        g = eval_module._grade_needle
        self.assertEqual(g({"content": "K = VAL_AB12", "reasoning": ""}, "AB12"), "recalled")
        self.assertEqual(g({"content": "I could not find it.", "reasoning": "K = VAL_AB12"}, "AB12"), "missed")
        self.assertEqual(g({"content": "", "reasoning": "thinking...", "finish_reason": "length"}, "AB12"), "truncated")
        self.assertEqual(g({"error": "HTTPError 400"}, "AB12"), "error")


class _NeedleServer:
    """Counts 4 characters per token and answers with the hidden value.
    `kv_budget`: prompt tokens the KV cache holds at once (later requests
    wait). `serial_prefill`: prompts are read one at a time, like a
    compute-bound server. `counters`: expose queue counters like vLLM."""

    def __init__(self, max_context=8000, kv_budget=10**9, fail_at=None, error_at=None, first_error=None,
                 serial_prefill=False, counters=True, kv_tokens=None):
        self.max_context = max_context
        self.kv_budget = kv_budget
        self.fail_at, self.error_at, self.first_error = fail_at, error_at, first_error
        self.serial_prefill, self.counters, self.kv_tokens = serial_prefill, counters, kv_tokens
        self.in_flight = 0
        self.running = 0
        self.waiting = 0
        self.cond = threading.Condition()
        self.prefill_lock = threading.Lock()
        self.calls = []

    def fetch_queue_state(self):
        if not self.counters:
            return None
        with self.cond:
            return {"running": self.running, "waiting": self.waiting, "preemptions": 0, "source": "/metrics"}

    def fetch_kv_capacity(self):
        return {"kv_cache_tokens": self.kv_tokens, "max_concurrency_at_max_len": 1.5} if self.kv_tokens else None

    def call(self, messages, max_tokens=None, chunk_callback=None, **kw):
        import re
        prompt = messages[0]["content"]
        tokens = len(prompt) // 4
        self.calls.append(tokens)
        target = int(re.search(r"Target: (\d+)", prompt).group(1))
        if self.first_error and len(self.calls) == 1:
            return {"error": self.first_error, "failure_kind": "infra"}
        if self.error_at == target:
            return {"error": "HTTPError 400: Bad Request - maximum context length exceeded", "failure_kind": "client"}
        key, secret = re.search(r"(NEEDLE_\w+) = VAL_(\w+)", prompt).groups()
        t0 = time.perf_counter()
        with self.cond:
            self.waiting += 1
            while self.in_flight and self.in_flight + tokens > self.kv_budget:
                self.cond.wait()
            self.waiting -= 1
            self.in_flight += tokens
            self.running += 1
        try:
            if self.serial_prefill:
                with self.prefill_lock:
                    time.sleep(0.05)
            else:
                time.sleep(0.05)
            t_first = time.perf_counter()
            for i in range(5):
                time.sleep(0.002)
                if chunk_callback:
                    chunk_callback("x", t_first, time.perf_counter(), i + 1)
        finally:
            with self.cond:
                self.in_flight -= tokens
                self.running -= 1
                self.cond.notify_all()
        answer = "Summary." if self.fail_at == target else f"Summary.\n{key} = VAL_{secret}"
        return {"content": answer, "reasoning": "", "prompt_tokens": tokens, "cached_tokens": 0,
                "ttft": t_first - t0, "completion_tokens": 5, "decode_speed": 50.0, "finish_reason": "stop"}


class TestCapacityVerdictText(unittest.TestCase):
    def test_live_case_wording(self):
        """Live on the test server: 4 x 513,800 tokens did not fit (2 preempted),
        4 x 128,400 did. The KV cache holds 2 full-context requests -- the
        verdict must not say "one at a time"."""
        full = {"prompt_tokens_each": 513800, "concurrent": False, "error_kinds": []}
        split = {"prompt_tokens_each": 128400, "concurrent": True, "error_kinds": []}
        v = eval_module.context_capacity_verdict(4, full, split, {"kv_cache_tokens": 1119425})
        self.assertEqual(v, "4 requests of 513,800 tokens do not fit at once; the KV cache (1,119,425 tokens) "
                            "holds 2 at this size; 4 requests of 128,400 tokens run at the same time")


class TestContextScalingRun(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(eval_module, "QUEUE_SAMPLE_INTERVAL", 0.002)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run12(self, server, steps, slots=None, capacity=True):
        return eval_module.run_test_context_scaling(server, steps, capacity_slots=slots, run_capacity=capacity)

    def test_calibration_hits_the_target(self):
        """The first step learns tokens per filler block from the server, so the
        large steps land within 1% of their targets on any tokenizer."""
        server = _NeedleServer(max_context=70000)
        res = self.run12(server, [4000, 16000, 64000], capacity=False)
        self.assertEqual(res["status"], "PASS")
        for step in res["steps"][1:]:
            self.assertLess(abs(step["actual_prompt_tokens"] - step["target_tokens"]) / step["target_tokens"], 0.01)

    def test_stops_at_the_first_failed_step(self):
        server = _NeedleServer(fail_at=8000)
        res = self.run12(server, [4000, 8000, 16000, 32000])
        self.assertEqual(len(server.calls), 2)
        self.assertEqual([s["status"] for s in res["steps"]], ["PASS", "FAIL", "SKIPPED", "SKIPPED"])
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(res["capacity"]["skipped"])

    def test_final_step_refused_is_partial(self):
        res = self.run12(_NeedleServer(error_at=16000), [4000, 16000], slots=2)
        self.assertEqual(res["status"], "PARTIAL")
        self.assertIn("configured context not usable", res["note"])
        self.assertEqual(res["steps"][-1]["error_kind"], "context_limit_rejected")
        self.assertFalse(res["max_context_verified"])

    def test_capacity_all_slots_at_full_context(self):
        res = self.run12(_NeedleServer(), [4000, 8000], slots=2)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["capacity"]["full_context"]["concurrent"])
        self.assertTrue(res["capacity"]["verdict"].startswith("yes: 2 requests of 8,000 tokens"))
        self.assertIsNone(res["capacity"]["split_context"])

    def test_capacity_full_context_one_at_a_time(self):
        """Shared KV cache that the server does not report: two full-context
        prompts are sent and do not run together; two half-size prompts do.
        The status stays PASS (report-only)."""
        res = self.run12(_NeedleServer(kv_budget=12000), [4000, 8000], slots=2)
        cap = res["capacity"]
        self.assertEqual(res["status"], "PASS")
        self.assertFalse(cap["full_context"]["concurrent"])
        self.assertNotIn("not_sent", cap["full_context"])
        self.assertEqual(cap["full_context"]["server_queue"]["max_running"], 1)
        self.assertTrue(cap["split_context"]["concurrent"])
        self.assertIn("2 requests of 8,000 tokens do not fit at once", cap["verdict"])

    def test_capacity_not_sent_when_kv_size_rules_it_out(self):
        """Live: 4 x 513,800 tokens took 16 min to show what the reported KV
        size (1,119,425 tokens, 2 at that size) already said. Only the
        context / slots attempt is sent."""
        server = _NeedleServer(kv_budget=12000, kv_tokens=12000)
        res = self.run12(server, [4000, 8000], slots=2)
        cap = res["capacity"]
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(cap["full_context"]["not_sent"])
        self.assertFalse(cap["full_context"]["concurrent"])
        sent = server.calls[2:]  # after the 2 steps: only the split pair, no 8,000-token requests
        self.assertEqual(len(sent), 2)
        self.assertTrue(all(t < 6000 for t in sent), sent)
        self.assertTrue(cap["split_context"]["concurrent"])
        self.assertEqual(cap["verdict"], "2 requests of 8,000 tokens do not fit at once; the KV cache "
                                         "(12,000 tokens) holds 1 at this size (not sent); "
                                         "2 requests of 4,000 tokens run at the same time")

    def test_capacity_sent_when_kv_size_allows_it(self):
        server = _NeedleServer(kv_tokens=10**6)
        res = self.run12(server, [4000, 8000], slots=2)
        self.assertNotIn("not_sent", res["capacity"]["full_context"])
        self.assertTrue(res["capacity"]["full_context"]["concurrent"])
        self.assertEqual(len(server.calls), 4)  # 2 steps + 2 full-context requests

    def test_no_split_attempt_when_the_workload_test_runs(self):
        """With Test 46 selected, the context / slots attempt is not sent:
        Test 46 measures concurrent users at the workload size."""
        server = _NeedleServer(kv_budget=12000)
        res = eval_module.run_test_context_scaling(server, [4000, 8000], capacity_slots=2, capacity_split=False)
        self.assertIsNone(res["capacity"]["split_context"])
        self.assertEqual(len(server.calls), 4)  # 2 steps + the 2 full-context requests only
        verdict = res["capacity"]["verdict"]
        self.assertTrue(verdict.startswith("2 requests of 8,000 tokens do not fit at once"), verdict)
        self.assertNotIn("even at", verdict)  # no smaller size was tried

    def test_prefill_queueing_is_not_a_memory_limit(self):
        """Seen live: 4 x 128k prompts fit the KV cache, but prompts are read one
        at a time, so a short answer finished before the last prompt was read.
        The timing rule called that "queued"; the server's counters show every
        request running at once."""
        res = self.run12(_NeedleServer(serial_prefill=True), [4000, 8000], slots=2)
        full = res["capacity"]["full_context"]
        self.assertEqual(full["queued_by_timing"], 1)
        self.assertTrue(full["concurrent"])
        self.assertTrue(res["capacity"]["verdict"].startswith("yes"))

    def test_no_counters_is_undetermined(self):
        res = self.run12(_NeedleServer(counters=False), [4000, 8000], slots=2)
        cap = res["capacity"]
        self.assertIsNone(cap["full_context"]["concurrent"])
        self.assertIsNone(cap["split_context"])
        self.assertTrue(cap["verdict"].startswith("undetermined"))

    def test_capacity_needs_two_slots(self):
        self.assertTrue(self.run12(_NeedleServer(), [4000], slots=1)["capacity"]["skipped"])

    def test_unreachable_server_is_infra(self):
        res = self.run12(_NeedleServer(first_error="Connection refused"), [4000, 8000])
        self.assertEqual(res["failure_kind"], "infra")

    def test_capacity_slots_prefer_test3(self):
        report = {"results": {"concurrency": {"verified_max_parallel": 4}}}
        self.assertEqual(eval_module._capacity_slots(report, 2), 4)
        self.assertEqual(eval_module._capacity_slots({"results": {}}, 2), 2)

    def test_report_reads_the_result(self):
        res = self.run12(_NeedleServer(error_at=16000), [4000, 16000], capacity=False)
        m = reporting.extract_test_metrics("context_scaling", res)
        self.assertEqual((m["status"], m["score"]), ("PARTIAL", 0.5))
        self.assertIn("Failed at 16,000: context_limit_rejected", m["desc"])



class TestReviewFixes(unittest.TestCase):
    """Defects found in code review; each test failed on the old code."""

    def test_slots_above_sweep_limit_are_not_a_failure(self):
        """128 advertised slots, sweep capped at 64, every level parallel: the
        old verdict was PARTIAL ('only 64 of the 128 advertised slots')."""
        levels = [_level(c, 10.0 * c) for c in (1, 2, 4, 8, 16, 32, 64)]
        v = eval_module.capacity_verdict(levels, 128, False)
        self.assertEqual((v["status"], v["verified_max_parallel"]), ("PASS", 64))
        self.assertTrue(v["verified_is_lower_bound"])
        self.assertIn("tested up to 64", v["note"])

    def test_sweep_stopped_early_below_advertised_is_partial(self):
        levels = [_level(1, 10), _level(2, 20), _level(4, 21, queued=2)]
        self.assertEqual(eval_module.capacity_verdict(levels, 8, True)["status"], "PARTIAL")

    def test_token_count_is_not_http_413(self):
        c = eval_module.classify_context_error
        self.assertEqual(c("HTTPError 500: Internal Server Error - prompt of 41300 tokens: CUDA out of memory"),
                         "server_error_or_oom")
        self.assertEqual(c("HTTPError 503: Service Unavailable - 4130 tokens queued"), "server_error_or_oom")
        self.assertEqual(c("HTTPError 400: Bad Request - 41300 tokens exceeds the maximum context length"),
                         "context_limit_rejected")

    def test_early_eos_peer_does_not_mark_slow_prefill_as_queued(self):
        early = dict(_stream(1.0, 2.0, 40), finish_reason="stop")
        slow = [dict(_stream(2.5, 12.0, 256), finish_reason="length") for _ in range(2)]
        lv = eval_module.analyze_level(3, [early] + slow)
        self.assertEqual((lv["queued"], lv["queued_unconfirmed"]), (0, 2))

    def test_early_eos_with_server_waiting_is_queued(self):
        early = dict(_stream(1.0, 2.0, 40), finish_reason="stop")
        late = dict(_stream(2.5, 12.0, 256), finish_reason="length")
        lv = eval_module.analyze_level(2, [early, late], {"max_waiting": 1, "max_running": 1})
        self.assertEqual((lv["queued"], lv["queued_unconfirmed"]), (1, 0))

    def test_full_length_finish_still_shows_queueing(self):
        first = dict(_stream(1.0, 2.0, 100), finish_reason="length")
        second = dict(_stream(2.5, 3.5, 100), finish_reason="length")
        self.assertEqual(eval_module.analyze_level(2, [first, second])["queued"], 1)

    def test_context_ttft_is_the_smallest_step(self):
        """The 125k step's 40 s prefill went into the headline mean TTFT."""
        res = {"status": "PASS", "final_step_tokens": 125000, "steps": [
            {"status": "PASS", "target_tokens": 4000, "ttft_s": 0.12, "decode_tok_s": 50.0},
            {"status": "PASS", "target_tokens": 125000, "ttft_s": 40.0, "decode_tok_s": 30.0}]}
        self.assertEqual(reporting.extract_test_metrics("context_scaling", res)["ttft_ms"], 120.0)


def _probe_server(routes):
    """urlopen stand-in: routes[path] is a body (str), an int HTTP status,
    or an exception to raise. Records every requested path."""
    import io
    import urllib.error
    seen = []

    def urlopen(req, timeout=None):
        path = "/" + req.full_url.split("/", 3)[3]
        seen.append(path)
        answer = routes.get(path, 404)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, int):
            raise urllib.error.HTTPError(req.full_url, answer, "status", {}, io.BytesIO(b""))
        resp = mock.MagicMock()
        resp.read.return_value = answer.encode()
        resp.__enter__.return_value = resp
        return resp
    return urlopen, seen


class TestProbeEndpoints(unittest.TestCase):
    """A probe endpoint that is not there is not asked again."""

    def _client(self, routes):
        import client as client_module
        urlopen, seen = _probe_server(routes)
        patcher = mock.patch.object(client_module.urllib.request, "urlopen", side_effect=urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)
        return LLMClient("http://srv:8888/v1"), seen

    def test_server_without_counters_is_asked_once(self):
        c, seen = self._client({"/health": '{"status": "ok"}'})
        self.assertIsNone(c.fetch_queue_state())
        self.assertEqual(seen, ["/metrics", "/health", "/slots"])
        for _ in range(5):
            self.assertIsNone(c.fetch_queue_state())
        self.assertEqual(len(seen), 3)

    def test_working_source_is_the_only_one_polled(self):
        """/metrics answers without queue counters; /health has them. The old
        code fetched /metrics again on every poll."""
        c, seen = self._client({"/metrics": "process_resident_memory_bytes 1e9\n",
                                "/health": '{"gpu_active_jobs": 2, "gpu_pending_jobs": 0}'})
        for _ in range(4):
            self.assertEqual(c.fetch_queue_state()["running"], 2)
        self.assertEqual(seen, ["/metrics", "/health", "/health", "/health", "/health"])

    def test_timeout_does_not_give_up(self):
        c, seen = self._client({"/metrics": TimeoutError("timed out"), "/health": '{"status": "ok"}'})
        c.fetch_queue_state()
        c.fetch_queue_state()
        self.assertEqual(seen.count("/metrics"), 2)

    def test_absent_metrics_not_scraped_again(self):
        c, seen = self._client({})
        self.assertIsNone(c.fetch_server_metrics())
        self.assertIsNone(c.fetch_kv_capacity())
        self.assertEqual(seen, ["/metrics"])

    def test_refused_connection_is_remembered(self):
        import urllib.error
        c, seen = self._client({"/health": urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))})
        self.assertIsNone(c.fetch_health())
        self.assertIsNone(c.fetch_health())
        self.assertEqual(seen, ["/health"])


if __name__ == "__main__":
    unittest.main()
