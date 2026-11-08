#!/usr/bin/env python3
"""Tests for the phased run: isolated tests first, then a parallel phase under
one request limit (eval.py scheduling, client limiter and log buffering,
polyglot toolchain locks, reporting of under-load speeds)."""

import os
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import client as client_module
import eval as eval_module
import polyglot
import reporting
from client import LLMClient

Job = eval_module.Job


def _job(num, key=None, fn=None):
    return Job(num, key or f"t{num}", fn or (lambda: {"status": "PASS"}), f"Test {num}")


class _ConcurrencyMeter:
    """Records the most calls that were inside `hold()` at the same time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = self.peak = 0

    def hold(self, seconds=0.03):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(seconds)
        with self.lock:
            self.now -= 1


class TestSplitPhases(unittest.TestCase):
    def test_isolated_tests_first_in_fixed_order(self):
        jobs = [_job(n) for n in (1, 2, 3, 9, 12, 20, 32)]
        phase1, phase2 = eval_module.split_phases(jobs)
        self.assertEqual([j.num for j in phase1], [3, 12, 1, 9])
        self.assertEqual([j.num for j in phase2], [2, 20, 32])

    def test_timing_sensitive_tests_are_isolated(self):
        """The old --parallel-suite mode ran Tests 1, 9 and 13 in the pool.
        Test 11's raw probes bypass the request limit and time out after 15 s,
        so in the pool they queued and were graded as server faults."""
        for num in (1, 3, 7, 8, 9, 11, 12, 13, 31, 46):
            self.assertIn(num, eval_module.ISOLATED_TESTS)


class TestParallelPhase(unittest.TestCase):
    def test_runs_jobs_concurrently_and_flags_speeds(self):
        meter = _ConcurrencyMeter()

        def work():
            meter.hold()
            return {"status": "PASS", "tok_s": 40.0}

        report = {"results": {}}
        eval_module.run_parallel_phase([_job(n, fn=work) for n in (2, 4, 5, 6)], workers=4, report=report)
        self.assertGreater(meter.peak, 1)
        self.assertEqual(set(report["results"]), {"t2", "t4", "t5", "t6"})
        self.assertTrue(all(r["measured_under_load"] for r in report["results"].values()))

    def test_a_crashing_test_does_not_stop_the_phase(self):
        def boom():
            raise RuntimeError("harness bug")

        report = {"results": {}}
        eval_module.run_parallel_phase([_job(2, fn=boom), _job(4)], workers=2, report=report)
        self.assertEqual(report["results"]["t2"]["status"], "FAIL")
        self.assertIn("harness bug", report["results"]["t2"]["error"])
        self.assertEqual(report["results"]["t4"]["status"], "PASS")

    def test_logs_of_parallel_tests_do_not_interleave(self):
        emitted = []

        def chatty(name):
            def fn():
                for i in range(5):
                    client_module.log(f"{name}-{i}")
                    time.sleep(0.005)
                return {"status": "PASS"}
            return fn

        with mock.patch.object(client_module, "_emit", side_effect=lambda line, msg: emitted.append(str(msg))):
            eval_module.run_parallel_phase([_job(2, fn=chatty("A")), _job(4, fn=chatty("B"))], workers=2,
                                           report={"results": {}})
        body = [m for m in emitted if m[:2] in ("A-", "B-")]
        a_idx = [i for i, m in enumerate(body) if m.startswith("A-")]
        b_idx = [i for i, m in enumerate(body) if m.startswith("B-")]
        self.assertEqual(a_idx, list(range(a_idx[0], a_idx[0] + 5)))
        self.assertEqual(b_idx, list(range(b_idx[0], b_idx[0] + 5)))


class TestRequestLimiter(unittest.TestCase):
    def test_limit_caps_requests_in_flight(self):
        meter = _ConcurrencyMeter()
        client = LLMClient("http://srv:8080/v1")
        client.request_limiter = threading.BoundedSemaphore(2)
        with mock.patch.object(client, "_call", side_effect=lambda *a, **k: meter.hold() or {"status": "ok"}):
            threads = [threading.Thread(target=client.call, args=([],)) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(meter.peak, 2)

    def test_no_limit_by_default(self):
        """Test 3 must be able to exceed the slot count to measure it."""
        self.assertIsNone(LLMClient("http://srv:8080/v1").request_limiter)


class TestLimits(unittest.TestCase):
    def test_request_limit_precedence(self):
        tested = {"results": {"concurrency": {"verified_max_parallel": 4}}}
        r = eval_module.resolve_request_limit
        self.assertEqual(r(6, "--parallel", tested, 2, "server"), (6, "--parallel"))
        self.assertEqual(r(2, "server /health parallel", tested, 2, "server"), (4, "Test 3 verified"))
        self.assertEqual(r(2, "server /health parallel", {"results": {}}, 2, "server /health parallel"),
                         (2, "server /health parallel"))
        self.assertEqual(r(4, "default", {"results": {}}, None, "not reported"), (4, "default"))

    def test_judge_limit(self):
        j = eval_module.resolve_judge_limit
        self.assertEqual(j(None, None, True)[0], None)
        self.assertEqual(j(3, {"slots": 2, "source": "/health parallel"}, False), (3, "--judge-parallel"))
        self.assertEqual(j(None, {"slots": 2, "source": "/health parallel"}, False),
                         (2, "judge server /health parallel"))
        self.assertEqual(j(None, {"slots": None, "source": "not reported"}, False)[0], 2)


class TestToolchainLocks(unittest.TestCase):
    def _peak(self, lang):
        meter = _ConcurrencyMeter()

        def build(lang_):
            with polyglot._build_lock(lang_):
                meter.hold(0.02)

        threads = [threading.Thread(target=build, args=(lang,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return meter.peak

    def test_shared_state_toolchains_build_one_at_a_time(self):
        """dotnet restore shares the NuGet folder."""
        self.assertEqual(self._peak("csharp"), 1)

    def test_independent_toolchains_build_in_parallel(self):
        """TypeScript runs get their own node_modules folder, so they need no lock."""
        self.assertGreater(self._peak("go"), 1)
        self.assertGreater(self._peak("typescript"), 1)


class TestPolyglotAggregate(unittest.TestCase):
    def test_statuses(self):
        agg = eval_module.aggregate_polyglot
        self.assertEqual(agg({"go": {"status": "PASS"}, "rust": {"status": "PASS"}})["status"], "PASS")
        mixed = agg({"go": {"status": "PASS"}, "rust": {"status": "FAIL"}})
        self.assertEqual((mixed["status"], mixed["polyglot_pass_count"]), ("PARTIAL", 1))
        self.assertEqual(agg({"go": {"status": "FAIL"}, "rust": {"status": "FAIL"}})["status"], "FAIL")


class TestUnderLoadSpeeds(unittest.TestCase):
    def test_headline_speed_uses_isolated_tests_only(self):
        report = {"results": {
            "streaming": {"status": "PASS", "completion_tokens": 100, "tok_s": 50.0, "ttft_ms": 500},
            "aime_olympiad": {"status": "PASS", "completion_tokens": 100, "tok_s": 10.0, "ttft_ms": 9000,
                              "measured_under_load": True},
        }}
        summary = reporting.compute_executive_summary(report)
        self.assertEqual(summary["avg_decode_tok_s"], 50.0)


class TestServerDown(unittest.TestCase):
    """A tested server that goes down mid-run: no more requests, no more
    tests, a report marked incomplete. Before, every remaining test kept
    sending requests (with retries) to a server that was not there."""

    def _refusing_client(self):
        import urllib.error
        c = LLMClient("http://srv:8080/v1", model="m")
        refused = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        urlopen = mock.patch.object(client_module.urllib.request, "urlopen", side_effect=refused)
        sleep = mock.patch.object(client_module.time, "sleep")
        self.addCleanup(urlopen.stop)
        self.addCleanup(sleep.stop)
        return c, urlopen.start(), sleep.start()

    def test_no_request_after_the_server_is_found_down(self):
        c, urlopen, sleep = self._refusing_client()
        first = c.call([{"role": "user", "content": "hi"}])
        self.assertTrue(first["endpoint_unreachable"])
        self.assertIsNotNone(c.server_down)
        # A short wait for a restart, then the server is declared down.
        delays = client_module.UNREACHABLE_RETRY_DELAYS
        self.assertEqual(urlopen.call_count, 1 + len(delays))
        self.assertEqual([call.args[0] for call in sleep.call_args_list], list(delays))
        self.assertLessEqual(sum(delays), 60)
        sent = urlopen.call_count
        second = c.call([{"role": "user", "content": "hi"}])
        self.assertEqual(urlopen.call_count, sent)
        self.assertEqual(second["failure_kind"], "infra")
        self.assertIn("not sent", second["error"])

    def test_server_restart_within_the_retry_window_is_not_down(self):
        import io
        import urllib.error

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        refused = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        ok = Response(b'{"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}')
        c = LLMClient("http://srv:8080/v1", model="m")
        with mock.patch.object(client_module.urllib.request, "urlopen", side_effect=[refused, refused, ok]), \
                mock.patch.object(client_module.time, "sleep") as sleep:
            res = c.call([{"role": "user", "content": "hi"}], stream=False, retries=0)
        self.assertIsNone(res.get("error"))
        self.assertEqual(res["content"], "hello")
        self.assertIsNone(c.server_down)
        self.assertEqual(sleep.call_count, 2)

    def test_timeout_is_not_down(self):
        c = LLMClient("http://srv:8080/v1", model="m")
        with mock.patch.object(client_module.urllib.request, "urlopen", side_effect=TimeoutError("timed out")), \
                mock.patch.object(client_module.time, "sleep"):
            res = c.call([{"role": "user", "content": "hi"}])
        self.assertEqual(res["failure_kind"], "infra")
        self.assertIsNone(c.server_down)

    def test_parallel_phase_starts_no_job_after_stop(self):
        started = []
        jobs = [_job(n, fn=(lambda n=n: started.append(n) or {"status": "PASS"})) for n in (20, 21, 22)]
        report = {"results": {}}
        eval_module.run_parallel_phase(jobs, workers=1, report=report, stop=lambda: True)
        self.assertEqual((started, report["results"]), ([], {}))

    def test_incomplete_report_lists_tests_not_run(self):
        jobs = [_job(1, "streaming"), _job(4, "capabilities_4tasks"), _job(26, "prompt_injection_defense")]
        report = {"results": {"streaming": {"status": "PASS"}}}
        not_run = eval_module.mark_incomplete(report, jobs, "tested server unreachable")
        self.assertEqual(not_run, ["capabilities_4tasks", "prompt_injection_defense"])
        self.assertEqual(report["results"]["streaming"]["status"], "PASS")
        self.assertEqual(report["results"]["capabilities_4tasks"]["status"], "SKIPPED")
        self.assertEqual(report["incomplete"]["tests_not_run"], not_run)
        summary = reporting.compute_executive_summary(report)
        self.assertEqual(summary["total_evaluated"], 1)  # tests not run are not scored


class TestJobTimings(unittest.TestCase):
    """Every test that runs records its start, end and duration."""

    def _slow(self, seconds):
        def fn():
            time.sleep(seconds)
            return {"status": "PASS"}
        return fn

    def _check(self, timing, key, min_duration):
        self.assertEqual(timing["key"], key)
        self.assertGreaterEqual(timing["duration_s"], min_duration)
        start = time.mktime(time.strptime(timing["started_at"][:19], "%Y-%m-%dT%H:%M:%S"))
        end = time.mktime(time.strptime(timing["finished_at"][:19], "%Y-%m-%dT%H:%M:%S"))
        self.assertAlmostEqual(end - start, timing["duration_s"], delta=1.0)

    def test_parallel_phase_records_each_job(self):
        report = {"results": {}}
        eval_module.run_parallel_phase([_job(20, fn=self._slow(0.25)), _job(21)], workers=2, report=report)
        by_key = {t["key"]: t for t in report["test_timings"]}
        self.assertEqual(set(by_key), {"t20", "t21"})
        self._check(by_key["t20"], "t20", 0.2)

    def test_isolated_job_records_timing_and_result(self):
        report = {"results": {}}
        eval_module.run_job_in_order(_job(3, fn=self._slow(0.25)), report)
        self.assertEqual(report["results"]["t3"]["status"], "PASS")
        self._check(report["test_timings"][0], "t3", 0.2)

    def test_crashing_job_still_records_timing(self):
        def boom():
            raise RuntimeError("harness bug")
        report = {"results": {}}
        eval_module.run_job_in_order(_job(3, fn=boom), report)
        self.assertEqual(report["results"]["t3"]["status"], "FAIL")
        self.assertEqual(len(report["test_timings"]), 1)

    def test_finished_line_shows_times_and_duration(self):
        lines = []
        with mock.patch.object(eval_module, "log", side_effect=lambda msg, **_: lines.append(msg)):
            eval_module.run_job_in_order(_job(3, fn=self._slow(0.25)), {"results": {}})
        finished = [m for m in lines if "<< finished" in m]
        self.assertEqual(len(finished), 1)
        self.assertRegex(finished[0], r"-> PASS \| \d\d:\d\d:\d\d to \d\d:\d\d:\d\d \(0\.\d s\)")

    def test_timing_table_lists_tests_in_start_order(self):
        report = {"results": {"b": {"status": "PASS"}, "a": {"status": "FAIL"}},
                  "total_suite_wall_time_s": 3700,
                  "test_timings": [
                      {"key": "b", "label": "Test 4 b", "started_at": "2026-11-06T20:10:00+0000",
                       "finished_at": "2026-11-06T20:28:06+0000", "duration_s": 1086.0},
                      {"key": "a", "label": "Test 3 a", "started_at": "2026-11-06T20:00:00+0000",
                       "finished_at": "2026-11-06T20:00:42+0000", "duration_s": 42.3}]}
        lines = []
        with mock.patch.object(reporting, "log", side_effect=lambda msg="", **_: lines.append(msg)):
            reporting.print_test_timings(report)
        rows = [m for m in lines if m.startswith("  Test ") and "Started" not in m]
        self.assertIn("Test 3 a", rows[0])
        self.assertIn("FAIL", rows[0])
        self.assertIn("42.3 s", rows[0])
        self.assertIn("18m 06s", rows[1])
        self.assertTrue(any("1h 01m 40s" in m for m in lines))


class TestRunHeaderFormatting(unittest.TestCase):
    def test_duration_format(self):
        self.assertEqual(reporting.format_duration(42.34), "42.3 s")
        self.assertEqual(reporting.format_duration(59.96), "1m 00s")
        self.assertEqual(reporting.format_duration(60), "1m 00s")
        self.assertEqual(reporting.format_duration(1086), "18m 06s")
        self.assertEqual(reporting.format_duration(3723), "1h 02m 03s")

    def test_test_numbers_collapse_into_ranges(self):
        self.assertEqual(reporting.format_test_numbers(range(1, 46)), "1-45")
        self.assertEqual(reporting.format_test_numbers({8, 1, 2, 3, 5, 7}), "1-3, 5, 7-8")
        self.assertEqual(reporting.format_test_numbers({12}), "12")

    def test_test_headers_use_the_real_total(self):
        """The headers said [TEST n/31] after the suite grew to 45 tests."""
        with open(os.path.join(ROOT, "eval.py")) as f:
            src = f.read()
        self.assertNotRegex(src, r"\[TEST \d+/31\]")


if __name__ == "__main__":
    unittest.main()
