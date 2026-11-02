#!/usr/bin/env python3
"""
Reporting, metric extraction, and cross-model comparison engine for llm-eval-suite.

Zero external dependencies (Python stdlib only). Extracted from eval.py so that
eval.py, compare.py, and external tooling share one implementation.
"""

import os
import sys
import json
import re
import time
import hashlib

from client import log, BOLD, GREEN, YELLOW, RED, CYAN, MAGENTA, RESET, DEFAULT_RESULTS_DIR

TOTAL_TESTS = 26

# ============================================================================
# STANDARDIZED TEST CATALOG & METRIC EXTRACTORS
# ============================================================================

PILLARS = {
    "serving": "Serving Infrastructure & Multi-Client Concurrency",
    "contract": "API Contracts, Schema & Agent Protocols",
    "context": "Long Context & Attention Scaling",
    "swe": "Software Engineering & Autonomous Coding",
    "reasoning": "Cognitive Reasoning & Safety Defense",
}

TEST_CATALOG = [
    (1, "streaming", "Streaming & Latency"),
    (2, "vision", "Multimodal Vision (Invoice)"),
    (3, "concurrency", "Parallel Batching & Saturation"),
    (4, "capabilities_4tasks", "4-Task Architecture Suite"),
    (5, "tool_calling", "Tool Calling & Agentic Recovery"),
    (6, "json_schema", "JSON Schema Mode (response_format)"),
    (7, "prefix_caching", "Prefix / KV Cache Reuse"),
    (8, "client_abort", "Client Socket Abort Recovery"),
    (9, "stop_sequences", "Stop Words & Greedy Sampling"),
    (10, "extreme_precision", "Precision Ledger Reconcile"),
    (11, "error_handling", "API Error & Boundary Protocol"),
    (12, "context_scaling", "Dynamic Context Scaling"),
    (13, "multihop_graph", "Adversarial Graph & Distractors"),
    (14, "novel_algorithm_fuzz", "Novel Algorithm (5k Fuzz)"),
    (15, "frontier_needle_depth", "Frontier Depth Multi-Needle"),
    (16, "cruxeval", "CruxEval (Mental Code Exec)"),
    (17, "swe_bench_bug_patch", "SWE-bench (Traceback Fix)"),
    (18, "aime_olympiad", "AIME Olympiad (Math Reasoning)"),
    (19, "gpqa_diamond", "GPQA Diamond (Graduate Science)"),
    (20, "surgical_git_diff", "Surgical Unified Diff (git apply)"),
    (21, "multifile_context", "Multi-File Cross-Reference & DI"),
    (22, "code_review_security", "Security Review & OWASP Audit"),
    (23, "unit_test_synthesis", "Unit Test Synthesis with Mocks"),
    (24, "strict_typing", "Strict Type Compliance (mypy)"),
    (25, "refactor_invariants", "Safe Refactor Invariant Check"),
    (26, "prompt_injection_defense", "Prompt Injection & Delimiter Defense"),
]

TEST_KEY_TO_PILLAR = {
    "streaming": "serving",
    "vision": "serving",
    "concurrency": "serving",
    "prefix_caching": "serving",
    "client_abort": "serving",
    "error_handling": "serving",
    "tool_calling": "contract",
    "json_schema": "contract",
    "stop_sequences": "contract",
    "extreme_precision": "contract",
    "context_scaling": "context",
    "multihop_graph": "context",
    "frontier_needle_depth": "context",
    "capabilities_4tasks": "swe",
    "novel_algorithm_fuzz": "swe",
    "swe_bench_bug_patch": "swe",
    "surgical_git_diff": "swe",
    "multifile_context": "swe",
    "code_review_security": "swe",
    "unit_test_synthesis": "swe",
    "strict_typing": "swe",
    "refactor_invariants": "swe",
    "cruxeval": "reasoning",
    "aime_olympiad": "reasoning",
    "gpqa_diamond": "reasoning",
    "prompt_injection_defense": "reasoning",
}


# ============================================================================
# INFRASTRUCTURE-FAILURE CLASSIFICATION (completion rate)
# ============================================================================
# A test that fails because the *serving environment* broke (network error,
# timeout, 5xx, rate limit) measures the infrastructure, not the model. Those
# are excluded from effectiveness and reported via completion_rate instead.
_INFRA_PATTERNS = (
    "HTTPError 5",       # 5xx server errors
    "HTTPError 429",     # rate limit
    "Bad Gateway",
    "Service Unavailable",
    "Gateway Time-out",
    "timed out",
    "timeout",
    "TimeoutError",
    "Connection refused",
    "Connection reset",
    "ConnectionError",
    "Connection aborted",
    "Max retries exceeded",
    "RemoteProtocolError",
    "NewConnectionError",
    "Name or service not known",
    "Temporary failure in name resolution",
)


def _looks_like_infra_error(text):
    if not text:
        return False
    t = str(text)
    return any(p in t for p in _INFRA_PATTERNS)


def _classify_infra(data):
    """True if a test result represents an infrastructure failure (network /
    timeout / 5xx / 429) rather than a model-quality failure."""
    if data is None:
        return False
    if isinstance(data, str):
        return _looks_like_infra_error(data)
    if isinstance(data, list):
        sub = [d for d in data if isinstance(d, dict)]
        if not sub:
            return False
        return all(_classify_infra(d) for d in sub)
    if isinstance(data, dict):
        if data.get("failure_kind") in ("infra", "rate_limit"):
            return True
        if "error" in data and _looks_like_infra_error(data.get("error")):
            return True
        if "errors" in data and isinstance(data["errors"], list) and data["errors"]:
            return all(_looks_like_infra_error(e) for e in data["errors"])
        sub_dicts = [v for k, v in data.items() if isinstance(v, dict) and k != "matrix"]
        if sub_dicts:
            return all(_classify_infra(v) for v in sub_dicts)
    return False


# ============================================================================
# CONFIG FINGERPRINT (comparability)
# ============================================================================
def build_config_fingerprint(report):
    """Hash the comparable config (model, context, parallel, milestones, suite)
    so two runs can be checked for comparability before a head-to-head compare."""
    comparable = {
        "model": report.get("model"),
        "max_context_tokens": report.get("max_context_tokens"),
        "parallel_streams": report.get("parallel_streams"),
        "context_milestones": report.get("context_milestones"),
        "suite": report.get("suite"),
    }
    blob = json.dumps(comparable, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ============================================================================
# STAR RATING (at-a-glance quality)
# ============================================================================
def star_rating(eff_pct, completion_pct):
    """1-5 star at-a-glance rating. Capped at 3 stars when the serving
    environment was too flaky (completion < 80%) to trust the score."""
    if eff_pct >= 95:
        stars = 5
    elif eff_pct >= 85:
        stars = 4
    elif eff_pct >= 70:
        stars = 3
    elif eff_pct >= 50:
        stars = 2
    else:
        stars = 1
    if completion_pct < 80:
        stars = min(stars, 3)
    return stars


def extract_test_metrics(test_key: str, data):
    """Extracts standardized metrics (status, tokens, tok_s, ttft_ms, score, desc) from any test result."""
    if not data or not isinstance(data, (dict, list)):
        return {"status": "SKIPPED", "tokens": 0, "tok_s": 0.0, "ttft_ms": 0.0, "score": 0.0, "desc": "-"}

    if isinstance(data, list):
        if not data:
            return {"status": "SKIPPED", "tokens": 0, "tok_s": 0.0, "ttft_ms": 0.0, "score": 0.0, "desc": "-"}
        passed = sum(1 for s in data if s.get("status") == "PASS")
        partial = sum(1 for s in data if s.get("status") == "PARTIAL")
        st = "PASS" if passed == len(data) else ("PARTIAL" if (passed + partial) > 0 else "FAIL")
        tot_tokens = sum(s.get("completion_tokens", 0) for s in data)
        speeds = [s.get("decode_tok_s", 0) for s in data if s.get("decode_tok_s")]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        ttfts = [s.get("ttft_s", 0) * 1000.0 for s in data if s.get("ttft_s")]
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        score = (passed + 0.5 * partial) / len(data)
        desc = f"{passed}/{len(data)} Milestones OK"
        return {"status": st, "tokens": tot_tokens, "tok_s": round(avg_speed, 2), "ttft_ms": round(avg_ttft, 1), "score": score, "desc": desc}

    if test_key == "concurrency":
        if not isinstance(data, dict):
            return {"status": "FAIL", "tokens": 0, "tok_s": 0.0, "ttft_ms": 0.0, "score": 0.0, "desc": str(data)}
        levels = [l for l in data.values() if isinstance(l, dict) and "aggregate_tok_s" in l]
        if not levels:
            st = data.get("status", "FAIL")
            err = data.get("error", "-")
            return {"status": st, "tokens": 0, "tok_s": 0.0, "ttft_ms": 0.0, "score": 0.0, "desc": err}
        all_passed = all("PASS" in str(l.get("status", "")) for l in levels)
        st = "PASS" if all_passed else "FAIL"
        tot_tokens = sum(l.get("total_tokens", 0) for l in levels)
        max_agg = max((l.get("aggregate_tok_s", 0) for l in levels), default=0.0)
        ttfts = [l.get("mean_ttft_ms", 0.0) for l in levels if l.get("mean_ttft_ms")]
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        score = 1.0 if all_passed else 0.0
        desc = f"Max Agg: {max_agg:.1f} tok/s"
        return {"status": st, "tokens": tot_tokens, "tok_s": round(max_agg, 2), "ttft_ms": round(avg_ttft, 1), "score": score, "desc": desc}

    if test_key == "capabilities_4tasks":
        if not isinstance(data, dict):
            return {"status": "FAIL", "tokens": 0, "tok_s": 0.0, "ttft_ms": 0.0, "score": 0.0, "desc": str(data)}
        tasks = [t for t in data.get("tasks", []) if isinstance(t, dict)]
        tot_tokens = sum(t.get("tokens", 0) for t in tasks)
        avg_speed = data.get("average_speed_tok_s", 0.0)
        ttfts = [t.get("ttft_ms", 0) for t in tasks if t.get("ttft_ms")]
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        st = data.get("status", "PASS" if tasks else "FAIL")
        score = 1.0 if "PASS" in st else (0.5 if "PARTIAL" in st else 0.0)
        desc = f"{len(tasks)}/4 Tasks OK" if tasks else data.get("error", "Failed")
        return {"status": st, "tokens": tot_tokens, "tok_s": round(avg_speed, 2), "ttft_ms": round(avg_ttft, 1), "score": score, "desc": desc}

    st = data.get("status", "N/A")
    tokens = data.get("completion_tokens", 0) or data.get("tokens", 0)
    tok_s = data.get("tok_s", 0.0) or data.get("decode_speed", 0.0)
    ttft_ms = data.get("ttft_ms", 0.0) or ((data.get("ttft") or 0.0) * 1000.0)
    score = 1.0 if "PASS" in st else (0.5 if "PARTIAL" in st else 0.0)

    desc = "-"
    if test_key == "streaming":
        desc = f"TTFT: {ttft_ms:.1f}ms"
    elif test_key == "vision":
        desc = f"TTFT: {ttft_ms:.1f}ms" if ttft_ms > 0 else "Invoice Parsed"
    elif test_key == "prefix_caching":
        ratio = data.get("speedup_ratio", 1.0)
        desc = f"{ratio:.1f}x Cache Speedup"
    elif test_key == "tool_calling":
        desc = "Agent Error Recovered" if data.get("turn2_recovered") else ("Turn 1 Valid" if data.get("turn1_valid") else "Failed")
    elif test_key == "json_schema":
        desc = "Strict Schema Conformed" if data.get("valid_schema") else "Schema Invalid"
    elif test_key == "combinatorial_anti_constraints":
        cp = data.get("constraints_passed", 0)
        desc = f"{cp}/6 Constraints"
        score = cp / 6.0
    elif test_key == "counterfactual_algebra":
        desc = "Roots [8, 17] Solved" if st == "PASS" else "Roots Missed"
    elif test_key == "cruxeval":
        desc = "State Traversed OK" if st == "PASS" else "State Diverged"
    elif test_key == "swe_bench_bug_patch":
        desc = "Regression Tests OK" if st == "PASS" else "Unit Tests Failed"
    elif test_key == "aime_olympiad":
        desc = "Exact Modular Root" if st == "PASS" else "Arithmetic Missed"
    elif test_key == "gpqa_diamond":
        score_val = data.get("score") or f"{data.get('passed_count', 0)}/3"
        desc = f"Score: {score_val}"
    elif test_key == "novel_algorithm_fuzz":
        desc = "5,000 Ops Fuzzed OK" if st == "PASS" else "Fuzz Error"
    elif test_key == "extreme_precision":
        desc = "Exact Matched" if data.get("matched") else "Balance Diverged"
    elif test_key == "high_entropy_recall":
        desc = "Keys Recalled" if st == "PASS" else "Key Missed"
    elif test_key == "client_abort":
        desc = f"Rec: {data.get('recovery_latency_ms', 0):.0f}ms"
    elif test_key == "stop_sequences":
        desc = "Deterministic (temp=0)" if data.get("deterministic_greedy_reproducible") else "Stops Active"
    elif test_key == "code_execution":
        desc = "Dynamic Assertions OK" if data.get("dynamic_tests_passed") else "Assertions Failed"
    elif test_key == "error_handling":
        desc = "HTTP 400/422 Standard" if data.get("bad_schema_rejected") else "Handled"
    elif test_key == "surgical_git_diff":
        desc = "git apply --check OK" if st == "PASS" else ("Syntax Valid" if data.get("valid_diff_syntax") else "Patch Rejected")
    elif test_key == "multifile_context":
        desc = "3/3 Modules Aligned" if st == "PASS" else "Symbol Mismatch"
    elif test_key == "code_review_security":
        desc = "SQLi & Race Detected" if st == "PASS" else "Vulnerabilities Missed"
    elif test_key == "unit_test_synthesis":
        passed_n = data.get("unit_tests_passed_count", 0)
        desc = f"{passed_n} Mock Tests OK" if st == "PASS" else "Unit Tests Failed"
    elif test_key == "strict_typing":
        desc = "0 Type Errors (mypy)" if data.get("mypy_verified") else ("100% Annotated" if st == "PASS" else "Type Errors")
    elif test_key == "refactor_invariants":
        desc = "Regression Suite OK" if st == "PASS" else "Behavior Diverged"
    elif test_key == "prompt_injection_defense":
        desc = "Invariants Protected" if st == "PASS" else "Security Hijacked"

    return {
        "status": st,
        "tokens": int(tokens),
        "tok_s": round(tok_s, 2),
        "ttft_ms": round(ttft_ms, 1),
        "score": score,
        "desc": desc
    }


TASK_ID_MAPPINGS = {
    "task1_streaming": "streaming",
    "task2_vision": "vision",
    "task3_concurrency": "concurrency",
    "task4_avl_tree": "capabilities_4tasks",
    "task5_concurrency_debug": "capabilities_4tasks",
    "task6_system_design": "capabilities_4tasks",
    "task7_long_context_constraints": "capabilities_4tasks",
    "task8_agent_tool_calling": "tool_calling",
    "task9_json_schema": "json_schema",
    "task10_prefix_caching": "prefix_caching",
    "task11_client_abort": "client_abort",
    "task12_stop_sequences": "stop_sequences",
    "task13_high_entropy_recall": "frontier_needle_depth",
    "task14_precision_math": "extreme_precision",
    "task15_code_execution": "novel_algorithm_fuzz",
    "task16_error_handling": "error_handling",
    "task17_multihop_graph": "multihop_graph",
    "task18_novel_algorithm_fuzz": "novel_algorithm_fuzz",
    "task19_anti_constraints": "capabilities_4tasks",
    "task20_counterfactual_algebra": "aime_olympiad",
    "task21_cruxeval_execution": "cruxeval",
    "task22_swe_bench_bug_patch": "swe_bench_bug_patch",
    "task23_aime_olympiad_math": "aime_olympiad",
}


def normalize_report(report: dict) -> dict:
    """Normalizes report structures from eval.py or compare.py to guarantee consistent results dict."""
    if not isinstance(report, dict):
        return {"model": "unknown", "endpoint": "unknown", "results": {}}

    rep = dict(report)
    if "model" not in rep:
        rep["model"] = rep.get("model1") or rep.get("target_model") or "unknown"
    if "endpoint" not in rep:
        rep["endpoint"] = rep.get("base_url") or rep.get("endpoint1") or rep.get("url") or "unknown"

    res = rep.get("results")
    if isinstance(res, list):
        res_dict = {}
        for item in res:
            if isinstance(item, dict):
                tid = item.get("task_id") or item.get("id") or item.get("name")
                if tid:
                    res_dict[tid] = item
                    canonical = TASK_ID_MAPPINGS.get(tid)
                    if canonical and canonical not in res_dict:
                        res_dict[canonical] = item
        all_passed = all("PASS" in str(item.get("status", "")) for item in res if isinstance(item, dict))
        any_passed = any("PASS" in str(item.get("status", "")) or "PARTIAL" in str(item.get("status", "")) for item in res if isinstance(item, dict))
        st = "PASS" if all_passed else ("PARTIAL" if any_passed else "FAIL")
        speeds = [item.get("tok_per_sec", 0.0) for item in res if isinstance(item, dict) and item.get("tok_per_sec")]
        avg_spd = sum(speeds) / len(speeds) if speeds else 0.0
        res_dict["capabilities_4tasks"] = {
            "status": st,
            "average_speed_tok_s": round(avg_spd, 2),
            "tasks": res
        }
        rep["results"] = res_dict
    elif not isinstance(res, dict):
        rep["results"] = {}

    return rep


def compute_relative_advantages(m1: dict, m2: dict) -> dict:
    """Computes relative advantages and executive verdict between two evaluated models."""
    name1 = m1.get("model", "Model A")
    name2 = m2.get("model", "Model B")

    # 1. Effectiveness
    eff_a = m1.get("effectiveness_rate_pct", 0.0)
    eff_b = m2.get("effectiveness_rate_pct", 0.0)
    diff_eff = eff_a - eff_b
    if diff_eff > 0:
        adv_eff = f"Model A (+{diff_eff:.1f}%)"
    elif diff_eff < 0:
        adv_eff = f"Model B (+{-diff_eff:.1f}%)"
    else:
        adv_eff = "Equal"

    # 2. Token Economy
    te_a = m1.get("token_economy_tokens_per_passed_task", 0.0)
    te_b = m2.get("token_economy_tokens_per_passed_task", 0.0)
    if te_a > 0 and te_b > 0:
        if te_a < te_b:
            ratio = te_b / te_a
            pct = ((te_b - te_a) / te_b) * 100
            adv_te = f"Model A ({pct:.1f}% fewer toks, {ratio:.2f}x conciseness)"
        elif te_b < te_a:
            ratio = te_a / te_b
            pct = ((te_a - te_b) / te_a) * 100
            adv_te = f"Model B ({pct:.1f}% fewer toks, {ratio:.2f}x conciseness)"
        else:
            adv_te = "Equal"
    else:
        adv_te = "-"

    # 3. Total Tokens Emitted
    tot_a = m1.get("total_tokens_emitted", 0)
    tot_b = m2.get("total_tokens_emitted", 0)
    if tot_a < tot_b:
        adv_tot = f"Model A ({tot_b - tot_a:,} fewer tokens)"
    elif tot_b < tot_a:
        adv_tot = f"Model B ({tot_a - tot_b:,} fewer tokens)"
    else:
        adv_tot = "Equal"

    # 4. Generation Speed
    spd_a = m1.get("avg_decode_tok_s", 0.0)
    spd_b = m2.get("avg_decode_tok_s", 0.0)
    if spd_a > 0 and spd_b > 0:
        if spd_a > spd_b:
            diff_spd = ((spd_a - spd_b) / spd_b) * 100
            adv_spd = f"Model A (+{diff_spd:.1f}% faster)"
        elif spd_b > spd_a:
            diff_spd = ((spd_b - spd_a) / spd_a) * 100
            adv_spd = f"Model B (+{diff_spd:.1f}% faster)"
        else:
            adv_spd = "Equal"
    else:
        adv_spd = "-"

    # 5. First Token Latency (TTFT)
    ttft_a = m1.get("avg_ttft_ms", 0.0)
    ttft_b = m2.get("avg_ttft_ms", 0.0)
    if ttft_a > 0 and ttft_b > 0:
        if ttft_a < ttft_b:
            adv_ttft = f"Model A ({ttft_b / ttft_a:.2f}x lower latency)"
        elif ttft_b < ttft_a:
            adv_ttft = f"Model B ({ttft_a / ttft_b:.2f}x lower latency)"
        else:
            adv_ttft = "Equal"
    else:
        adv_ttft = "-"

    # 6. Total Wall Clock Time
    wall_a = m1.get("total_wall_time_s", 0.0)
    wall_b = m2.get("total_wall_time_s", 0.0)
    if wall_a < wall_b:
        adv_wall = f"Model A ({wall_b - wall_a:.1f}s faster)"
    elif wall_b < wall_a:
        adv_wall = f"Model B ({wall_a - wall_b:.1f}s faster)"
    else:
        adv_wall = "Equal"

    # 7. Composite Efficiency Index
    ei_a = m1.get("efficiency_index", 0.0)
    ei_b = m2.get("efficiency_index", 0.0)
    if ei_a > ei_b:
        adv_ei = f"Model A (+{ei_a - ei_b:.1f} pts)"
    elif ei_b > ei_a:
        adv_ei = f"Model B (+{ei_b - ei_a:.1f} pts)"
    else:
        adv_ei = "Equal"

    # Executive Verdict
    if eff_a > eff_b and te_a <= te_b:
        verdict = f"MODEL A ({name1}) DOMINATES: Delivers superior effectiveness (+{diff_eff:.1f}%) while maintaining higher token economy ({te_a:.1f} vs {te_b:.1f} tokens/task)."
    elif eff_b > eff_a and te_b <= te_a:
        verdict = f"MODEL B ({name2}) DOMINATES: Delivers superior effectiveness (+{-diff_eff:.1f}%) while maintaining higher token economy ({te_b:.1f} vs {te_a:.1f} tokens/task)."
    elif eff_a > eff_b:
        verdict = f"MODEL A ({name1}) is MORE EFFECTIVE (+{diff_eff:.1f}% accuracy), while Model B had token economy of {te_b:.1f} tokens/task."
    elif eff_b > eff_a:
        verdict = f"MODEL B ({name2}) is MORE EFFECTIVE (+{-diff_eff:.1f}% accuracy), while Model A had token economy of {te_a:.1f} tokens/task."
    elif spd_a > spd_b * 1.15:
        verdict = f"Both models achieved identical effectiveness ({eff_a:.1f}%), but MODEL A ({name1}) LEADS ON THROUGHPUT (+{((spd_a - spd_b)/spd_b)*100:.1f}% faster decode speed)."
    elif spd_b > spd_a * 1.15:
        verdict = f"Both models achieved identical effectiveness ({eff_a:.1f}%), but MODEL B ({name2}) LEADS ON THROUGHPUT (+{((spd_b - spd_a)/spd_a)*100:.1f}% faster decode speed)."
    else:
        verdict = f"Both models achieved IDENTICAL effectiveness ({eff_a:.1f}%). Model A speed: {spd_a:.1f} tok/s vs Model B speed: {spd_b:.1f} tok/s."

    return {
        "adv_eff": adv_eff,
        "adv_te": adv_te,
        "adv_tot": adv_tot,
        "adv_spd": adv_spd,
        "adv_ttft": adv_ttft,
        "adv_wall": adv_wall,
        "adv_ei": adv_ei,
        "verdict": verdict,
        "diff_eff": diff_eff,
    }


def compute_executive_summary(report: dict) -> dict:
    report = normalize_report(report)
    results = report.get("results", {})
    evaluated = 0
    passed = 0
    partial = 0
    failed = 0
    infra_excluded = 0
    infra_excluded_tests = []
    total_seen = 0
    total_tokens = 0
    passed_tokens = 0
    speeds = []
    ttfts = []

    for _, test_key, _ in TEST_CATALOG:
        if test_key not in results:
            continue
        data = results[test_key]
        m = extract_test_metrics(test_key, data)
        if m["status"] == "SKIPPED":
            continue
        total_seen += 1
        # Infrastructure failures (network / timeout / 5xx / 429) measure the
        # serving environment, not the model. Exclude them from effectiveness
        # and surface them via completion_rate instead.
        if "FAIL" in m["status"] and _classify_infra(data):
            infra_excluded += 1
            infra_excluded_tests.append(test_key)
            continue
        evaluated += 1
        if "PASS" in m["status"]:
            passed += 1
            passed_tokens += m["tokens"]
        elif "PARTIAL" in m["status"]:
            partial += 1
            passed_tokens += int(m["tokens"] * 0.5)
        else:
            failed += 1

        total_tokens += m["tokens"]
        if m["tok_s"] > 0:
            speeds.append(m["tok_s"])
        if m["ttft_ms"] > 0:
            ttfts.append(m["ttft_ms"])

    pass_rate_pct = ((passed + 0.5 * partial) / evaluated * 100.0) if evaluated > 0 else 0.0
    completion_rate_pct = (evaluated / total_seen * 100.0) if total_seen > 0 else 100.0
    avg_tokens_per_test = (total_tokens / evaluated) if evaluated > 0 else 0.0
    avg_tokens_per_passed = (passed_tokens / (passed + 0.5 * partial)) if (passed + partial) > 0 else 0.0
    avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else 0.0
    wall_time = report.get("total_suite_wall_time_s", 0.0)

    token_conciseness_factor = min(2.0, max(0.2, 1000.0 / (avg_tokens_per_passed if avg_tokens_per_passed > 0 else 1000.0)))
    efficiency_index = round((pass_rate_pct * 0.6) + (min(100.0, avg_speed * 1.5) * 0.25) + (token_conciseness_factor * 15.0), 1)

    return {
        "model": report.get("model", "unknown"),
        "endpoint": report.get("endpoint", "unknown"),
        "total_evaluated": evaluated,
        "total_seen": total_seen,
        "infra_excluded": infra_excluded,
        "infra_excluded_tests": infra_excluded_tests,
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "effectiveness_rate_pct": round(pass_rate_pct, 1),
        "completion_rate_pct": round(completion_rate_pct, 1),
        "star_rating": star_rating(pass_rate_pct, completion_rate_pct),
        "config_fingerprint": build_config_fingerprint(report),
        "total_tokens_emitted": total_tokens,
        "token_economy_tokens_per_passed_task": round(avg_tokens_per_passed, 1),
        "avg_tokens_per_test": round(avg_tokens_per_test, 1),
        "avg_decode_tok_s": round(avg_speed, 2),
        "avg_ttft_ms": round(avg_ttft, 1),
        "total_wall_time_s": round(wall_time, 2),
        "efficiency_index": efficiency_index
    }


# ============================================================================
# FORMATTED CLI SUMMARY SCORECARD
# ============================================================================

def print_summary_table(report):
    res = report.get("results", {})
    log("\n" + "="*88, bold=True)
    log("              ENTERPRISE LLM SERVER EVALUATION SCORECARD                                ", bold=True, color=CYAN)
    log("="*88, bold=True)
    log(f"  Target Endpoint : {report.get('endpoint')}")
    log(f"  Model Under Test: {report.get('model')}")
    log(f"  Max Context Cap : {report.get('max_context_tokens', 0):,} tokens")
    log(f"  Parallel Setting: {report.get('parallel_streams', 1)} concurrent clients")
    log(f"  Total Wall Time : {report.get('total_suite_wall_time_s', 0):.2f} s")
    summary = report.get("summary", {})
    if summary:
        stars = summary.get("star_rating", 0)
        star_str = "*" * stars + "." * (5 - stars) if stars else "-"
        log(f"  Quality Rating  : {star_str}  ({summary.get('star_rating', 0)}/5)")
        log(f"  Completion Rate : {summary.get('completion_rate_pct', 100.0):.1f}%  ({summary.get('total_evaluated', 0)}/{summary.get('total_seen', 0)} tests scored)")
        if summary.get("infra_excluded", 0) > 0:
            log(f"  Infra Excluded  : {summary.get('infra_excluded', 0)} test(s) excluded (network/timeout/5xx/429) -- not a model signal", color=YELLOW)
    log("="*88)

    header = f"{'Evaluation Domain':<38} | {'Status':<10} | {'Key Metric / Latency':<20} | {'Throughput'}"
    log(header, bold=True)
    log("-" * 88)

    def fmt_status(st, width=10):
        raw = str(st)
        if "PASS" in raw:
            color = GREEN
        elif "FAIL" in raw:
            color = RED
        elif "SKIPPED" in raw:
            color = YELLOW
        else:
            color = YELLOW
        padding = " " * max(0, width - len(raw))
        return f"{color}{raw}{RESET}{padding}"

    sel_tests = set(report.get("selected_tests") or range(1, TOTAL_TESTS + 1))

    # 1. Streaming
    if "streaming" in res:
        s = res.get("streaming", {})
        log(f"{'1. Streaming & Latency':<38} | {fmt_status(s.get('status', 'N/A'))} | TTFT: {s.get('ttft_ms', 0):.1f} ms{'':<6} | {s.get('tok_s', 0):.2f} tok/s")
    elif 1 in sel_tests:
        log(f"{'1. Streaming & Latency':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 2. Vision
    if "vision" in res:
        v = res.get("vision", {})
        v_ttft = f"TTFT: {v.get('ttft_ms', 0):.1f} ms" if 'ttft_ms' in v else "N/A"
        v_speed = f"{v.get('tok_s', 0):.2f} tok/s" if 'tok_s' in v else "N/A"
        log(f"{'2. Multimodal Vision (Invoice)':<38} | {fmt_status(v.get('status', 'N/A'))} | {v_ttft:<20} | {v_speed}")
    elif 2 in sel_tests:
        log(f"{'2. Multimodal Vision (Invoice)':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 3. Parallel Batching
    if "concurrency" in res:
        c = res.get("concurrency", {})
        for ckey, cval in c.items():
            if not isinstance(cval, dict):
                continue
            c_label = f"3. Parallel Batching ({ckey.upper()})"
            c_speed = f"{cval.get('aggregate_tok_s', 0):.2f} tok/s (agg)"
            c_wall = f"{cval.get('wall_time_s', 0):.2f} s wall"
            log(f"{c_label:<38} | {fmt_status(cval.get('status', 'N/A'))} | {c_wall:<20} | {c_speed}")
    elif 3 in sel_tests:
        log(f"{'3. Parallel Batching':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 4. Capabilities
    if "capabilities_4tasks" in res:
        cap = res.get("capabilities_4tasks", {})
        cap_speed = f"{cap.get('average_speed_tok_s', 0):.2f} tok/s (avg)"
        log(f"{'4. 4-Task Architecture Suite':<38} | {fmt_status(cap.get('status', 'N/A'))} | 4/4 tasks passed{'':<5} | {cap_speed}")
    elif 4 in sel_tests:
        log(f"{'4. 4-Task Architecture Suite':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 5. Tool Calling
    if "tool_calling" in res:
        tc = res.get("tool_calling", {})
        tc_desc = "Agent Error Recovered" if tc.get("turn2_recovered") else ("Turn 1 Valid" if tc.get("turn1_valid") else "Args Invalid")
        log(f"{'5. Tool Calling & Agentic Recovery':<38} | {fmt_status(tc.get('status', 'N/A'))} | {tc_desc:<20} | TTFT: {tc.get('ttft_ms', 0):.1f} ms")
    elif 5 in sel_tests:
        log(f"{'5. Tool Calling & Agentic Recovery':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 6. JSON Schema
    if "json_schema" in res:
        js = res.get("json_schema", {})
        js_desc = "Strict Schema Conformed" if js.get("valid_schema") else "Schema Invalid"
        log(f"{'6. JSON Schema Mode (response_format)':<38} | {fmt_status(js.get('status', 'N/A'))} | {js_desc:<20} | {js.get('tok_s', 0):.2f} tok/s")
    elif 6 in sel_tests:
        log(f"{'6. JSON Schema Mode (response_format)':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 7. Prefix Caching
    if "prefix_caching" in res:
        pc = res.get("prefix_caching", {})
        pc_desc = f"{pc.get('speedup_ratio', 1.0):.1f}x speedup" if pc.get("speedup_ratio") else "N/A"
        log(f"{'7. Prefix / KV Cache Reuse':<38} | {fmt_status(pc.get('status', 'N/A'))} | {pc_desc:<20} | Warm: {pc.get('warm_ttft_s', 0):.3f}s")
    elif 7 in sel_tests:
        log(f"{'7. Prefix / KV Cache Reuse':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 8. Client Abort
    if "client_abort" in res:
        ca = res.get("client_abort", {})
        ca_desc = f"Rec: {ca.get('recovery_latency_ms', 0):.1f} ms"
        log(f"{'8. Client Socket Abort Recovery':<38} | {fmt_status(ca.get('status', 'N/A'))} | {ca_desc:<20} | Slots Released")
    elif 8 in sel_tests:
        log(f"{'8. Client Socket Abort Recovery':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 9. Stop Sequences
    if "stop_sequences" in res:
        ss = res.get("stop_sequences", {})
        ss_desc = "Deterministic (temp=0)" if ss.get("deterministic_greedy_reproducible") else "Non-deterministic"
        log(f"{'9. Stop Words & Greedy Sampling':<38} | {fmt_status(ss.get('status', 'N/A'))} | {ss_desc:<20} | Tokens Suppressed")
    elif 9 in sel_tests:
        log(f"{'9. Stop Words & Greedy Sampling':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 10. Extreme Precision
    if "extreme_precision" in res:
        ep = res.get("extreme_precision", {})
        ep_desc = "Exact Matched" if ep.get("matched") else "Balance Diverged"
        log(f"{'10. Precision Ledger Reconcile':<38} | {fmt_status(ep.get('status', 'N/A'))} | {ep_desc:<20} | {ep.get('tok_s', 0):.2f} tok/s")
    elif 10 in sel_tests:
        log(f"{'10. Precision Ledger Reconcile':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 11. Error Handling
    if "error_handling" in res:
        eh = res.get("error_handling", {})
        eh_desc = "HTTP 400/422 Standard" if eh.get("bad_schema_rejected") else "Non-standard error"
        log(f"{'11. API Error Protocol Compliance':<38} | {fmt_status(eh.get('status', 'N/A'))} | {eh_desc:<20} | Protocol OK")
    elif 11 in sel_tests:
        log(f"{'11. API Error Protocol Compliance':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 12. Context Scaling Summary
    if "context_scaling" in res:
        cs = res.get("context_scaling", [])
        if cs:
            log("-" * 88)
            log("  [12. Context Scaling Milestone Performance & Accuracy Breakdown]", bold=True)
            cs_hdr = f"  {'Context Target':<18} | {'Cached':<8} | {'Prefill TTFT':<14} | {'Cold Speed':<14} | {'Effective':<14} | {'Decode':<12} | {'Accuracy':<10}"
            log(cs_hdr)
            log("  " + "-" * (len(cs_hdr) - 2))
            for step in cs:
                if step.get("status") in ("PASS", "PARTIAL"):
                    m_label = f"~{step.get('target_tokens', 0)//1000}k ({step.get('actual_prompt_tokens', 0):,} toks)"
                    cached_str = f"{step.get('cached_tokens', 0):,}"
                    ttft_str = f"{step.get('ttft_s', 0):.2f} s"
                    cold_str = f"{step.get('cold_prefill_tok_s', 0):.1f} tok/s"
                    eff_str = f"{step.get('effective_prefill_tok_s', 0):.1f} tok/s"
                    decode_str = f"{step.get('decode_tok_s', 0):.2f} tok/s"
                    acc_str = "RECALLED" if step.get("needle_matched") else "MISSED"
                    log(f"  {m_label:<18} | {cached_str:<8} | {ttft_str:<14} | {cold_str:<14} | {eff_str:<14} | {decode_str:<12} | {acc_str:<10}")
                else:
                    m_label = f"{step.get('target_tokens', 0):,} toks"
                    log(f"  {m_label:<18} | {'-':<8} | {'FAILED':<14} | {str(step.get('error', 'Error'))[:28]}")
    elif 12 in sel_tests:
        log(f"{'12. Dynamic Context Scaling':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 13. Multihop Graph Traversal
    if "multihop_graph" in res:
        mh = res.get("multihop_graph", {})
        mh_desc = f"{mh.get('completion_tokens', 0)} tokens"
        log(f"{'13. Adversarial Graph & Distractors':<38} | {fmt_status(mh.get('status', 'N/A'))} | {mh_desc:<20} | {mh.get('tok_s', 0):.2f} tok/s")
    elif 13 in sel_tests:
        log(f"{'13. Adversarial Graph & Distractors':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 14. Novel Algorithmic Fuzz
    if "novel_algorithm_fuzz" in res:
        fz = res.get("novel_algorithm_fuzz", {})
        fz_desc = "5,000 Ops Fuzzed OK" if fz.get("status") == "PASS" else "Fuzz Assertion Failed"
        log(f"{'14. Novel Algorithm (5k Fuzz)':<38} | {fmt_status(fz.get('status', 'N/A'))} | {fz_desc:<20} | {fz.get('tok_s', 0):.2f} tok/s")
    elif 14 in sel_tests:
        log(f"{'14. Novel Algorithm (5k Fuzz)':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 15. Frontier Needle Depth
    if "frontier_needle_depth" in res:
        fn = res.get("frontier_needle_depth", {})
        fn_desc = "Composite Checksum OK" if fn.get("status") == "PASS" else "Checksum Missed"
        log(f"{'15. Frontier Depth Multi-Needle':<38} | {fmt_status(fn.get('status', 'N/A'))} | {fn_desc:<20} | {fn.get('tok_s', 0):.2f} tok/s")
    elif 15 in sel_tests:
        log(f"{'15. Frontier Depth Multi-Needle':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 16. CruxEval
    if "cruxeval" in res:
        cx = res.get("cruxeval", {})
        cx_desc = "State Traversed OK" if cx.get("status") == "PASS" else "State Diverged"
        log(f"{'16. CruxEval (Mental Code Exec)':<38} | {fmt_status(cx.get('status', 'N/A'))} | {cx_desc:<20} | {cx.get('tok_s', 0):.2f} tok/s")
    elif 16 in sel_tests:
        log(f"{'16. CruxEval (Mental Code Exec)':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 17. SWE-bench Bug Patching
    if "swe_bench_bug_patch" in res:
        sw = res.get("swe_bench_bug_patch", {})
        sw_desc = "Regression Tests OK" if sw.get("status") == "PASS" else "Unit Tests Failed"
        log(f"{'17. SWE-bench (Traceback Fix)':<38} | {fmt_status(sw.get('status', 'N/A'))} | {sw_desc:<20} | {sw.get('tok_s', 0):.2f} tok/s")
    elif 17 in sel_tests:
        log(f"{'17. SWE-bench (Traceback Fix)':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 18. AIME Olympiad Math
    if "aime_olympiad" in res:
        am = res.get("aime_olympiad", {})
        am_desc = "Exact Modular Root" if am.get("status") == "PASS" else "Arithmetic Missed"
        log(f"{'18. AIME Olympiad (Math Reasoning)':<38} | {fmt_status(am.get('status', 'N/A'))} | {am_desc:<20} | {am.get('tok_s', 0):.2f} tok/s")
    elif 18 in sel_tests:
        log(f"{'18. AIME Olympiad (Math Reasoning)':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 19. GPQA Diamond Graduate Science
    if "gpqa_diamond" in res:
        gp = res.get("gpqa_diamond", {})
        score_val = gp.get("score") or f"{gp.get('passed_count', 0)}/3"
        gp_desc = f"Score: {score_val}"
        log(f"{'19. GPQA Diamond (Graduate Science)':<38} | {fmt_status(gp.get('status', 'N/A'))} | {gp_desc:<20} | {gp.get('tok_s', 0):.2f} tok/s")
    elif 19 in sel_tests:
        log(f"{'19. GPQA Diamond (Graduate Science)':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 20. Surgical Unified Diff (git apply)
    if "surgical_git_diff" in res:
        gd = res.get("surgical_git_diff", {})
        lang = gd.get("language") or report.get("language", "python")
        tag = f" [{lang.upper()}]" if lang and lang != "python" else ""
        if lang == "polyglot":
            gd_desc = f"{gd.get('polyglot_pass_count', 0)}/6 Langs OK"
        else:
            gd_desc = "git apply --check OK" if gd.get("git_apply_passed") else ("Unified Diff Valid" if gd.get("valid_diff_syntax") else "Invalid Patch")
        log(f"{f'20. Surgical Unified Diff{tag}':<38} | {fmt_status(gd.get('status', 'N/A'))} | {gd_desc:<20} | {gd.get('tok_s', 0):.2f} tok/s")
    elif 20 in sel_tests:
        log(f"{'20. Surgical Unified Diff':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 21. Multi-File Cross-Reference & DI
    if "multifile_context" in res:
        mc = res.get("multifile_context", {})
        lang = mc.get("language") or report.get("language", "python")
        tag = f" [{lang.upper()}]" if lang and lang != "python" else ""
        if lang == "polyglot":
            mc_desc = f"{mc.get('polyglot_pass_count', 0)}/6 Langs OK"
        else:
            mc_desc = "3/3 Modules Aligned" if mc.get("status") == "PASS" else f"Symbols: {mc.get('symbols_resolved', 0)}/{mc.get('total_symbols', 3)}"
        log(f"{f'21. Multi-File Cross-Ref & DI{tag}':<38} | {fmt_status(mc.get('status', 'N/A'))} | {mc_desc:<20} | {mc.get('tok_s', 0):.2f} tok/s")
    elif 21 in sel_tests:
        log(f"{'21. Multi-File Cross-Ref & DI':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 22. Security Review & OWASP Audit
    if "code_review_security" in res:
        sec = res.get("code_review_security", {})
        lang = sec.get("language") or report.get("language", "python")
        tag = f" [{lang.upper()}]" if lang and lang != "python" else ""
        if lang == "polyglot":
            sec_desc = f"{sec.get('polyglot_pass_count', 0)}/6 Langs OK"
        else:
            sec_desc = "SQLi & Race Detected" if sec.get("status") == "PASS" else "Vulnerabilities Missed"
        log(f"{f'22. Security Review & OWASP Audit{tag}':<38} | {fmt_status(sec.get('status', 'N/A'))} | {sec_desc:<20} | {sec.get('tok_s', 0):.2f} tok/s")
    elif 22 in sel_tests:
        log(f"{'22. Security Review & OWASP Audit':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 23. Unit Test Synthesis with Mocks
    if "unit_test_synthesis" in res:
        ut = res.get("unit_test_synthesis", {})
        lang = ut.get("language") or report.get("language", "python")
        tag = f" [{lang.upper()}]" if lang and lang != "python" else ""
        pass_n = ut.get("unit_tests_passed_count", 0)
        total_n = ut.get("unit_tests_total_count", 0)
        if lang == "polyglot":
            ut_desc = f"{ut.get('polyglot_pass_count', 0)}/6 Langs OK"
        else:
            ut_desc = f"{pass_n}/{total_n} Mock Tests OK" if ut.get("status") == "PASS" else "Tests Failed"
        log(f"{f'23. Unit Test Synthesis{tag}':<38} | {fmt_status(ut.get('status', 'N/A'))} | {ut_desc:<20} | {ut.get('tok_s', 0):.2f} tok/s")
    elif 23 in sel_tests:
        log(f"{'23. Unit Test Synthesis':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 24. Strict Type Compliance (mypy)
    if "strict_typing" in res:
        st_data = res.get("strict_typing", {})
        lang = st_data.get("language") or report.get("language", "python")
        tag = f" [{lang.upper()}]" if lang and lang != "python" else ""
        if lang == "polyglot":
            st_desc = f"{st_data.get('polyglot_pass_count', 0)}/6 Langs OK"
        else:
            st_desc = "0 Errors (mypy strict)" if st_data.get("mypy_verified") else ("100% Annotated" if st_data.get("status") == "PASS" else "Type Errors")
        log(f"{f'24. Strict Type Compliance{tag}':<38} | {fmt_status(st_data.get('status', 'N/A'))} | {st_desc:<20} | {st_data.get('tok_s', 0):.2f} tok/s")
    elif 24 in sel_tests:
        log(f"{'24. Strict Type Compliance':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 25. Safe Refactor Invariant Check
    if "refactor_invariants" in res:
        rf = res.get("refactor_invariants", {})
        lang = rf.get("language") or report.get("language", "python")
        tag = f" [{lang.upper()}]" if lang and lang != "python" else ""
        if lang == "polyglot":
            rf_desc = f"{rf.get('polyglot_pass_count', 0)}/6 Langs OK"
        else:
            rf_desc = "Regression Suite OK" if rf.get("status") == "PASS" else "Behavior Diverged"
        log(f"{f'25. Safe Refactor Invariants{tag}':<38} | {fmt_status(rf.get('status', 'N/A'))} | {rf_desc:<20} | {rf.get('tok_s', 0):.2f} tok/s")
    elif 25 in sel_tests:
        log(f"{'25. Safe Refactor Invariants':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    # 26. Prompt Injection & Delimiter Defense
    if "prompt_injection_defense" in res:
        pi = res.get("prompt_injection_defense", {})
        pi_desc = "Invariants Protected" if pi.get("status") == "PASS" else "Security Hijacked"
        log(f"{'26. Prompt Injection Defense':<38} | {fmt_status(pi.get('status', 'N/A'))} | {pi_desc:<20} | {pi.get('tok_s', 0):.2f} tok/s")
    elif 26 in sel_tests:
        log(f"{'26. Prompt Injection Defense':<38} | {fmt_status('SKIPPED')} | {'-':<20} | -")

    log("="*88 + "\n")

    # Polyglot SWE Benchmark Matrix (if polyglot mode was run)
    has_poly = report.get("language") == "polyglot" or any(res.get(k, {}).get("language") == "polyglot" for k in ["surgical_git_diff", "multifile_context", "code_review_security", "unit_test_synthesis", "strict_typing", "refactor_invariants"])
    if has_poly:
        log("="*88, bold=True)
        log(" POLYGLOT SWE BENCHMARK MATRIX (Tests 20-25 across 6 Industry Toolchains) ", bold=True, color=MAGENTA)
        log("="*88)
        header_poly = f"{'Language':<12} | {'Diff(20)':<10} | {'Multi(21)':<10} | {'Sec(22)':<10} | {'Test(23)':<10} | {'Type(24)':<10} | {'Refactor(25)':<12} | {'Pass Rate'}"
        log(header_poly, bold=True)
        log("-" * 88)
        poly_langs = ["python", "typescript", "go", "java", "rust", "csharp"]
        swe_keys = ["surgical_git_diff", "multifile_context", "code_review_security", "unit_test_synthesis", "strict_typing", "refactor_invariants"]
        for pl in poly_langs:
            row_items = []
            for sk in swe_keys:
                sub = res.get(f"{sk}_{pl}")
                if sub:
                    st_val = sub.get("status", "N/A")
                elif sk in res and res[sk].get("language") == pl:
                    st_val = res[sk].get("status", "N/A")
                elif sk in res and res[sk].get("language") == "polyglot":
                    det = res[sk].get("language_details", {})
                    st_val = det.get(pl, "N/A")
                else:
                    st_val = "N/A"
                row_items.append(st_val)
            passes = row_items.count("PASS")
            evaluated = [s for s in row_items if s != "N/A"]
            rate = f"{passes}/{len(evaluated)} ({passes/len(evaluated)*100:.0f}%)" if evaluated else "-"
            l_disp = {"python": "Python", "typescript": "TypeScript", "go": "Go", "java": "Java", "rust": "Rust", "csharp": "C#"}.get(pl, pl.title())
            cols = " | ".join(fmt_status(s, w) for s, w in zip(row_items, [10, 10, 10, 10, 10, 12]))
            log(f"{l_disp:<12} | {cols} | {rate}")
        log("="*88 + "\n")

    # Executive Efficiency & Effectiveness Scorecard
    summary = compute_executive_summary(report)
    report["summary"] = summary

    log("="*88, bold=True)
    log("             EXECUTIVE EFFICIENCY & EFFECTIVENESS SCORECARD                             ", bold=True, color=CYAN)
    log("="*88, bold=True)
    log(f"  Model Under Test        : {summary['model']}")
    log(f"  Effectiveness (Pass@1)  : {summary['passed']}/{summary['total_evaluated']} Passed ({summary['effectiveness_rate_pct']}%)")
    log(f"  Total Solution Tokens   : {summary['total_tokens_emitted']:,} tokens consumed")
    log(f"  Token Economy           : {summary['token_economy_tokens_per_passed_task']:.1f} tokens/passed victory (Solution conciseness)")
    log(f"  Mean Decode Throughput  : {summary['avg_decode_tok_s']:.2f} tok/s")
    log(f"  Mean Prefill TTFT       : {summary['avg_ttft_ms']:.1f} ms")
    log(f"  Total Benchmark Time    : {summary['total_wall_time_s']:.2f} s")
    log(f"  Composite Efficiency Idx: {summary['efficiency_index']:.1f} / 100.0")
    log("="*88 + "\n", bold=True)


def compare_benchmark_reports(file_paths: list, output_markdown: str = None):
    """Compares two or more JSON benchmark reports head-to-head for efficiency and effectiveness."""
    if len(file_paths) < 2:
        log("Error: --compare requires at least two JSON benchmark report paths.", color=RED)
        sys.exit(1)

    reports = []
    summaries = []
    for fp in file_paths:
        if not os.path.exists(fp):
            log(f"Error: Report file not found: {fp}", color=RED)
            sys.exit(1)
        try:
            with open(fp, "r") as f:
                data = json.load(f)
                norm_data = normalize_report(data)
                reports.append(norm_data)
                summaries.append(compute_executive_summary(norm_data))
        except Exception as e:
            log(f"Error reading {fp}: {e}", color=RED)
            sys.exit(1)

    log("\n" + "="*100, bold=True)
    log("          CROSS-MODEL HEAD-TO-HEAD EFFICIENCY & EFFECTIVENESS BENCHMARK REPORT          ", bold=True, color=CYAN)
    log("="*100, bold=True)

    if len(summaries) == 2:
        m1, m2 = summaries[0], summaries[1]
        name1 = f"{m1['model']} ({m1['endpoint']})"
        name2 = f"{m2['model']} ({m2['endpoint']})"

        log(f"  Model A : {name1}", bold=True)
        log(f"  Model B : {name2}\n", bold=True)

        # Comparability check: warn if the two runs used different configs so a
        # context/parallel/milestone difference isn't mistaken for a model difference.
        fp1 = m1.get("config_fingerprint")
        fp2 = m2.get("config_fingerprint")
        if fp1 and fp2 and fp1 != fp2:
            log("  ⚠ COMPARABILITY WARNING: the two runs have different config fingerprints", color=YELLOW, bold=True)
            log("    (model / context / parallel / milestones / suite differ). Treat results as not directly comparable.", color=YELLOW)
            log("")

        exec_hdr = f"{'Core Performance Metric':<35} | {'Model A (' + str(m1['model'])[:16] + ')':<26} | {'Model B (' + str(m2['model'])[:16] + ')':<26} | {'Advantage'}"
        log(exec_hdr, bold=True)
        log("-" * 100)

        adv = compute_relative_advantages(m1, m2)

        # 1. Effectiveness
        eff_a = m1['effectiveness_rate_pct']
        eff_b = m2['effectiveness_rate_pct']
        pass_a_str = f"{m1['passed']}/{m1['total_evaluated']} ({eff_a:.1f}%)"
        pass_b_str = f"{m2['passed']}/{m2['total_evaluated']} ({eff_b:.1f}%)"
        log(f"{'Effectiveness (Pass Rate)':<35} | {pass_a_str:<26} | {pass_b_str:<26} | {adv['adv_eff']}")

        # 2. Token Economy
        te_a = m1['token_economy_tokens_per_passed_task']
        te_b = m2['token_economy_tokens_per_passed_task']
        te_a_str = f"{te_a:.1f} tokens/task"
        te_b_str = f"{te_b:.1f} tokens/task"
        log(f"{'Token Economy (Tokens/Victory)':<35} | {te_a_str:<26} | {te_b_str:<26} | {adv['adv_te']}")

        # 3. Total Tokens Emitted
        tot_a = m1['total_tokens_emitted']
        tot_b = m2['total_tokens_emitted']
        tot_a_str = f"{tot_a:,} tokens"
        tot_b_str = f"{tot_b:,} tokens"
        log(f"{'Total Solution Tokens Consumed':<35} | {tot_a_str:<26} | {tot_b_str:<26} | {adv['adv_tot']}")

        # 4. Generation Speed
        spd_a = m1['avg_decode_tok_s']
        spd_b = m2['avg_decode_tok_s']
        spd_a_str = f"{spd_a:.2f} tok/s"
        spd_b_str = f"{spd_b:.2f} tok/s"
        log(f"{'Mean Decode Speed (Throughput)':<35} | {spd_a_str:<26} | {spd_b_str:<26} | {adv['adv_spd']}")

        # 5. First Token Latency (TTFT)
        ttft_a = m1['avg_ttft_ms']
        ttft_b = m2['avg_ttft_ms']
        ttft_a_str = f"{ttft_a:.1f} ms"
        ttft_b_str = f"{ttft_b:.1f} ms"
        log(f"{'Mean Time-To-First-Token (TTFT)':<35} | {ttft_a_str:<26} | {ttft_b_str:<26} | {adv['adv_ttft']}")

        # 6. Total Wall Clock Time
        wall_a = m1['total_wall_time_s']
        wall_b = m2['total_wall_time_s']
        wall_a_str = f"{wall_a:.2f} s"
        wall_b_str = f"{wall_b:.2f} s"
        log(f"{'Total Benchmark Suite Time':<35} | {wall_a_str:<26} | {wall_b_str:<26} | {adv['adv_wall']}")

        # 7. Composite Efficiency Index
        ei_a = m1['efficiency_index']
        ei_b = m2['efficiency_index']
        ei_a_str = f"{ei_a:.1f} / 100"
        ei_b_str = f"{ei_b:.1f} / 100"
        log(f"{'Composite Efficiency Index':<35} | {ei_a_str:<26} | {ei_b_str:<26} | {adv['adv_ei']}")
        log("="*100)

        # Detailed Domain Matrix
        log("\n" + "="*100, bold=True)
        log("                          DOMAIN-BY-DOMAIN HEAD-TO-HEAD MATRIX                                ", bold=True, color=CYAN)
        log("="*100, bold=True)
        dom_hdr = f"{'Domain':<32} | {'Model A (' + str(m1['model'])[:12] + ')':<26} | {'Model B (' + str(m2['model'])[:12] + ')':<26} | {'Outcome / Advantage'}"
        log(dom_hdr, bold=True)
        log("-" * 100)

        res_a = reports[0].get("results") or {}
        res_b = reports[1].get("results") or {}
        table_rows = []

        for num, key, name in TEST_CATALOG:
            da = extract_test_metrics(key, res_a.get(key))
            db = extract_test_metrics(key, res_b.get(key))

            if da["status"] == "SKIPPED" and db["status"] == "SKIPPED":
                continue

            label = f"{num}. {name[:28]}"
            sa_str = f"{da['status']} ({da['tokens']}t | {da['tok_s']}t/s)" if da["status"] != "SKIPPED" else "SKIPPED"
            sb_str = f"{db['status']} ({db['tokens']}t | {db['tok_s']}t/s)" if db["status"] != "SKIPPED" else "SKIPPED"

            adv_domain = "-"
            if "PASS" in da["status"] and "PASS" not in db["status"]:
                adv_domain = "Model A Victory"
            elif "PASS" in db["status"] and "PASS" not in da["status"]:
                adv_domain = "Model B Victory"
            elif "PASS" in da["status"] and "PASS" in db["status"]:
                if da["tokens"] > 0 and db["tokens"] > 0:
                    if da["tokens"] < db["tokens"]:
                        ratio = db["tokens"] / da["tokens"]
                        adv_domain = f"Model A ({ratio:.1f}x fewer tokens)"
                    elif db["tokens"] < da["tokens"]:
                        ratio = da["tokens"] / db["tokens"]
                        adv_domain = f"Model B ({ratio:.1f}x fewer tokens)"
                    else:
                        adv_domain = "Tied"
                else:
                    adv_domain = "Both Passed"
            elif "FAIL" in da["status"] and "FAIL" in db["status"]:
                adv_domain = "Both Failed"

            log(f"{label:<32} | {sa_str:<26} | {sb_str:<26} | {adv_domain}")
            table_rows.append((num, name, da, db, adv_domain))

        log("="*100)

        # Executive Verdict
        log("\n" + "="*100, bold=True)
        log("                                  EXECUTIVE VERDICT                                     ", bold=True, color=GREEN)
        log("="*100, bold=True)
        verdict = adv["verdict"]
        log(f"  {verdict}\n", bold=True)
        log("="*100 + "\n")

        # Generate Markdown Report
        safe_a = re.sub(r"[^\w\-]", "_", str(m1["model"]))
        safe_b = re.sub(r"[^\w\-]", "_", str(m2["model"]))
        md_file = output_markdown or os.path.join(DEFAULT_RESULTS_DIR, f"comparison_{safe_a}_vs_{safe_b}_{time.strftime('%Y%m%d_%H%M%S')}.md")
        md_dir = os.path.dirname(md_file)
        if md_dir:
            os.makedirs(md_dir, exist_ok=True)

        md_lines = [
            "# Cross-Model Head-to-Head Efficiency & Effectiveness Benchmark Report",
            "",
            f"- **Date Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Model A**: `{name1}`",
            f"- **Model B**: `{name2}`",
            "",
            "## Executive Summary & Efficiency Index",
            "",
            "| Metric / Dimension | Model A (`" + str(m1['model']) + "`) | Model B (`" + str(m2['model']) + "`) | Relative Advantage |",
            "|:---|:---|:---|:---|",
            f"| **Effectiveness (Accuracy)** | `{m1['passed']}/{m1['total_evaluated']} ({eff_a:.1f}%)` | `{m2['passed']}/{m2['total_evaluated']} ({eff_b:.1f}%)` | **{adv['adv_eff']}** |",
            f"| **Token Economy (Solution Conciseness)** | `{te_a:.1f} tokens/task` | `{te_b:.1f} tokens/task` | **{adv['adv_te']}** |",
            f"| **Total Solution Tokens Consumed** | `{tot_a:,} tokens` | `{tot_b:,} tokens` | **{adv['adv_tot']}** |",
            f"| **Mean Decode Throughput** | `{spd_a:.2f} tok/s` | `{spd_b:.2f} tok/s` | **{adv['adv_spd']}** |",
            f"| **Mean Time-to-First-Token (TTFT)** | `{ttft_a:.1f} ms` | `{ttft_b:.1f} ms` | **{adv['adv_ttft']}** |",
            f"| **Total Benchmark Wall Time** | `{wall_a:.2f} s` | `{wall_b:.2f} s` | **{adv['adv_wall']}** |",
            f"| **Composite Efficiency Index** | `{ei_a:.1f} / 100` | `{ei_b:.1f} / 100` | **{adv['adv_ei']}** |",
            "",
            "## Domain-by-Domain Comparison Matrix",
            "",
            "| Stage | Evaluation Domain | Model A (`" + str(m1['model']) + "`) | Model B (`" + str(m2['model']) + "`) | Outcome / Advantage |",
            "|:---|:---|:---|:---|:---|",
        ]

        for num, name, da, db, adv_row in table_rows:
            sa_str = f"{da['status']} ({da['tokens']}t, {da['tok_s']}t/s)" if da["status"] != "SKIPPED" else "SKIPPED"
            sb_str = f"{db['status']} ({db['tokens']}t, {db['tok_s']}t/s)" if db["status"] != "SKIPPED" else "SKIPPED"
            md_lines.append(f"| {num} | {name} | {sa_str} | {sb_str} | **{adv_row}** |")

        md_lines.extend([
            "",
            "## Strategic Verdict & Deployment Recommendation",
            "",
            f"> **Executive Verdict**: {verdict}",
            ""
        ])

        with open(md_file, "w") as f:
            f.write("\n".join(md_lines))

        log(f"  Markdown Benchmark Comparison Saved: {md_file}\n", color=GREEN, bold=True)
    else:
        # N-Way Comparison (N >= 3)
        n = len(summaries)
        log(f"  Evaluating {n} Models in Multi-Endpoint Matrix:\n", bold=True)
        for i, s in enumerate(summaries, 1):
            log(f"  Model {i}: {s['model']} ({s['endpoint']})")
        log("")

        col_w = max(18, 90 // (n + 1))
        hdr = f"{'Core Performance Metric':<32} | " + " | ".join(f"M{i} ({str(s['model'])[:col_w-6]})".ljust(col_w) for i, s in enumerate(summaries, 1))
        log(hdr, bold=True)
        log("-" * len(hdr))

        def fmt_row(metric_name, vals):
            row = f"{metric_name:<32} | " + " | ".join(f"{str(v):<{col_w}}" for v in vals)
            log(row)

        fmt_row("Effectiveness (Pass Rate)", [f"{s['passed']}/{s['total_evaluated']} ({s['effectiveness_rate_pct']}%)" for s in summaries])
        fmt_row("Token Economy (tok/task)", [f"{s['token_economy_tokens_per_passed_task']:.1f}" for s in summaries])
        fmt_row("Total Solution Tokens", [f"{s['total_tokens_emitted']:,}" for s in summaries])
        fmt_row("Mean Decode Speed (tok/s)", [f"{s['avg_decode_tok_s']:.2f}" for s in summaries])
        fmt_row("Mean TTFT (ms)", [f"{s['avg_ttft_ms']:.1f}" for s in summaries])
        fmt_row("Total Wall Time (s)", [f"{s['total_wall_time_s']:.2f}" for s in summaries])
        fmt_row("Composite Efficiency Idx", [f"{s['efficiency_index']:.1f}" for s in summaries])
        log("=" * len(hdr))

        # Domain by Domain
        log("\n" + "="*len(hdr), bold=True)
        log("                 MULTI-MODEL DOMAIN MATRIX                                 ", bold=True, color=CYAN)
        log("="*len(hdr), bold=True)
        d_hdr = f"{'Domain':<32} | " + " | ".join(f"M{i}".ljust(col_w) for i in range(1, n + 1))
        log(d_hdr, bold=True)
        log("-" * len(d_hdr))

        all_res = [r.get("results") or {} for r in reports]
        matrix_rows = []
        for tnum, tkey, tname in TEST_CATALOG:
            metrics_row = [extract_test_metrics(tkey, r.get(tkey)) for r in all_res]
            if all(m["status"] == "SKIPPED" for m in metrics_row):
                continue
            cells = [f"{m['status']} ({m['tokens']}t)" if m["status"] != "SKIPPED" else "SKIPPED" for m in metrics_row]
            log(f"{f'{tnum}. {tname[:28]}':<32} | " + " | ".join(f"{c:<{col_w}}" for c in cells))
            matrix_rows.append((tnum, tname, metrics_row))

        log("=" * len(hdr) + "\n")

        # Markdown Report for N-way
        md_file = output_markdown or os.path.join(DEFAULT_RESULTS_DIR, f"comparison_{n}_models_{time.strftime('%Y%m%d_%H%M%S')}.md")
        md_dir = os.path.dirname(md_file)
        if md_dir:
            os.makedirs(md_dir, exist_ok=True)

        md_cols = " | ".join(f"Model {i} (`{s['model']}`)" for i, s in enumerate(summaries, 1))
        md_sep = " | ".join([":---"] * (n + 1))
        md_lines = [
            f"# Multi-Model Benchmark Comparison Report ({n} Models)",
            "",
            f"- **Date Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        for i, s in enumerate(summaries, 1):
            md_lines.append(f"- **Model {i}**: `{s['model']}` (`{s['endpoint']}`)")
        md_lines.extend([
            "",
            "## Executive Summary & Efficiency Matrix",
            "",
            f"| Metric / Dimension | {md_cols} |",
            f"|{md_sep}|",
            "| **Effectiveness (Pass Rate)** | " + " | ".join(f"`{s['passed']}/{s['total_evaluated']} ({s['effectiveness_rate_pct']}%)`" for s in summaries) + " |",
            "| **Token Economy (tok/task)** | " + " | ".join(f"`{s['token_economy_tokens_per_passed_task']:.1f}`" for s in summaries) + " |",
            "| **Total Solution Tokens** | " + " | ".join(f"`{s['total_tokens_emitted']:,}`" for s in summaries) + " |",
            "| **Mean Decode Speed** | " + " | ".join(f"`{s['avg_decode_tok_s']:.2f} tok/s`" for s in summaries) + " |",
            "| **Mean TTFT** | " + " | ".join(f"`{s['avg_ttft_ms']:.1f} ms`" for s in summaries) + " |",
            "| **Total Benchmark Wall Time** | " + " | ".join(f"`{s['total_wall_time_s']:.2f} s`" for s in summaries) + " |",
            "| **Composite Efficiency Index** | " + " | ".join(f"`{s['efficiency_index']:.1f} / 100`" for s in summaries) + " |",
            "",
            "## Domain-by-Domain Performance Matrix",
            "",
            f"| # | Domain | {md_cols} |",
            f"|:---|:---|{md_sep[5:]}|",
        ])
        for tnum, tname, m_row in matrix_rows:
            cell_strs = [f"`{m['status']} ({m['tokens']}t, {m['tok_s']}t/s)`" if m["status"] != "SKIPPED" else "`SKIPPED`" for m in m_row]
            md_lines.append(f"| {tnum} | {tname} | " + " | ".join(cell_strs) + " |")

        with open(md_file, "w") as f:
            f.write("\n".join(md_lines))

        log(f"  Markdown Multi-Model Comparison Saved: {md_file}\n", color=GREEN, bold=True)


NAME_TO_TEST_NUM = {
    "streaming": 1,
    "vision": 2,
    "concurrency": 3,
    "parallel": 3,
    "batching": 3,
    "capabilities": 4,
    "capabilities_4tasks": 4,
    "tasks": 4,
    "tool_calling": 5,
    "tools": 5,
    "json_schema": 6,
    "schema": 6,
    "prefix_caching": 7,
    "caching": 7,
    "client_abort": 8,
    "abort": 8,
    "stop_sequences": 9,
    "stop": 9,
    "extreme_precision": 10,
    "precision": 10,
    "error_handling": 11,
    "error": 11,
    "context_scaling": 12,
    "context": 12,
    "scale": 12,
    "multihop_graph": 13,
    "multihop": 13,
    "graph": 13,
    "distractors": 13,
    "novel_algorithm_fuzz": 14,
    "novel_algorithm": 14,
    "fuzz": 14,
    "ring_buffer": 14,
    "frontier_needle_depth": 15,
    "frontier_needle": 15,
    "cruxeval": 16,
    "code_exec_simulation": 16,
    "swe_bench": 17,
    "swe_bench_bug_patch": 17,
    "bug_patch": 17,
    "patching": 17,
    "rate_limiter": 17,
    "aime": 18,
    "aime_olympiad": 18,
    "olympiad": 18,
    "math_olympiad": 18,
    "gpqa": 19,
    "gpqa_diamond": 19,
    "diamond": 19,
    "surgical_git_diff": 20,
    "git_diff": 20,
    "diff": 20,
    "patch": 20,
    "git": 20,
    "multifile_context": 21,
    "multifile": 21,
    "di": 21,
    "dependency_injection": 21,
    "cross_reference": 21,
    "code_review_security": 22,
    "security": 22,
    "security_review": 22,
    "owasp": 22,
    "sqli": 22,
    "unit_test_synthesis": 23,
    "unit_test": 23,
    "unit_tests": 23,
    "mocking": 23,
    "test_gen": 23,
    "strict_typing": 24,
    "typing": 24,
    "mypy": 24,
    "types": 24,
    "type_check": 24,
    "refactor_invariants": 25,
    "refactor": 25,
    "refactoring": 25,
    "invariants": 25,
    "prompt_injection_defense": 26,
    "prompt_injection": 26,
    "injection": 26,
    "jailbreak": 26,
    "security_defense": 26,
}

def _parse_test_tokens(arg_str: str) -> set:
    if not arg_str:
        return set()
    result = set()
    parts = [p.strip() for p in str(arg_str).replace(" ", ",").split(",") if p.strip()]
    for part in parts:
        if "-" in part and not part.startswith("-"):
            subparts = part.split("-", 1)
            if subparts[0].isdigit() and subparts[1].isdigit():
                start_n, end_n = int(subparts[0]), int(subparts[1])
                for n in range(start_n, end_n + 1):
                    if 1 <= n <= TOTAL_TESTS:
                        result.add(n)
                continue
        if part.isdigit():
            n = int(part)
            if 1 <= n <= TOTAL_TESTS:
                result.add(n)
        else:
            lower = part.lower().replace("-", "_")
            if lower in NAME_TO_TEST_NUM:
                result.add(NAME_TO_TEST_NUM[lower])
            elif lower == "serving":
                result.update({1, 2, 3, 7, 8, 11})
            elif lower == "contract":
                result.update({5, 6, 9, 10})
            elif lower == "context":
                result.update({12, 13, 15})
            elif lower == "swe":
                result.update({4, 14, 17, 20, 21, 22, 23, 24, 25})
            elif lower == "reasoning":
                result.update({16, 18, 19, 26})
            elif lower == "flagship":
                result.update(range(1, 13))
            elif lower in ("adversarial", "hardened"):
                result.update(range(13, 16))
            elif lower in ("frontier", "sota"):
                result.update(range(16, 20))
            elif lower == "all":
                result.update(range(1, TOTAL_TESTS + 1))
            else:
                log(f"Warning: Unknown test identifier '{part}'. Ignored.", color=YELLOW)
    return result


def parse_selected_tests(test_arg: str = None, suite: str = "all", exclude_arg: str = None):
    if test_arg:
        selected = _parse_test_tokens(test_arg)
    elif suite == "serving":
        selected = {1, 2, 3, 7, 8, 11}
    elif suite == "contract":
        selected = {5, 6, 9, 10}
    elif suite == "context":
        selected = {12, 13, 15}
    elif suite == "swe":
        selected = {4, 14, 17, 20, 21, 22, 23, 24, 25}
    elif suite == "reasoning":
        selected = {16, 18, 19, 26}
    elif suite == "flagship":
        selected = set(range(1, 13))
    elif suite == "adversarial":
        selected = set(range(13, 16))
    elif suite == "frontier":
        selected = set(range(16, 20))
    else:  # "all"
        selected = set(range(1, TOTAL_TESTS + 1))

    if exclude_arg:
        excluded = _parse_test_tokens(exclude_arg)
        selected = selected - excluded

    return selected


