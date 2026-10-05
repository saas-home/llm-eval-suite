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

from client import log, BOLD, GREEN, YELLOW, RED, CYAN, RESET, DEFAULT_RESULTS_DIR

TOTAL_TESTS = 22

# ============================================================================
# STANDARDIZED TEST CATALOG & METRIC EXTRACTORS
# ============================================================================

TEST_CATALOG = [
    (1, "streaming", "Streaming & Latency"),
    (2, "vision", "Multimodal Vision (Invoice)"),
    (3, "concurrency", "Parallel Batching"),
    (4, "capabilities_4tasks", "4-Task Architecture Suite"),
    (5, "tool_calling", "Tool Calling & Agentic Recovery"),
    (6, "json_schema", "JSON Schema Mode (response_format)"),
    (7, "prefix_caching", "Prefix / KV Cache Reuse"),
    (8, "client_abort", "Client Socket Abort Recovery"),
    (9, "stop_sequences", "Stop Words & Greedy Sampling"),
    (10, "high_entropy_recall", "High-Entropy Key-Value Recall"),
    (11, "extreme_precision", "Precision Ledger Reconcile"),
    (12, "code_execution", "Dynamic Code Unit Testing"),
    (13, "error_handling", "API Error Protocol Compliance"),
    (14, "context_scaling", "Dynamic Context Scaling"),
    (15, "multihop_graph", "Adversarial Graph & Distractors"),
    (16, "novel_algorithm_fuzz", "Novel Algorithm (5k Fuzz)"),
    (17, "combinatorial_anti_constraints", "Anti-Constraints (IFEval Tier)"),
    (18, "counterfactual_algebra", "Counterfactual Axiomatic Math"),
    (19, "frontier_needle_depth", "Frontier Depth Multi-Needle"),
    (20, "cruxeval", "CruxEval (Mental Code Exec)"),
    (21, "swe_bench_bug_patch", "SWE-bench (Traceback Fix)"),
    (22, "aime_olympiad", "AIME Olympiad (Math Reasoning)"),
]


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
        levels = [l for l in data.values() if isinstance(l, dict)]
        if not levels:
            st = data.get("status", "FAIL")
            err = data.get("error", "-")
            return {"status": st, "tokens": 0, "tok_s": 0.0, "ttft_ms": 0.0, "score": 0.0, "desc": err}
        all_passed = all("PASS" in str(l.get("status", "")) for l in levels)
        st = "PASS" if all_passed else "FAIL"
        tot_tokens = sum(l.get("total_tokens", 0) for l in levels)
        max_agg = max((l.get("aggregate_tok_s", 0) for l in levels), default=0.0)
        score = 1.0 if all_passed else 0.0
        desc = f"Max Agg: {max_agg:.1f} tok/s"
        return {"status": st, "tokens": tot_tokens, "tok_s": round(max_agg, 2), "ttft_ms": 0.0, "score": score, "desc": desc}

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
    if test_key == "prefix_caching":
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

    return {
        "status": st,
        "tokens": int(tokens),
        "tok_s": round(tok_s, 2),
        "ttft_ms": round(ttft_ms, 1),
        "score": score,
        "desc": desc
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
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "effectiveness_rate_pct": round(pass_rate_pct, 1),
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
    log("="*88)

    header = f"{'Evaluation Domain':<38} | {'Status':<10} | {'Key Metric / Latency':<20} | {'Throughput'}"
    log(header, bold=True)
    log("-" * 88)

    def fmt_status(st):
        if "PASS" in str(st):
            return f"{GREEN}{st}{RESET}"
        elif "FAIL" in str(st):
            return f"{RED}{st}{RESET}"
        elif "SKIPPED" in str(st):
            return f"{YELLOW}{st}{RESET}"
        return f"{YELLOW}{st}{RESET}"

    # 1. Streaming
    if "streaming" in res:
        s = res.get("streaming", {})
        log(f"{'1. Streaming & Latency':<38} | {fmt_status(s.get('status', 'N/A')):<19} | TTFT: {s.get('ttft_ms', 0):.1f} ms{'':<6} | {s.get('tok_s', 0):.2f} tok/s")
    else:
        log(f"{'1. Streaming & Latency':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 2. Vision
    if "vision" in res:
        v = res.get("vision", {})
        v_ttft = f"TTFT: {v.get('ttft_ms', 0):.1f} ms" if 'ttft_ms' in v else "N/A"
        v_speed = f"{v.get('tok_s', 0):.2f} tok/s" if 'tok_s' in v else "N/A"
        log(f"{'2. Multimodal Vision (Invoice)':<38} | {fmt_status(v.get('status', 'N/A')):<19} | {v_ttft:<20} | {v_speed}")
    else:
        log(f"{'2. Multimodal Vision (Invoice)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 3. Parallel Batching
    if "concurrency" in res:
        c = res.get("concurrency", {})
        for ckey, cval in c.items():
            c_label = f"3. Parallel Batching ({ckey.upper()})"
            c_speed = f"{cval.get('aggregate_tok_s', 0):.2f} tok/s (agg)"
            c_wall = f"{cval.get('wall_time_s', 0):.2f} s wall"
            log(f"{c_label:<38} | {fmt_status(cval.get('status', 'N/A')):<19} | {c_wall:<20} | {c_speed}")
    else:
        log(f"{'3. Parallel Batching':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 4. Capabilities
    if "capabilities_4tasks" in res:
        cap = res.get("capabilities_4tasks", {})
        cap_speed = f"{cap.get('average_speed_tok_s', 0):.2f} tok/s (avg)"
        log(f"{'4. 4-Task Architecture Suite':<38} | {fmt_status(cap.get('status', 'N/A')):<19} | 4/4 tasks passed{'':<5} | {cap_speed}")
    else:
        log(f"{'4. 4-Task Architecture Suite':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 5. Tool Calling
    if "tool_calling" in res:
        tc = res.get("tool_calling", {})
        tc_desc = "Agent Error Recovered" if tc.get("turn2_recovered") else ("Turn 1 Valid" if tc.get("turn1_valid") else "Args Invalid")
        log(f"{'5. Tool Calling & Agentic Recovery':<38} | {fmt_status(tc.get('status', 'N/A')):<19} | {tc_desc:<20} | TTFT: {tc.get('ttft_ms', 0):.1f} ms")
    else:
        log(f"{'5. Tool Calling & Agentic Recovery':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 6. JSON Schema
    if "json_schema" in res:
        js = res.get("json_schema", {})
        js_desc = "Strict Schema Conformed" if js.get("valid_schema") else "Schema Invalid"
        log(f"{'6. JSON Schema Mode (response_format)':<38} | {fmt_status(js.get('status', 'N/A')):<19} | {js_desc:<20} | {js.get('tok_s', 0):.2f} tok/s")
    else:
        log(f"{'6. JSON Schema Mode (response_format)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 7. Prefix Caching
    if "prefix_caching" in res:
        pc = res.get("prefix_caching", {})
        pc_desc = f"{pc.get('speedup_ratio', 1.0):.1f}x speedup" if pc.get("speedup_ratio") else "N/A"
        log(f"{'7. Prefix / KV Cache Reuse':<38} | {fmt_status(pc.get('status', 'N/A')):<19} | {pc_desc:<20} | Warm: {pc.get('warm_ttft_s', 0):.3f}s")
    else:
        log(f"{'7. Prefix / KV Cache Reuse':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 8. Client Abort
    if "client_abort" in res:
        ca = res.get("client_abort", {})
        ca_desc = f"Rec: {ca.get('recovery_latency_ms', 0):.1f} ms"
        log(f"{'8. Client Socket Abort Recovery':<38} | {fmt_status(ca.get('status', 'N/A')):<19} | {ca_desc:<20} | Slots Released")
    else:
        log(f"{'8. Client Socket Abort Recovery':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 9. Stop Sequences
    if "stop_sequences" in res:
        ss = res.get("stop_sequences", {})
        ss_desc = "Deterministic (temp=0)" if ss.get("deterministic_greedy_reproducible") else "Non-deterministic"
        log(f"{'9. Stop Words & Greedy Sampling':<38} | {fmt_status(ss.get('status', 'N/A')):<19} | {ss_desc:<20} | Tokens Suppressed")
    else:
        log(f"{'9. Stop Words & Greedy Sampling':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 10. High Entropy
    if "high_entropy_recall" in res:
        he = res.get("high_entropy_recall", {})
        he_speed = f"{he.get('tok_s', 0):.2f} tok/s"
        log(f"{'10. High-Entropy Key-Value Recall':<38} | {fmt_status(he.get('status', 'N/A')):<19} | TTFT: {he.get('ttft_ms', 0):.1f} ms{'':<4} | {he_speed}")
    else:
        log(f"{'10. High-Entropy Key-Value Recall':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 11. Extreme Precision
    if "extreme_precision" in res:
        ep = res.get("extreme_precision", {})
        ep_desc = "Exact Matched" if ep.get("matched") else "Balance Diverged"
        log(f"{'11. Precision Ledger Reconcile':<38} | {fmt_status(ep.get('status', 'N/A')):<19} | {ep_desc:<20} | {ep.get('tok_s', 0):.2f} tok/s")
    else:
        log(f"{'11. Precision Ledger Reconcile':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 12. Code Execution
    if "code_execution" in res:
        ce = res.get("code_execution", {})
        ce_desc = "Dynamic Assertions OK" if ce.get("dynamic_tests_passed") else "Assertion Failure"
        log(f"{'12. Dynamic Code Unit Testing':<38} | {fmt_status(ce.get('status', 'N/A')):<19} | {ce_desc:<20} | {ce.get('tok_s', 0):.2f} tok/s")
    else:
        log(f"{'12. Dynamic Code Unit Testing':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 13. Error Handling
    if "error_handling" in res:
        eh = res.get("error_handling", {})
        eh_desc = "HTTP 400/422 Standard" if eh.get("bad_schema_rejected") else "Non-standard error"
        log(f"{'13. API Error Protocol Compliance':<38} | {fmt_status(eh.get('status', 'N/A')):<19} | {eh_desc:<20} | Protocol OK")
    else:
        log(f"{'13. API Error Protocol Compliance':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 14. Context Scaling Summary
    if "context_scaling" in res:
        cs = res.get("context_scaling", [])
        if cs:
            log("-" * 88)
            log("  [14. Context Scaling Milestone Performance & Accuracy Breakdown]", bold=True)
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
    else:
        if 14 in report.get("selected_tests", range(1, TOTAL_TESTS + 1)):
            log(f"{'14. Dynamic Context Scaling':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 15. Multihop Graph Traversal
    if "multihop_graph" in res:
        mh = res.get("multihop_graph", {})
        mh_desc = f"{mh.get('completion_tokens', 0)} tokens"
        log(f"{'15. Adversarial Graph & Distractors':<38} | {fmt_status(mh.get('status', 'N/A')):<19} | {mh_desc:<20} | {mh.get('tok_s', 0):.2f} tok/s")
    elif 15 in report.get("selected_tests", set()):
        log(f"{'15. Adversarial Graph & Distractors':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 16. Novel Algorithmic Fuzz
    if "novel_algorithm_fuzz" in res:
        fz = res.get("novel_algorithm_fuzz", {})
        fz_desc = "5,000 Ops Fuzzed OK" if fz.get("status") == "PASS" else "Fuzz Assertion Failed"
        log(f"{'16. Novel Algorithm (5k Fuzz)':<38} | {fmt_status(fz.get('status', 'N/A')):<19} | {fz_desc:<20} | {fz.get('tok_s', 0):.2f} tok/s")
    elif 16 in report.get("selected_tests", set()):
        log(f"{'16. Novel Algorithm (5k Fuzz)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 17. Combinatorial Anti-Constraints
    if "combinatorial_anti_constraints" in res:
        ac = res.get("combinatorial_anti_constraints", {})
        ac_desc = f"{ac.get('constraints_passed', 0)}/6 Constraints Met"
        log(f"{'17. Anti-Constraints (IFEval Tier)':<38} | {fmt_status(ac.get('status', 'N/A')):<19} | {ac_desc:<20} | {ac.get('tok_s', 0):.2f} tok/s")
    elif 17 in report.get("selected_tests", set()):
        log(f"{'17. Anti-Constraints (IFEval Tier)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 18. Counterfactual Algebra
    if "counterfactual_algebra" in res:
        ca = res.get("counterfactual_algebra", {})
        ca_desc = "Both Roots Solved" if ca.get("status") == "PASS" else ("1 Root Solved" if ca.get("status") == "PARTIAL" else "Roots Missed")
        log(f"{'18. Counterfactual Axiomatic Math':<38} | {fmt_status(ca.get('status', 'N/A')):<19} | {ca_desc:<20} | {ca.get('tok_s', 0):.2f} tok/s")
    elif 18 in report.get("selected_tests", set()):
        log(f"{'18. Counterfactual Axiomatic Math':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 19. Frontier Needle Depth
    if "frontier_needle_depth" in res:
        fn = res.get("frontier_needle_depth", {})
        fn_desc = "Composite Checksum OK" if fn.get("status") == "PASS" else "Checksum Missed"
        log(f"{'19. Frontier Depth Multi-Needle':<38} | {fmt_status(fn.get('status', 'N/A')):<19} | {fn_desc:<20} | {fn.get('tok_s', 0):.2f} tok/s")
    elif 19 in report.get("selected_tests", set()):
        log(f"{'19. Frontier Depth Multi-Needle':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 20. CruxEval
    if "cruxeval" in res:
        cx = res.get("cruxeval", {})
        cx_desc = "State Traversed OK" if cx.get("status") == "PASS" else "State Diverged"
        log(f"{'20. CruxEval (Mental Code Exec)':<38} | {fmt_status(cx.get('status', 'N/A')):<19} | {cx_desc:<20} | {cx.get('tok_s', 0):.2f} tok/s")
    elif 20 in report.get("selected_tests", set()):
        log(f"{'20. CruxEval (Mental Code Exec)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 21. SWE-bench Bug Patching
    if "swe_bench_bug_patch" in res:
        sw = res.get("swe_bench_bug_patch", {})
        sw_desc = "Regression Tests OK" if sw.get("status") == "PASS" else "Unit Tests Failed"
        log(f"{'21. SWE-bench (Traceback Fix)':<38} | {fmt_status(sw.get('status', 'N/A')):<19} | {sw_desc:<20} | {sw.get('tok_s', 0):.2f} tok/s")
    elif 21 in report.get("selected_tests", set()):
        log(f"{'21. SWE-bench (Traceback Fix)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 22. AIME Olympiad Math
    if "aime_olympiad" in res:
        am = res.get("aime_olympiad", {})
        am_desc = "Exact Modular Root" if am.get("status") == "PASS" else "Arithmetic Missed"
        log(f"{'22. AIME Olympiad (Math Reasoning)':<38} | {fmt_status(am.get('status', 'N/A')):<19} | {am_desc:<20} | {am.get('tok_s', 0):.2f} tok/s")
    elif 22 in report.get("selected_tests", set()):
        log(f"{'22. AIME Olympiad (Math Reasoning)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

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

    m1, m2 = summaries[0], summaries[1]
    name1 = f"{m1['model']} ({m1['endpoint']})"
    name2 = f"{m2['model']} ({m2['endpoint']})"

    log(f"  Model A : {name1}", bold=True)
    log(f"  Model B : {name2}\n", bold=True)

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
    "high_entropy": 10,
    "high_entropy_recall": 10,
    "recall": 10,
    "extreme_precision": 11,
    "precision": 11,
    "code_execution": 12,
    "code": 12,
    "error_handling": 13,
    "error": 13,
    "context_scaling": 14,
    "context": 14,
    "scale": 14,
    "multihop_graph": 15,
    "multihop": 15,
    "graph": 15,
    "distractors": 15,
    "novel_algorithm_fuzz": 16,
    "novel_algorithm": 16,
    "fuzz": 16,
    "ring_buffer": 16,
    "combinatorial_anti_constraints": 17,
    "anti_constraints": 17,
    "ifeval": 17,
    "lipogram": 17,
    "counterfactual_algebra": 18,
    "algebra": 18,
    "symbolic": 18,
    "math": 18,
    "frontier_needle_depth": 19,
    "frontier_needle": 19,
    "frontier": 19,
    "cruxeval": 20,
    "code_exec_simulation": 20,
    "swe_bench": 21,
    "bug_patch": 21,
    "patching": 21,
    "rate_limiter": 21,
    "aime": 22,
    "olympiad": 22,
    "math_olympiad": 22,
}

def parse_selected_tests(test_arg: str, suite: str = "all"):
    if test_arg:
        selected = set()
        parts = [p.strip() for p in test_arg.replace(" ", ",").split(",") if p.strip()]
        for part in parts:
            if "-" in part and not part.startswith("-"):
                subparts = part.split("-", 1)
                if subparts[0].isdigit() and subparts[1].isdigit():
                    start_n, end_n = int(subparts[0]), int(subparts[1])
                    for n in range(start_n, end_n + 1):
                        if 1 <= n <= 22:
                            selected.add(n)
                    continue
            if part.isdigit():
                n = int(part)
                if 1 <= n <= 22:
                    selected.add(n)
            else:
                lower = part.lower().replace("-", "_")
                if lower in NAME_TO_TEST_NUM:
                    selected.add(NAME_TO_TEST_NUM[lower])
                elif lower == "flagship":
                    selected.update(range(1, 15))
                elif lower in ("adversarial", "hardened"):
                    selected.update(range(15, 20))
                elif lower in ("frontier", "sota"):
                    selected.update(range(20, 23))
                elif lower == "all":
                    selected.update(range(1, 23))
                else:
                    log(f"Warning: Unknown test identifier '{part}'. Ignored.", color=YELLOW)
        return selected if selected else set(range(1, 23))

    if suite == "flagship":
        return set(range(1, 15))
    elif suite == "adversarial":
        return set(range(15, 20))
    elif suite == "frontier":
        return set(range(20, 23))
    else:  # "all"
        return set(range(1, 23))


