#!/usr/bin/env python3
"""
usecases.py -- turns test results into a verdict per use case.

Each use case lists its scored tests (many items, repeats, confidence
intervals) and its checks (single-run core tests). The verdict is one of:

  Recommended               every scored test's 95% CI lower bound is at or
                            above its pass mark, every repeated test answers
                            at least 85% of items right on every attempt
                            (pass^k), no check failed, and the evidence is
                            strong.
  Usable with human review  every scored test's mean is at or above its
                            partial mark (or, with checks only, none failed).
  Not recommended           below that.
  Not measured              none of its tests ran.

A use case is only as strong as its weakest scored test. Evidence is strong
with at least 40 scored items, limited with fewer or with checks only; limited
evidence is never Recommended. Within a verdict, use cases rank by the margin
of their CI lower bounds over the pass marks, with original tasks weighted
twice as much as public benchmarks (public items may be in training data).

Zero external dependencies.
"""

import statistics
from typing import Any, Dict, List, Optional

import suites

RECOMMENDED, REVIEW, NOT_RECOMMENDED, NOT_MEASURED = (
    "Recommended", "Usable with human review", "Not recommended", "Not measured")
TIER_ORDER = {RECOMMENDED: 0, REVIEW: 1, NOT_RECOMMENDED: 2, NOT_MEASURED: 3}
RELIABILITY_MIN_PCT = 85.0
STRONG_EVIDENCE_ITEMS = 40

USE_CASES: List[Dict[str, Any]] = [
    {"key": "long_document", "name": "Long-document Q&A and review",
     "scored": ["contract_review", "workload"],
     "checks": ["context_scaling", "multihop_graph"], "speed_gate": True},
    {"key": "legal", "name": "Legal reasoning", "scored": ["legalbench", "contract_review"], "checks": []},
    {"key": "drafting", "name": "Drafting and writing", "scored": ["ifeval_full", "writing_tasks"], "checks": []},
    {"key": "summarisation", "name": "Summarisation", "scored": ["summarisation"], "checks": []},
    {"key": "extraction", "name": "Structured extraction (JSON)", "scored": [],
     "checks": ["json_schema", "vision", "extreme_precision"]},
    {"key": "coding", "name": "Coding", "scored": ["multipl_e"],
     "checks": ["novel_algorithm_fuzz", "swe_bench_bug_patch", "surgical_git_diff", "multifile_context",
                "code_review_security", "unit_test_synthesis", "strict_typing", "refactor_invariants"]},
    {"key": "design", "name": "Software design and architecture", "scored": [], "checks": ["capabilities_4tasks"]},
    {"key": "agent", "name": "Agent and tool use", "scored": [], "checks": ["tool_calling"]},
    {"key": "medical", "name": "Medical and scientific Q&A",
     "scored": ["medqa_usmle", "medmcqa", "pubmedqa_full", "mmlu_medical"], "checks": []},
    {"key": "reasoning", "name": "Maths and reasoning", "scored": ["mmlu_pro", "math500", "bbh"], "checks": []},
    {"key": "assistant", "name": "Personal assistant", "scored": ["personal_tasks", "simpleqa", "xstest"],
     "checks": []},
]
# Safety checks that apply to every use case.
SAFETY_CHECKS = ["prompt_injection_defense"]

# Public benchmarks: their items may be in a model's training data, so a high
# score can overstate real skill. Everything else is original to this suite.
PUBLIC_BENCHMARKS = frozenset({
    "multipl_e", "medqa_usmle", "medmcqa", "pubmedqa_full", "mmlu_medical", "mmlu_pro", "math500", "bbh",
    "ifeval_full", "simpleqa", "xstest", "legalbench", "contract_review", "summarisation", "workload",
})
# Tests scored by an LLM judge.
JUDGED_TESTS = frozenset({"writing_tasks", "personal_tasks", "summarisation", "simpleqa", "xstest"})


# A result with one of these failure kinds is not a measurement of the model.
NOT_A_SCORE = frozenset({"infra", "harness"})


def scored_summary(key: str, data: Any) -> Optional[Dict[str, Any]]:
    """Accuracy, CI, marks and reliability of one scored test, or None when
    it did not produce a model score (skipped, infra failure, missing)."""
    if not isinstance(data, dict) or data.get("status") == "SKIPPED" or data.get("failure_kind") in NOT_A_SCORE:
        return None
    details = data.get("details") or []
    if "accuracy_pct" in data:
        acc, lo, hi = data["accuracy_pct"], data.get("ci95_low_pct", 0.0), data.get("ci95_high_pct", 100.0)
        items = data.get("items") or len(details)
    elif details:  # e.g. the workload test: one graded answer per user
        scores = [d.get("score", 0.0) for d in details]
        low, high = suites.bootstrap_ci(scores)
        acc, lo, hi, items = 100.0 * statistics.mean(scores), 100.0 * low, 100.0 * high, len(scores)
    else:
        return None
    # Older or partial records may lack a mark; the benchmark's own marks fill the gap.
    marks = dict(suites._marks_record(key), **{k: v for k, v in (data.get("pass_marks") or {}).items()
                                              if v is not None})
    rel = data.get("reliability") or {}
    return {"key": key, "accuracy_pct": round(acc, 1), "ci95_low_pct": round(lo, 1), "ci95_high_pct": round(hi, 1),
            "items": items, "repeats": data.get("repeats", 1), "pass_pct": marks["pass_pct"],
            "partial_pct": marks["partial_pct"], "pass_all_repeats_pct": rel.get("pass_all_repeats_pct"),
            "inconsistent_items": rel.get("inconsistent_items"), "public": key in PUBLIC_BENCHMARKS,
            "judged": key in JUDGED_TESTS,
            # Graded without a judge (none configured, or it went down): only
            # format and exact-match checks were made.
            "objective": bool(data.get("objective_only") or data.get("partly_objective")),
            "unscored_items": data.get("unscored_items", 0)}


def scored_tier(s: Dict[str, Any]) -> str:
    reliable = s["repeats"] < 2 or s["pass_all_repeats_pct"] is None or s["pass_all_repeats_pct"] >= RELIABILITY_MIN_PCT
    if s["ci95_low_pct"] >= s["pass_pct"] and reliable:
        return RECOMMENDED
    if s["accuracy_pct"] >= s["partial_pct"]:
        return REVIEW
    return NOT_RECOMMENDED


def failure_examples(results: Dict[str, Any], keys: List[str], limit: int = 2) -> List[Dict[str, Any]]:
    """Up to `limit` failed attempts (with the end of the answer) across the
    given tests, one per item, for the reader to see what going wrong looks like."""
    out = []
    for key in keys:
        data = results.get(key)
        if not isinstance(data, dict):
            continue
        for d in data.get("details") or []:
            for attempt, score in zip(d.get("attempts") or [], d.get("attempt_scores") or []):
                if score is not None and score < 1.0 and isinstance(attempt, dict) and attempt.get("response_tail"):
                    shown = {k: v for k, v in attempt.items()
                             if k in ("expected", "extracted", "outcome", "coverage", "faithful") and v is not None}
                    out.append({"test": key, "item": d.get("id"), "detail": shown,
                                "answer_tail": attempt["response_tail"][-400:]})
                    break
            if len(out) >= limit:
                return out
    return out


def check_note(data: Dict[str, Any]) -> str:
    """How many retries a check with self-correction needed (Test 20): '' when
    none did, else e.g. "retried: go, rust; score 0.93"."""
    per_lang = data.get("language_retries")
    if isinstance(per_lang, dict):
        retried = [lang for lang, n in per_lang.items() if n]
        if not retried:
            return ""
        return f"retried: {', '.join(retried)}; score {data.get('score', 0):.2f}"
    retries = data.get("retries")
    if retries and data.get("status") == "PASS":
        return f"fixed after {retries} retr{'y' if retries == 1 else 'ies'}; score {data.get('score', 0):.2f}"
    return ""


def evaluate_use_case(uc: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    scored = [s for s in (scored_summary(k, results.get(k)) for k in uc["scored"]) if s]
    # A scored test that ran (or was due to run) but produced no score leaves
    # the weakest link unknown: the verdict cannot be Recommended.
    unmeasured = [k for k in uc["scored"] if k in results and scored_summary(k, results.get(k)) is None]
    checks = []
    # Under --language polyglot the coding checks run per language; the
    # combined result is under the plain key.
    for key in uc["checks"] + SAFETY_CHECKS:
        data = results.get(key)
        if not isinstance(data, dict):
            continue
        status = data.get("status")
        if status and status != "SKIPPED" and data.get("failure_kind") not in NOT_A_SCORE:
            checks.append({"key": key, "status": status,
                           "safety": key in SAFETY_CHECKS, "note": check_note(data)})
    own_checks = [c for c in checks if not c["safety"]]
    items = sum(s["items"] for s in scored)
    evidence = "strong" if items >= STRONG_EVIDENCE_ITEMS else ("limited" if (scored or own_checks) else "none")
    reasons: List[str] = []
    if not scored and not own_checks:
        tier = NOT_MEASURED
    elif scored:
        tiers = [scored_tier(s) for s in scored]
        tier = max(tiers, key=lambda t: TIER_ORDER[t])
        weakest = scored[tiers.index(tier)]
        if tier != RECOMMENDED:
            reasons.append(_tier_reason(weakest))
    else:
        tier = REVIEW  # checks only: limited evidence
    failed = [c["key"] for c in checks if c["status"] == "FAIL"]
    if failed and not scored and tier == REVIEW:
        tier = NOT_RECOMMENDED
    if failed and tier == RECOMMENDED:
        tier = REVIEW
    if failed:
        reasons.append("failed check(s): " + ", ".join(sorted(set(failed))))
    if unmeasured:
        if tier == RECOMMENDED:
            tier = REVIEW
        reasons.append("no score from: " + ", ".join(unmeasured) + " (skipped, infra or harness failure)")
    if tier == RECOMMENDED and evidence != "strong":
        tier = REVIEW
        reasons.append(f"limited evidence ({items} scored items)")
    no_judge = [s["key"] for s in scored if s["objective"]]
    if no_judge:
        if tier == RECOMMENDED:
            tier = REVIEW
        reasons.append("graded without a judge (format and exact-match checks only): " + ", ".join(no_judge))
    if uc.get("speed_gate") and speed_gate_status(results) in ("failed", "partly met"):
        wl = results.get("workload") or {}
        reasons.append(f"speed gate {speed_gate_status(results)}: {wl.get('goodput_users', 0)} of "
                       f"{len(wl.get('users') or [])} users within the workload's speed limits")
    if evidence == "limited" and not scored:
        reasons.append("checks only: no scored benchmark")
    margins = [(s["ci95_low_pct"] - s["pass_pct"], 1 if s["public"] else 2) for s in scored]
    margin = (sum(m * w for m, w in margins) / sum(w for _, w in margins)) if margins else None
    pass_k = [(s["pass_all_repeats_pct"], s["items"]) for s in scored
              if s["pass_all_repeats_pct"] is not None and s["repeats"] > 1]
    return {
        "key": uc["key"], "name": uc["name"], "verdict": tier, "evidence": evidence, "scored_items": items,
        "scored": scored, "checks": checks, "reasons": reasons, "unmeasured": unmeasured,
        "weighted_margin": round(margin, 1) if margin is not None else None,
        "pass_all_repeats_pct": round(sum(p * n for p, n in pass_k) / sum(n for _, n in pass_k), 1) if pass_k else None,
        "public_only": bool(scored) and all(s["public"] for s in scored),
        "judged": any(s["judged"] for s in scored),
        "no_judge": no_judge,
        "speed_gate": bool(uc.get("speed_gate")),
        "speed_gate_status": speed_gate_status(results) if uc.get("speed_gate") else None,
        "spread": spread(scored),
        "failure_examples": failure_examples(results, uc["scored"]),
    }


def speed_gate_status(results: Dict[str, Any]) -> str:
    """'met', 'partly met', 'failed' or 'not measured' for Test 46. Speed is
    a gate shown next to the accuracy verdict, not part of it."""
    wl = results.get("workload")
    if not isinstance(wl, dict) or wl.get("status") == "SKIPPED" or wl.get("failure_kind") in NOT_A_SCORE:
        return "not measured"
    return {"PASS": "met", "PARTIAL": "partly met"}.get(str(wl.get("status")), "failed")


def _tier_reason(s: Dict[str, Any]) -> str:
    if s["accuracy_pct"] < s["partial_pct"]:
        return f"{s['key']}: {s['accuracy_pct']:.0f}% is below the {s['partial_pct']:.0f}% floor"
    if s["ci95_low_pct"] < s["pass_pct"]:
        return (f"{s['key']}: CI lower bound {s['ci95_low_pct']:.0f}% is below the {s['pass_pct']:.0f}% pass mark "
                f"({s['accuracy_pct']:.0f}% mean, {s['items']} items)")
    return (f"{s['key']}: only {s['pass_all_repeats_pct']:.0f}% of items right on every attempt "
            f"(needs {RELIABILITY_MIN_PCT:.0f}%)")


def spread(scored: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Low, average and high of a use case's scored tests. Low and high are
    the tests furthest below and above their own pass marks (tests use
    different scales: 10% is par on SimpleQA, not on MedQA). Average is the
    item-weighted mean accuracy, next to the item-weighted mean pass mark."""
    if not scored:
        return None
    by_margin = sorted(scored, key=lambda s: s["accuracy_pct"] - s["pass_pct"])
    items = sum(s["items"] for s in scored) or 1

    def point(s):
        return {"key": s["key"], "accuracy_pct": s["accuracy_pct"], "pass_pct": s["pass_pct"]}
    return {"low": point(by_margin[0]), "high": point(by_margin[-1]), "tests": len(scored), "items": items,
            "average_pct": round(sum(s["accuracy_pct"] * s["items"] for s in scored) / items, 1),
            "average_pass_pct": round(sum(s["pass_pct"] * s["items"] for s in scored) / items, 1)}


def rank_use_cases(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every use case with its verdict, best suited first."""
    rows = [evaluate_use_case(uc, results) for uc in USE_CASES]
    evidence_order = {"strong": 0, "limited": 1, "none": 2}
    rows.sort(key=lambda r: (TIER_ORDER[r["verdict"]], evidence_order[r["evidence"]],
                             -(r["weighted_margin"] if r["weighted_margin"] is not None else -1e9)))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows
