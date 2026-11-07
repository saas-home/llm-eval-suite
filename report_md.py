#!/usr/bin/env python3
"""
report_md.py -- the decision report: one Markdown file for three readers.

  1. Summary: use cases ranked from best suited to not recommended, and
     whether the configured workload is workable on this server.
  2. Choosing a model: verdicts side by side with earlier runs, paired
     differences on the same questions, and vendor-published scores.
  3. Sizing hardware: the server, single-user and concurrent speeds, the
     workload test per user, goodput and servers needed.
  4. Approving for a team: per use case, the evidence, the reasons for the
     verdict and examples of wrong answers.

eval.py writes it next to the JSON report. To rebuild it, or to compare
runs after the fact:

    python3 report_md.py results/run.json [--compare-with results/old.json ...]

Written for GitHub's Markdown renderer. Zero external dependencies.
"""

import html
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import judge_validation
import suites
import usecases
import workload
from reporting import REPORTABLE_CATALOG, extract_test_metrics, format_duration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REFERENCE = os.path.join("data", "reference", "published_scores.json")
USE_CASE_NAMES = {uc["key"]: uc["name"] for uc in usecases.USE_CASES}
TEST_NAMES = {key: name for _num, key, name in REPORTABLE_CATALOG}
TEST_NUMBERS = {key: num for num, key, _name in REPORTABLE_CATALOG}
VERDICT_MARK = {usecases.RECOMMENDED: "✅", usecases.REVIEW: "⚠️", usecases.NOT_RECOMMENDED: "❌",
                usecases.NOT_MEASURED: "➖"}


# ----------------------------------------------------------------------------
# Small formatting helpers
# ----------------------------------------------------------------------------
def _md(text: Any) -> str:
    """Text safe inside a Markdown table cell or an HTML <summary>: model
    output must not open or close tags in the report."""
    return html.escape(str(text), quote=False).replace("|", "\\|").replace("\n", " ")


def _num(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}{unit}" if digits else f"{int(round(value)):,}{unit}"
    return str(value)


def _test(key: str) -> str:
    num = TEST_NUMBERS.get(key)
    return f"{TEST_NAMES.get(key, key)} (Test {num})" if num else key


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_md(c) for c in row) + " |" for row in rows]
    return out


def _point(p: Dict[str, Any]) -> str:
    return f"{TEST_NAMES.get(p['key'], p['key'])}: {p['accuracy_pct']:.0f}% (pass {p['pass_pct']:.0f}%)"


def _scored_cell(s: Dict[str, Any]) -> str:
    text = f"{s['accuracy_pct']:.0f}% [{s['ci95_low_pct']:.0f}-{s['ci95_high_pct']:.0f}]"
    if s.get("pass_all_repeats_pct") is not None and s.get("repeats", 1) > 1:
        text += f", {s['pass_all_repeats_pct']:.0f}% every try"
    return text


# ----------------------------------------------------------------------------
# Comparison helpers
# ----------------------------------------------------------------------------
def paired_difference(details_a: Optional[List[Dict[str, Any]]],
                      details_b: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Mean difference (a - b, percentage points) over the items both runs
    answered, with a 95% bootstrap CI over items. None without shared items."""
    a = {d["id"]: float(d["score"]) for d in details_a or [] if d.get("id") is not None and d.get("score") is not None}
    b = {d["id"]: float(d["score"]) for d in details_b or [] if d.get("id") is not None and d.get("score") is not None}
    shared = [k for k in a if k in b]
    if not shared:
        return None
    diffs = [a[k] - b[k] for k in shared]
    lo, hi = suites.bootstrap_ci(diffs)
    return {"diff_pts": round(100 * sum(diffs) / len(diffs), 1), "ci_low_pts": round(100 * lo, 1),
            "ci_high_pts": round(100 * hi, 1), "items": len(shared)}


def load_reference(path: Optional[str], report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The published-scores set that applies to the tested model (matched on
    the model name or the weights the server reports), or None."""
    path = path or os.path.join(BASE_DIR, DEFAULT_REFERENCE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    names = " ".join(str(x) for x in (report.get("model"), (report.get("server_info") or {}).get("weights")) if x)
    for ref in data.get("reference_sets", []):
        if any(a.lower() in names.lower() for a in ref.get("applies_to", [])):
            return ref
    return None


def _label(report: Dict[str, Any]) -> str:
    model = report.get("model") or "model"
    date = (report.get("timestamp") or "")[:10]
    return f"{model} ({date})" if date else model


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
def _header(report: Dict[str, Any]) -> List[str]:
    info = report.get("server_info") or {}
    judge = report.get("judge") or {}
    engine = " ".join(x for x in (info.get("engine"), info.get("version")) if x) or "not reported"
    lines = [f"# Model evaluation: {report.get('model', 'unknown model')}", ""]
    if report.get("incomplete"):
        lines += ["> [!WARNING]", f"> The run stopped early: {report['incomplete'].get('reason')}. "
                  f"{len(report['incomplete'].get('tests_not_run', []))} test(s) did not run, so verdicts below "
                  "may rest on less evidence than usual.", ""]
    judge_txt = _judge_status(judge)
    lines += _table(["Item", "Value"], [
        ["Date", report.get("timestamp", "-")],
        ["Endpoint", f"`{report.get('endpoint', '-')}`"],
        ["Inference server", engine],
        ["Model weights", info.get("weights") or report.get("model") or "-"],
        ["Hardware", report.get("hardware") or "not given (set `hardware` in config.json or `--hardware`)"],
        ["Judge for open-ended answers", judge_txt],
        ["Sampling", _sampling_text(report)],
        ["Output budget per answer", _budget_text(report)],
        ["Run time", format_duration(report.get("total_suite_wall_time_s", 0))],
    ])
    return lines + [""]


def _judge_configured(judge: Dict[str, Any]) -> Optional[bool]:
    """True/False from the report; None when it has no judge record."""
    if not judge:
        return None
    if "configured" in judge:
        return bool(judge["configured"])
    return bool(judge.get("model")) and not judge.get("self_judged")  # reports before 2026-11-07


def _judge_status(judge: Dict[str, Any]) -> str:
    if judge.get("self_judged"):  # reports before 2026-11-07
        return "the tested model (self-judged)"
    if _judge_configured(judge) is None:
        return "-"
    if not _judge_configured(judge):
        return "none: judged tests graded objectively (format and exact-match checks only)"
    text = judge.get("model") or "-"
    if judge.get("objective_reason"):
        text += " (went down during the run; later items graded objectively)"
    elif judge.get("failed_calls"):
        n = judge["failed_calls"]
        text += f" ({n} call{'s' if n != 1 else ''} failed; {'those items' if n != 1 else 'that item'} graded objectively)"
    return text


def _budget_text(report: Dict[str, Any]) -> str:
    n = report.get("max_output_tokens")
    if not n:
        return "8,192 tokens (reports before the setting existed)"
    return f"{n:,} tokens, capped per request to fit the context window (Test 46: its workload setting)"


def _sampling_text(report: Dict[str, Any]) -> str:
    s = report.get("sampling")
    if not s:
        return "suite default: temperature 0.6 for repeated questions (0.0 for one repeat)"
    return ", ".join(f"{k}={v}" for k, v in s.items() if k != "source") + (f" ({s['source']})" if s.get("source") else "")


def _workload_line(report: Dict[str, Any]) -> Optional[str]:
    wl = (report.get("results") or {}).get("workload")
    if not isinstance(wl, dict) or "users" not in wl:
        return None
    w = wl.get("workload", {})
    g = w.get("gates", {})
    n, good = len(wl["users"]), wl.get("goodput_users", 0)
    firsts = [u["first_token_s"] for u in wl["users"] if "first_token_s" in u]
    text = (f"**Workload ({n} users, new {w.get('prompt_tokens', 0):,}-token documents, ~{w.get('answer_tokens', 0):,}"
            f"-token answers): {good} of {n} users within the gates** (first token ≤ {g.get('first_token_s')} s, "
            f"complete answer ≤ {g.get('complete_s')} s, ≥ {g.get('per_user_tok_s')} tok/s per user).")
    if firsts:
        text += f" Users waited {min(firsts):.0f}-{max(firsts):.0f} s for the first token."
    sz = workload.sizing(wl)
    text += (f" {n} users need about {sz['servers_for_users']} server(s) like this one ({sz['basis']})."
             if sz["servers_for_users"] else
             f" Not even one user alone met the gates, so this server cannot carry this workload.")
    return text


def _summary(report: Dict[str, Any], ranked: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Summary", ""]
    by_tier: Dict[str, List[str]] = {}
    for r in ranked:
        by_tier.setdefault(r["verdict"], []).append(r["name"])
    for tier in (usecases.RECOMMENDED, usecases.REVIEW, usecases.NOT_RECOMMENDED, usecases.NOT_MEASURED):
        if by_tier.get(tier):
            lines.append(f"- {VERDICT_MARK[tier]} **{tier}:** {', '.join(by_tier[tier])}")
    wl = _workload_line(report)
    if wl:
        lines += ["", wl]
    lines += ["", "### Use cases, best suited first", ""]
    rows = []
    for r in ranked:
        sp = r.get("spread")
        own = [c for c in r["checks"] if not c["safety"]]
        if sp:
            low, avg, high = (_point(sp["low"]), f"{sp['average_pct']:.0f}% (pass {sp['average_pass_pct']:.0f}%)",
                              _point(sp["high"]))
            if sp["tests"] == 1:
                avg = high = "same"
        elif own and r["verdict"] != usecases.NOT_MEASURED:
            low, avg, high = f"{sum(c['status'] == 'PASS' for c in own)}/{len(own)} checks pass", "-", "-"
        else:
            low = avg = high = "-"
        evidence = (f"{r['evidence']} ({r['scored_items']} items)" if r["scored_items"]
                    else (f"{r['evidence']} (checks only)" if r["evidence"] == "limited" else r["evidence"]))
        verdict = f"{VERDICT_MARK[r['verdict']]} {r['verdict']}"
        if r.get("speed_gate_status") in ("failed", "partly met"):
            verdict += f" (speed gate {r['speed_gate_status']})"
        rows.append([r["rank"], r["name"], verdict,
                     evidence, low, avg, high, "; ".join(r["reasons"]) or "-"])
    lines += _table(["#", "Use case", "Verdict", "Evidence", "Low", "Average", "High", "Why"], rows)
    lines += ["", "Low and High are the tests furthest below and above their own pass marks; Average is the "
              "item-weighted mean accuracy. The verdict follows the weakest test. Details per use case, with 95% "
              "CIs and the share of questions right on every repeat, are under "
              "[Approving for a team](#approving-for-a-team).", ""]
    return lines


def _choosing(report: Dict[str, Any], ranked: List[Dict[str, Any]], others: List[Dict[str, Any]],
              reference: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["## Choosing a model", ""]
    if others:
        header = ["Use case", _label(report)] + [_label(o) for o in others]
        other_ranked = [{r["key"]: r for r in usecases.rank_use_cases(o.get("results") or {})} for o in others]
        rows = []
        for r in ranked:
            row = [r["name"], f"{VERDICT_MARK[r['verdict']]} {r['verdict']}"]
            for o in other_ranked:
                v = o[r["key"]]["verdict"]
                row.append(f"{VERDICT_MARK[v]} {v}")
            rows.append(row)
        lines += ["### Verdicts side by side", ""] + _table(header, rows) + [""]
        lines += ["### Same questions, paired", "",
                  "Difference = this run minus the other run, in percentage points, over the questions both "
                  "answered. A CI that excludes 0 is a real difference.", ""]
        rows = []
        results = report.get("results") or {}
        for o in others:
            for key, data in results.items():
                if not isinstance(data, dict) or not data.get("details"):
                    continue
                theirs = (o.get("results") or {}).get(key)
                if not isinstance(theirs, dict):
                    continue
                pd = paired_difference(data["details"], theirs.get("details"))
                if pd:
                    rows.append([_test(key), _label(o), f"{pd['diff_pts']:+.1f}",
                                 f"[{pd['ci_low_pts']:+.1f}, {pd['ci_high_pts']:+.1f}]", pd["items"]])
        lines += (_table(["Test", "Compared with", "Difference (pts)", "95% CI", "Shared questions"], rows)
                  if rows else ["No test has questions in common with the compared runs."]) + [""]
    else:
        lines += ["No other run was given, so this section ranks this model alone (see the summary). To compare, "
                  "pass earlier reports with `--compare-with results/<run>.json`, or list them in config.json "
                  "under `compare_reports`.", ""]
    if reference:
        lines += _published(reference)
    return lines


def _published(ref: Dict[str, Any]) -> List[str]:
    models = ref.get("models", [])
    lines = [f"### Published by the vendor ([{ref.get('title')}]({ref.get('source')}), retrieved {ref.get('retrieved')})",
             "", f"> [!NOTE]\n> {ref.get('note')}", ""]
    params = ref.get("parameters") or {}
    rows = [[name] + [vals.get(m, "-") for m in models] for name, vals in params.items()]
    rows += [[f"{b['name']} ({USE_CASE_NAMES.get(b['use_case'], b['use_case'])})"]
             + [_num(b["scores"].get(m)) for m in models] for b in ref.get("benchmarks", [])]
    lines += _table(["Benchmark (use case)"] + models, rows) + [""]
    for q in ref.get("quantisations", []):
        lines.append(f"- [{q['model']}]({q['source']}): {q['note']}")
    return lines + [""]


def _sizing(report: Dict[str, Any]) -> List[str]:
    results = report.get("results") or {}
    info = report.get("server_info") or {}
    t3, t12, wl = _result(results, "concurrency"), _result(results, "context_scaling"), _result(results, "workload")
    lines = ["## Sizing hardware", "", "### Server", ""]
    kv_mem = info.get("kv_cache_memory_bytes")
    lines += _table(["Item", "Value"], [
        ["Inference server", " ".join(x for x in (info.get("engine"), info.get("version")) if x) or "not reported"],
        ["Weights", info.get("weights") or "-"],
        ["Hardware", report.get("hardware") or "not given"],
        ["Configured context", _num(report.get("max_context_tokens"), 0, " tokens")],
        ["KV cache", (f"{info['kv_cache_tokens']:,} tokens" if info.get("kv_cache_tokens") else "not reported")
         + (f", {info.get('kv_cache_dtype')}" if info.get("kv_cache_dtype") else "")
         + (f", {kv_mem / 1e9:.0f} GB" if kv_mem else "")],
        ["Prefix caching", "not reported" if info.get("prefix_caching") is None
         else ("on" if info["prefix_caching"] else "off")],
        ["Full context verified", f"{t12.get('final_step_tokens', 0):,} tokens" if t12.get("max_context_verified")
         else (t12.get("note") or "not run")],
    ]) + [""]

    solo = wl.get("solo") or {}
    users = [u for u in wl.get("users", []) if "first_token_s" in u]
    n = len(wl.get("users", []))
    per_user_decode = [u["decode_tok_s"] for u in users]
    lines += ["### Speed", ""]
    lines += _table(["Measure", "1 user", f"{n} users (workload)" if n else "Workload",
                     f"Most parallel requests verified ({t3.get('verified_max_parallel', '-')})"], [
        ["Prefill, short prompts (tok/s)", _num(t3.get("peak_prefill_tok_s"), 0), "-", "-"],
        ["Prefill at the workload prompt (tok/s)", _num(solo.get("prefill_tok_s"), 0),
         _num(wl.get("aggregate_prefill_tok_s"), 0) + (" combined" if wl else ""), "-"],
        ["Decode per user (tok/s)", _num(t3.get("single_stream_decode_tok_s")),
         _range(per_user_decode, 1),
         _num(_level_at(t3, t3.get("verified_max_parallel"), "per_stream_decode_tok_s"))],
        ["Decode, all users combined (tok/s)", _num(t3.get("single_stream_decode_tok_s")),
         _num(wl.get("aggregate_decode_tok_s")), _num(_level_at(t3, t3.get("verified_max_parallel"), "aggregate_decode_tok_s"))],
        ["First token at the workload prompt (s)", _num(solo.get("first_token_s")),
         _range([u["first_token_s"] for u in users], 0), "-"],
    ]) + ["", "Short-prompt numbers come from Test 3 (one request at a time, then a concurrency sweep); "
          "workload numbers from Test 46.", ""]

    if wl and "users" in wl:
        w = wl.get("workload", {})
        lines += ["### Workload test, per user", "",
                  f"Each user sends a new document set of about {w.get('prompt_tokens', 0):,} tokens (nothing cached), "
                  f"asks for one clause and a ~{w.get('answer_tokens', 0):,}-token review, with thinking on. Users "
                  f"arrive {w.get('stagger_s', ['?', '?'])[0]:.0f}-{w.get('stagger_s', ['?', '?'])[1]:.0f} s apart.", ""]
        rows = []
        for i, u in enumerate([solo] + wl["users"]):
            who = "alone" if i == 0 else str(i)
            if u.get("error"):
                rows.append([who, _num(u.get("start_offset_s")), "error", "", "", "", "", "", u["error"][:80]])
                continue
            rows.append([who, _num(u.get("start_offset_s")), _num(u.get("prompt_tokens"), 0),
                         _num(u.get("first_token_s")), _num(u.get("answer_start_s")), _num(u.get("complete_s")),
                         _num(u.get("decode_tok_s")), "right" if u.get("score") else "wrong",
                         "met" if u.get("meets_gates") else "missed: " + ", ".join(
                             k for k, ok in (u.get("gate_checks") or {}).items() if not ok)])
        lines += _table(["User", "Arrives (s)", "Prompt tokens", "First token (s)", "Answer starts (s)",
                         "Complete (s)", "Decode (tok/s)", "Clause", "Gates"], rows) + [""]
        good = wl.get("goodput_users", 0)
        sz = workload.sizing(wl)
        kv_sessions = wl.get("kv_sessions_at_workload")
        lines.append(f"- **Goodput:** {good} of {n} users at once met every gate"
                     + (" (the user alone did)" if (solo.get("meets_gates") and not good) else "") + ".")
        if kv_sessions is not None:
            lines.append(f"- **Memory:** the KV cache holds {kv_sessions} sessions of this size at once, so memory "
                         "is not the limit when fewer users than that wait longer: prefill compute is.")
        if sz["servers_for_users"]:
            lines.append(f"- **Servers needed:** about {sz['servers_for_users']} like this one for {n} users at "
                         f"this workload ({sz['users_per_server']} user(s) per server: {sz['basis']}). This assumes "
                         "identical servers behind a load balancer.")
        else:
            lines.append("- **Servers needed:** not estimated: not even one user alone met the gates. Faster prefill "
                         "(more or faster GPUs), shorter prompts, or reusing documents across questions "
                         "(prefix caching) would be needed.")
        lines.append("")
    return lines


def _result(results: Dict[str, Any], key: str) -> Dict[str, Any]:
    data = results.get(key)
    return data if isinstance(data, dict) else {}


def _range(values: List[float], digits: int) -> str:
    """'20.0' when all values round the same, '85-325' otherwise, '-' when empty."""
    if not values:
        return "-"
    lo, hi = f"{min(values):,.{digits}f}", f"{max(values):,.{digits}f}"
    return lo if lo == hi else f"{lo}-{hi}"


def _level_at(t3: Dict[str, Any], clients: Any, field: str) -> Optional[float]:
    """`field` of Test 3's concurrency level with `clients` streams."""
    for lv in t3.get("levels") or []:
        if lv.get("clients") == clients:
            return lv.get(field)
    return None


def _approving(report: Dict[str, Any], ranked: List[Dict[str, Any]], judge_agreement: Optional[str]) -> List[str]:
    judge = report.get("judge") or {}
    lines = ["## Approving for a team", ""]
    for r in ranked:
        lines += [f"### {r['rank']}. {r['name']}: {VERDICT_MARK[r['verdict']]} {r['verdict']}", ""]
        if r["verdict"] == usecases.NOT_MEASURED:
            lines += ["None of this use case's tests ran.", ""]
            continue
        notes = [f"Evidence: **{r['evidence']}** ({r['scored_items']} scored items"
                 + (f", {len(r['checks'])} checks)" if r["checks"] else ")")]
        if r["reasons"]:
            notes.append("Why: " + "; ".join(r["reasons"]))
        if r["public_only"]:
            notes.append("All scored items come from public benchmarks, which may be in the model's training "
                         "data; real-world accuracy can be lower.")
        if r["no_judge"]:
            notes.append("Graded without a judge: only checks a program can make (format rules, exact answers, "
                         "word overlap with reference points). Tone, completeness and invented statements were "
                         "not checked.")
        elif r["judged"]:
            who = "the tested model itself" if judge.get("self_judged") else (judge.get("model") or "an LLM judge")
            notes.append(f"Open-ended answers were judged by {who}. "
                         + (judge_validation.validation_text(judge.get("validation")) or judge_agreement
                            or "The judge was not checked against expert ratings."))
        if r["speed_gate"]:
            wl = _workload_line(report)
            if wl:
                notes.append(wl)
        lines += [f"- {n}" for n in notes] + [""]
        if r["scored"]:
            rows = [[_test(s["key"]), f"{s['accuracy_pct']:.1f}% [{s['ci95_low_pct']:.0f}-{s['ci95_high_pct']:.0f}]",
                     f"{s['pass_pct']:.0f}% / {s['partial_pct']:.0f}%",
                     _num(s["pass_all_repeats_pct"], 0, "%") if s["repeats"] > 1 else "-",
                     f"{s['items']} × {s['repeats']}", "public" if s["public"] else "original"]
                    for s in r["scored"]]
            lines += _table(["Scored test", "Accuracy [95% CI]", "Pass / floor", "Right every try",
                             "Items × repeats", "Data"], rows) + [""]
        if r["checks"]:
            lines.append("Checks: " + ", ".join(
                f"{_test(c['key'])} {c['status']}" + (f" ({c['note']})" if c.get("note") else "")
                + (" (safety)" if c["safety"] else "") for c in r["checks"]))
            lines.append("")
        for ex in r["failure_examples"]:
            detail = ", ".join(f"{k}: {v}" for k, v in ex["detail"].items())
            lines += [f"<details><summary>Wrong answer: {_md(_test(ex['test']))}, item {_md(ex['item'])}"
                      f"{' (' + _md(detail) + ')' if detail else ''}</summary>", "", "```text",
                      ex["answer_tail"].replace("```", "'''"), "```", "", "</details>", ""]
    return lines


def _method(report: Dict[str, Any]) -> List[str]:
    q = report.get("qualification") or {}
    judge = report.get("judge") or {}
    lines = ["## Method and caveats", ""]
    lines += [
        f"- **Questions:** {q.get('samples', '-')} per benchmark, each asked {q.get('repeats', '-')} times. "
        "Accuracy is the mean item score; the 95% CI comes from a bootstrap over items.",
        f"- **Sampling:** {_sampling_text(report)}.",
        f"- **Output budget per answer (thinking included):** {_budget_text(report)}.",
        f"- **Verdict rule:** Recommended needs every scored test's CI lower bound at or above its pass mark, "
        f"at least {usecases.RELIABILITY_MIN_PCT:.0f}% of items right on every repeat, no failed check and at least "
        f"{usecases.STRONG_EVIDENCE_ITEMS} scored items. Usable with human review needs every mean at or above its "
        "floor. A use case is as strong as its weakest scored test.",
        "- **Pass marks** come from published scores of comparable models where they exist; each result records "
        "its basis (`pass_marks.basis` in the JSON).",
        "- **Public benchmarks** may be in a model's training data. Within a verdict, original tasks count twice "
        "as much as public ones when use cases are ranked.",
    ]
    if _judge_configured(judge) is False and not judge.get("self_judged"):
        lines.append("- **No judge:** judged tests (41-44, 48) were graded objectively: format rules, exact answers, "
                     "refusal phrases and word overlap with reference points. Items with nothing a program can "
                     "check are left out and counted as unscored.")
    elif judge.get("objective_reason"):
        lines.append(f"- **Judge went down:** {judge['objective_reason'][:160]}. Later judged items were graded "
                     "objectively, and those results are labelled.")
    vt = judge_validation.validation_text(judge.get("validation"))
    if vt:
        lines.append(f"- **Judge check:** {vt}")
    return lines + [""]


def _appendix(report: Dict[str, Any]) -> List[str]:
    results = report.get("results") or {}
    timings = {t["key"]: t for t in report.get("test_timings") or []}
    rows = []
    for num, key, name in REPORTABLE_CATALOG:
        if key not in results:
            continue
        m = extract_test_metrics(key, results[key])
        dur = timings.get(key, {}).get("duration_s")
        rows.append([num, name, m["status"], m["desc"], format_duration(dur) if dur is not None else "-"])
    return ["## Appendix: every test", ""] + _table(["#", "Test", "Status", "Result", "Duration"], rows) + [""]


def judge_agreement_text(report: Dict[str, Any], labels_path: Optional[str]) -> Optional[str]:
    """'The judge agreed with human labels on X of Y criteria (Z%).' from a
    labels file [{"test", "item", "criterion", "human": true/false}], matched
    to the first attempt's judge verdicts. None without a usable file."""
    if not labels_path or not os.path.exists(labels_path):
        return None
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    results = report.get("results") or {}
    agree = total = 0
    for lab in labels:
        data = results.get(lab.get("test"))
        if not isinstance(data, dict) or lab.get("human") is None:
            continue
        for d in data.get("details") or []:
            if d.get("id") == lab.get("item") and d.get("attempts"):
                verdict = (d["attempts"][0].get("judge") or {}).get(lab.get("criterion"))
                if verdict is not None:
                    total += 1
                    agree += verdict == lab["human"]
    if not total:
        return None
    return f"The judge agreed with human labels on {agree} of {total} criteria ({100 * agree / total:.0f}%)."


def build_report(report: Dict[str, Any], others: Optional[List[Dict[str, Any]]] = None,
                 reference: Optional[Dict[str, Any]] = None, judge_agreement: Optional[str] = None) -> str:
    ranked = usecases.rank_use_cases(report.get("results") or {})
    lines: List[str] = []
    lines += _header(report)
    lines += _summary(report, ranked)
    lines += _choosing(report, ranked, others or [], reference)
    lines += _sizing(report)
    lines += _approving(report, ranked, judge_agreement)
    lines += _method(report)
    lines += _appendix(report)
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: Dict[str, Any], json_path: str, compare_paths: Optional[List[str]] = None,
                 reference_path: Optional[str] = None, judge_labels: Optional[str] = None) -> str:
    """Writes <json_path minus .json>.md and returns its path."""
    others = []
    for p in compare_paths or []:
        with open(p, encoding="utf-8") as f:
            others.append(json.load(f))
    md = build_report(report, others, load_reference(reference_path, report),
                      judge_agreement_text(report, judge_labels))
    out = (json_path[:-5] if json_path.endswith(".json") else json_path) + ".md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Build the Markdown decision report from a JSON report.")
    ap.add_argument("report", help="JSON report written by eval.py")
    ap.add_argument("--compare-with", nargs="+", default=[], metavar="REPORT_JSON", help="Earlier JSON reports")
    ap.add_argument("--reference-scores", default=None, help=f"Published scores file (default: {DEFAULT_REFERENCE})")
    ap.add_argument("--judge-labels", default=None, help="Human labels for judged criteria (JSON list)")
    ap.add_argument("--hardware", default=None,
                    help="GPU type and count, when the run did not record it. Also saved in the JSON report.")
    args = ap.parse_args(argv)
    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)
    if args.hardware:
        report["hardware"] = args.hardware
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(write_report(report, args.report, args.compare_with, args.reference_scores, args.judge_labels))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
