#!/usr/bin/env python3
"""
judge_validation.py -- how often the LLM judge agrees with human experts.

Runs only when a judge model is configured. It asks the judge the same kind
of yes/no questions the judged tests ask, on public data that experts have
already rated:

  SummEval (MIT)  40 news summaries: 20 that every expert rated fully
                  consistent with the article, 20 rated inconsistent. The
                  judge gets the article and Test 48's faithfulness question.
                  Agreement = share of summaries where the judge's yes/no
                  matches the experts.
  LLMBar (MIT)    40 instruction-following answer pairs (20 natural, 20
                  adversarial), where experts marked the better answer. The
                  judge rates each answer on its own, as in Tests 41 and 44.
                  A pair is separated correctly when the judge accepts the
                  better answer and rejects the worse one.

The report states both numbers next to every judged result. Zero external
dependencies.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import suites
from client import CYAN, GREEN, YELLOW, log

LLMBAR_CRITERION = ("The response follows the user's instruction precisely and completely, and what it says is "
                    "accurate and helpful.")


def _summeval(judge, item: Dict[str, Any]) -> Optional[bool]:
    """True when the judge agrees with the experts, None without a verdict."""
    out = judge.checklist("Summarise the article.", item["summary"], [suites.FAITHFUL_CRITERION],
                          context=item["source"], count=False)
    verdict = out["verdicts"][0] if out.get("judged_by") == "judge" else None
    return None if verdict is None else verdict == item["consistent"]


def _llmbar(judge, pair: Dict[str, Any]) -> str:
    """'separated', 'tie' (same verdict for both), 'reversed', or 'no verdict'."""
    verdicts = []
    for key in ("output_1", "output_2"):
        out = judge.checklist(pair["instruction"], pair[key], [LLMBAR_CRITERION], count=False)
        verdicts.append(out["verdicts"][0] if out.get("judged_by") == "judge" else None)
    if None in verdicts:
        return "no verdict"
    better, worse = (verdicts[0], verdicts[1]) if pair["better"] == 1 else (verdicts[1], verdicts[0])
    if better and not worse:
        return "separated"
    if worse and not better:
        return "reversed"
    return "tie"


def summarise(summeval: List[Optional[bool]], llmbar: List[str]) -> Dict[str, Any]:
    answered = [x for x in summeval if x is not None]
    pairs = [x for x in llmbar if x != "no verdict"]
    return {
        "summeval": {"items": len(summeval), "answered": len(answered), "agree": sum(answered),
                     "agreement_pct": round(100.0 * sum(answered) / len(answered), 1) if answered else None},
        "llmbar": {"pairs": len(llmbar), "answered": len(pairs), "separated": pairs.count("separated"),
                   "ties": pairs.count("tie"), "reversed": pairs.count("reversed"),
                   "separated_pct": round(100.0 * pairs.count("separated") / len(pairs), 1) if pairs else None},
    }


def validation_text(v: Optional[Dict[str, Any]]) -> Optional[str]:
    """One sentence for the report, or None when no validation ran."""
    if not v:
        return None
    se, lb = v.get("summeval") or {}, v.get("llmbar") or {}
    parts = []
    if se.get("answered"):
        parts.append(f"agreed with expert raters on {se['agree']} of {se['answered']} summaries "
                     f"({se['agreement_pct']:.0f}%, SummEval faithfulness)")
    if lb.get("answered"):
        parts.append(f"accepted the better answer and rejected the worse one in {lb['separated']} of "
                     f"{lb['answered']} pairs ({lb['separated_pct']:.0f}%, LLMBar; {lb['ties']} ties, "
                     f"{lb['reversed']} reversed)")
    if not parts:
        return ("The judge check against expert ratings gave no usable verdicts: the judge was down, or its "
                "replies could not be read as yes/no answers.")
    return "Checked against expert ratings, the judge " + "; it ".join(parts) + "."


def run_judge_validation(judge, workers: int = 2) -> Optional[Dict[str, Any]]:
    """Runs both checks with `workers` judge requests at a time; None
    without a judge or without the data."""
    if not judge.configured:
        return None
    summeval, _ = suites.load_items("judge_validation", "summeval")
    llmbar, _ = suites.load_items("judge_validation", "llmbar")
    if not summeval and not llmbar:
        log("  [judge check] data missing (run scripts/fetch_benchmark_data.py judge_validation)", color=YELLOW)
        return None
    log(f"  [judge check] {len(summeval)} SummEval summaries and {len(llmbar)} LLMBar pairs, in the background",
        color=CYAN)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        se = list(pool.map(lambda it: _summeval(judge, it), summeval))
        lb = list(pool.map(lambda p: _llmbar(judge, p), llmbar))
    result = summarise(se, lb)
    judge.validation = result
    log(f"  [judge check] {validation_text(result)}", color=GREEN)
    return result
