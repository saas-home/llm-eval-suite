#!/usr/bin/env python3
"""
domain_benchmarks.py
Shared answer extraction for graders: the final answer text of a reasoning
model, multiple-choice letters, and PubMedQA yes/no/maybe decisions. IFEval
(Test 40) is verified by ifeval.py.

Zero runtime external dependencies: pure Python standard library.
"""

import re
from typing import Any, Dict, List, Optional


# ==============================================================================
# 1. Answer-Text Selection (shared by every extractor below)
# ==============================================================================

_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Removes inline <think>...</think> reasoning (including an unterminated
    trailing block from a truncated response)."""
    return _THINK_BLOCK.sub("", text or "")


def final_answer_text(res: Dict[str, Any]) -> str:
    """Returns the text an answer should be graded from.

    The model's final answer lives in `content`. Reasoning models stream their
    scratch work separately (`reasoning`), and that scratch work routinely
    quotes the answer format or drafts answers it later rejects, so it is
    only used when the model produced no final content at all.
    """
    content = strip_think_blocks(res.get("content") or "").strip()
    if content:
        return content
    return strip_think_blocks(res.get("reasoning") or res.get("text") or "").strip()


def final_content(res: Dict[str, Any]) -> str:
    """The model's final deliverable only (code, essay): `content` with any
    inline <think> block removed. Unlike final_answer_text() there is no
    reasoning fallback -- a model that never produced content delivered
    nothing, and scratch work in its reasoning must not be graded."""
    return strip_think_blocks(res.get("content") or "").strip()


def _last_match(patterns: List["re.Pattern[str]"], text: str) -> Optional["re.Match[str]"]:
    """Returns the match that ends latest in `text` across all patterns."""
    best = None
    for pat in patterns:
        for m in pat.finditer(text):
            if best is None or m.end() > best.end():
                best = m
    return best


# ==============================================================================
# 2. Multiple Choice Answer Extraction (MedQA, MMLU History, GPQA)
# ==============================================================================
# The option letter is matched case-sensitively ([A-D]) so the English article
# 'a' is never read as option A; keywords are matched case-insensitively via
# scoped (?i:...) groups. A letter must stand alone (no trailing alphanumeric)
# so 'Answer: Because' or 'is Bacterial' never yield a letter.
def _mc_patterns(letters: str):
    """Compiled extraction patterns for an option-letter range ('ABCD' for
    MedQA/MMLU, 'ABCDEFGHIJ' for MMLU-Pro)."""
    cls = f"[{letters[0]}-{letters[-1]}]"
    letter = rf"\(?({cls})\)?(?![A-Za-z0-9])"
    return {
        # Priority 1: a declaration line ('Answer: B', '**Final Answer:** (C)').
        "declaration": re.compile(
            r"^[ \t>#*_-]*(?i:(?:final|correct)\s+answer|answer)[ \t*_]*[:\uff1a][ \t*_]*" + letter, re.MULTILINE),
        # Priority 2: in-sentence conclusions. No filler word between 'is' and
        # the letter, so 'the answer is not A' never matches.
        "conclusions": [
            re.compile(r"(?i:\b(?:final\s+)?answer)[ \t*_]*[:\uff1a][ \t*_]*" + letter),
            re.compile(r"(?i:\b(?:answer|choice|option|diagnosis)\s+is)[ \t:*_]*" + letter),
            re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*(" + cls + r")\s*\}"),
            re.compile(r"(?i:\b(?:option|choice))\s*\(?(" + cls + r")\)?(?:\s*\([^)]*\))?"
                       r"\s+(?i:is\s+(?:clearly\s+|definitely\s+)?(?:the\s+)?(?:correct|best|right)\b)"),
        ],
        # Priority 3: a line holding nothing but the letter ('B', '(B)', '**B**').
        "bare": re.compile(r"^[ \t*_]*\(?(" + cls + r")\)?[ \t*_.]*$", re.MULTILINE),
        # Priority 4: the response opens with an option line ('D. Acute pancreatitis').
        "leading": re.compile(r"^\s*\(?(" + cls + r")[.:)]\s"),
    }


# The option letter is matched case-sensitively so the English article 'a' is
# never read as option A; keywords are case-insensitive via scoped (?i:...)
# groups. A letter must stand alone (no trailing alphanumeric).
_MC_CACHE: Dict[str, Dict[str, Any]] = {}


def extract_multiple_choice_answer(text: str, letters: str = "ABCD") -> Optional[str]:
    """Extracts the chosen option letter, or None when no unambiguous choice
    is declared. Within a priority tier the latest declaration wins, so a
    model that revises its answer is graded on its final choice."""
    text_clean = strip_think_blocks(text).strip()
    if not text_clean:
        return None
    pats = _MC_CACHE.get(letters)
    if pats is None:
        pats = _MC_CACHE[letters] = _mc_patterns(letters)
    for patterns in ([pats["declaration"]], pats["conclusions"], [pats["bare"]]):
        m = _last_match(patterns, text_clean)
        if m:
            return m.group(1)
    m = pats["leading"].match(text_clean)
    return m.group(1) if m else None


# ==============================================================================
# 3. PubMedQA Biomedical Decision Extraction
# ==============================================================================
# A declaration needs a colon or 'is' before the decision, so the instruction
# echo 'I need to answer yes, no, or maybe' is never a declaration. A decision
# followed by a quote ("'Conclusion: no'") is an echo of the requested format.
_PUBMED_DECLARATION = re.compile(
    r"\b(?:final\s+decision|final\s+answer|answer|conclusion|decision|verdict)"
    r"(?:\s*[:\uff1a]|\s+is)[\s*_]*(yes|no|maybe)\b(?![*_]*['\"\u2019\u201d])"
)
_PUBMED_WORD = re.compile(r"\b(yes|no|maybe)\b")


def _sole_decision(text: str) -> Optional[str]:
    """The decision word if exactly one distinct one appears, else None."""
    found = set(_PUBMED_WORD.findall(text))
    return found.pop() if len(found) == 1 else None


def extract_pubmedqa_decision(text: str) -> Optional[str]:
    """Extracts the 'yes' / 'no' / 'maybe' decision, or None if ambiguous."""
    text_lower = strip_think_blocks(text).lower().strip()
    if not text_lower:
        return None

    # 1. Latest explicit declaration ('Conclusion: no', 'final decision is yes').
    declarations = _PUBMED_DECLARATION.findall(text_lower)
    if declarations:
        return declarations[-1]

    # 2. The response opens with the decision.
    m = re.match(r"^[\s*_]*(yes|no|maybe)\b", text_lower)
    if m:
        return m.group(1)

    # 3. The final sentence, then the whole text, names exactly one decision.
    sentences = [s for s in re.split(r"[.!?\n]+", text_lower) if s.strip()]
    if sentences:
        last = _sole_decision(sentences[-1])
        if last:
            return last
    return _sole_decision(text_lower)


