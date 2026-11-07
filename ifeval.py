#!/usr/bin/env python3
"""
ifeval.py -- verifier for the full Google IFEval benchmark (541 prompts).

A standard-library port of the instruction checks in Google Research's
instruction_following_eval (Apache-2.0,
https://github.com/google-research/google-research/tree/master/instruction_following_eval).
Differences from the reference implementation, all documented:
  * `language:response_language` needs the `langdetect` package; prompts that
    use it are excluded (reported as `excluded_prompts`).
  * Sentence splitting uses a regex splitter equivalent to the reference's
    abbreviation-aware splitter for common text; rare edge cases may differ.
  * `change_case:english_capital/lowercase` check only the case (the
    reference also confirms the text is English with langdetect).

Scores follow the paper: prompt-level and instruction-level accuracy, each
in "strict" mode and "loose" mode (loose also accepts the response with its
first/last line removed and/or markdown asterisks stripped).
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

UNSUPPORTED = {"language:response_language"}


def _cmp(count: int, relation: str, target: int) -> bool:
    if relation == "less than":
        return count < target
    if relation == "at least":
        return count >= target
    raise ValueError(f"unknown relation {relation!r}")


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


_ABBREV = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Inc|Ltd|Co|U\.S|a\.m|p\.m)\."


def count_sentences(text: str) -> int:
    protected = re.sub(_ABBREV, lambda m: m.group(0).replace(".", "<DOT>"), text)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n+", protected) if p.strip()]
    return len(parts)


# ----------------------------------------------------------------------------
# Instruction checks: (response, kwargs) -> bool
# ----------------------------------------------------------------------------
def _keywords_existence(r, k):
    return all(re.search(re.escape(w), r, re.IGNORECASE) for w in k["keywords"])


def _keywords_frequency(r, k):
    n = len(re.findall(re.escape(k["keyword"]), r, re.IGNORECASE))
    return _cmp(n, k["relation"], k["frequency"])


def _forbidden_words(r, k):
    return not any(re.search(r"\b" + re.escape(w) + r"\b", r, re.IGNORECASE) for w in k["forbidden_words"])


def _letter_frequency(r, k):
    return _cmp(r.lower().count(k["letter"].lower()), k["let_relation"], k["let_frequency"])


def _number_sentences(r, k):
    return _cmp(count_sentences(r), k["relation"], k["num_sentences"])


def _number_paragraphs(r, k):
    paragraphs = re.split(r"\s?\*\*\*\s?", r)
    num = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i == 0 or i == len(paragraphs) - 1:
                num -= 1
            else:
                return False
    return num == k["num_paragraphs"]


def _number_words(r, k):
    return _cmp(count_words(r), k["relation"], k["num_words"])


def _nth_paragraph_first_word(r, k):
    paragraphs = [p for p in re.split(r"\n\n", r) if p.strip()]
    if len(paragraphs) != k["num_paragraphs"]:
        return False
    n = k["nth_paragraph"]
    if n > len(paragraphs):
        return False
    words = paragraphs[n - 1].strip().split()
    if not words:
        return False
    first = re.sub(r"[^\w]", "", words[0]).lower()
    return first == k["first_word"].lower()


def _number_placeholders(r, k):
    return len(re.findall(r"\[.*?\]", r)) >= k["num_placeholders"]


def _postscript(r, k):
    marker = k["postscript_marker"].strip()
    if marker.upper() == "P.S.":
        pat = r"\s*p\.\s?s\..*$"
    elif marker.upper() == "P.P.S":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    else:
        pat = r"\s*" + re.escape(marker.lower()) + r".*$"
    return bool(re.findall(pat, r.lower(), re.MULTILINE))


def _number_bullet_lists(r, k):
    stars = re.findall(r"^\s*\*[^\*].*$", r, re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", r, re.MULTILINE)
    return len(stars) + len(dashes) == k["num_bullets"]


def _constrained_response(r, k):
    return any(opt in r for opt in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


def _number_highlighted_sections(r, k):
    n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", r) if h.strip("*").strip())
    n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", r) if h[2:-2].strip())
    return n >= k["num_highlights"]


def _multiple_sections(r, k):
    sections = re.split(r"\s?" + re.escape(k["section_spliter"]) + r"\s?\d+\s?", r)
    return len(sections) - 1 >= k["num_sections"]


def _json_format(r, k):
    value = r.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.strip()
    if value.endswith("```"):
        value = value[:-3].strip()
    try:
        json.loads(value)
        return True
    except ValueError:
        return False


def _title(r, k):
    return any(t.strip("<>").strip() for t in re.findall(r"<<[^\n]+>>", r))


def _two_responses(r, k):
    parts = r.split("******")
    if len(parts) != 2:
        return False
    a, b = parts[0].strip(), parts[1].strip()
    return bool(a) and bool(b) and a != b


def _repeat_prompt(r, k):
    return r.strip().lower().startswith(k["prompt_to_repeat"].strip().lower())


def _end_checker(r, k):
    return r.strip().strip('"').lower().endswith(k["end_phrase"].strip().lower())


def _capital_word_frequency(r, k):
    n = len([w for w in re.findall(r"\w+", r) if w.isupper()])
    return _cmp(n, k["capital_relation"], k["capital_frequency"])


def _english_capital(r, k):
    return r.isupper()


def _english_lowercase(r, k):
    return r.islower()


def _no_comma(r, k):
    return "," not in r


def _quotation(r, k):
    v = r.strip()
    return len(v) > 1 and v[0] == '"' and v[-1] == '"'


CHECKS: Dict[str, Callable[[str, Dict[str, Any]], bool]] = {
    "keywords:existence": _keywords_existence,
    "keywords:frequency": _keywords_frequency,
    "keywords:forbidden_words": _forbidden_words,
    "keywords:letter_frequency": _letter_frequency,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:number_words": _number_words,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _number_bullet_lists,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:number_highlighted_sections": _number_highlighted_sections,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:end_checker": _end_checker,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "change_case:english_capital": _english_capital,
    "change_case:english_lowercase": _english_lowercase,
    "punctuation:no_comma": _no_comma,
    "startend:quotation": _quotation,
}


def _loose_variants(response: str) -> List[str]:
    lines = response.split("\n")
    first_removed = "\n".join(lines[1:]).strip()
    last_removed = "\n".join(lines[:-1]).strip()
    both_removed = "\n".join(lines[1:-1]).strip()
    variants = [response, response.replace("*", ""), first_removed, first_removed.replace("*", ""),
                last_removed, last_removed.replace("*", ""), both_removed, both_removed.replace("*", "")]
    return [v for v in variants if v.strip()]


def supported(item: Dict[str, Any]) -> bool:
    return not any(i in UNSUPPORTED for i in item["instruction_ids"])


def evaluate(response: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """Strict and loose results for one IFEval prompt."""
    strict, loose = [], []
    for inst_id, kwargs in zip(item["instruction_ids"], item["kwargs"]):
        check = CHECKS[inst_id]
        kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
        try:
            s_ok = bool(response.strip()) and check(response, kw)
        except (KeyError, ValueError, TypeError):
            s_ok = False
        l_ok = s_ok
        if not l_ok:
            for variant in _loose_variants(response):
                try:
                    if check(variant, kw):
                        l_ok = True
                        break
                except (KeyError, ValueError, TypeError):
                    continue
        strict.append(s_ok)
        loose.append(l_ok)
    return {"strict": strict, "loose": loose, "prompt_strict": all(strict), "prompt_loose": all(loose),
            "instructions": item["instruction_ids"]}
