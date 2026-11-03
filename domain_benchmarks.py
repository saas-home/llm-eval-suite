#!/usr/bin/env python3
"""
domain_benchmarks.py
Zero-dependency domain benchmark evaluation engine for:
1. Google IFEval Verifiable Instruction Following (Writing)
2. Medical Research & Clinical Competence (MedQA/USMLE + PubMedQA)
3. Elementary & Multi-Step Mathematics (OpenAI GSM8K)
4. Historical Causality & Humanities Analysis (MMLU History)

Zero runtime external dependencies: pure Python standard library.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_benchmark_dataset(rel_path: str) -> List[Dict[str, Any]]:
    """Loads a JSON dataset from the repository's data directory."""
    full_path = os.path.join(DATA_DIR, rel_path)
    if not os.path.exists(full_path):
        # Fallback to local execution directory
        full_path = os.path.join(os.getcwd(), "data", rel_path)
    if not os.path.exists(full_path):
        return []
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 1. Google IFEval Constraint Verification Engine
# ==============================================================================

def verify_ifeval_rules(text: str, rules: Dict[str, Any]) -> Tuple[bool, int, int, List[str]]:
    """
    Deterministically validates text against IFEval rules.
    Returns: (all_passed, passed_count, total_count, details)
    """
    # Strip thinking blocks (<think>...</think>) if present so internal thoughts
    # are not evaluated as generated user-facing text.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_clean = text.strip()
    words = [w for w in re.split(r"\s+", text_clean) if w]
    word_count = len(words)
    details = []
    passed = 0
    total = 0

    # Rule: min_words
    if "min_words" in rules:
        total += 1
        limit = rules["min_words"]
        if word_count >= limit:
            passed += 1
            details.append(f"PASS: Word count {word_count} >= {limit}")
        else:
            details.append(f"FAIL: Word count {word_count} < min {limit}")

    # Rule: max_words
    if "max_words" in rules:
        total += 1
        limit = rules["max_words"]
        if word_count <= limit:
            passed += 1
            details.append(f"PASS: Word count {word_count} <= {limit}")
        else:
            details.append(f"FAIL: Word count {word_count} > max {limit}")

    # Rule: bullet_count and bullet_symbol
    if "bullet_count" in rules:
        total += 1
        expected = rules["bullet_count"]
        sym = rules.get("bullet_symbol", "-")
        bullet_lines = [
            line for line in text.splitlines()
            if line.strip().startswith(sym)
        ]
        if len(bullet_lines) == expected:
            passed += 1
            details.append(f"PASS: Bullet count {len(bullet_lines)} == {expected} ('{sym}')")
        else:
            details.append(f"FAIL: Bullet count {len(bullet_lines)} != expected {expected}")

    # Rule: forbidden_words
    if "forbidden_words" in rules:
        total += 1
        found_forbidden = []
        for word in rules["forbidden_words"]:
            if re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
                found_forbidden.append(word)
        if not found_forbidden:
            passed += 1
            details.append("PASS: No forbidden words detected")
        else:
            details.append(f"FAIL: Forbidden word(s) detected: {found_forbidden}")

    # Rule: starts_with
    if "starts_with" in rules:
        total += 1
        prefix = rules["starts_with"]
        if text_clean.startswith(prefix) or text_clean.lstrip('"\'*# \n\r\t').startswith(prefix):
            passed += 1
            details.append(f"PASS: Starts with '{prefix}'")
        else:
            details.append(f"FAIL: Does not start with '{prefix}'")

    # Rule: ends_with
    if "ends_with" in rules:
        total += 1
        suffix = rules["ends_with"]
        if text_clean.endswith(suffix) or text_clean.rstrip('"\'* \n\r\t').endswith(suffix.rstrip('.')):
            passed += 1
            details.append(f"PASS: Ends with '{suffix}'")
        else:
            details.append(f"FAIL: Does not end with '{suffix}'")

    # Rule: paragraph_count
    if "paragraph_count" in rules:
        total += 1
        expected = rules["paragraph_count"]
        paras = [p for p in re.split(r"\n\s*\n", text_clean) if p.strip()]
        if len(paras) == expected:
            passed += 1
            details.append(f"PASS: Paragraph count {len(paras)} == {expected}")
        else:
            details.append(f"FAIL: Paragraph count {len(paras)} != expected {expected}")

    # Rule: forbidden_chars
    if "forbidden_chars" in rules:
        total += 1
        found_chars = [c for c in rules["forbidden_chars"] if c in text]
        if not found_chars:
            passed += 1
            details.append("PASS: No forbidden characters detected")
        else:
            details.append(f"FAIL: Forbidden characters detected: {found_chars}")

    # Rule: required_keyword_counts
    if "required_keyword_counts" in rules:
        total += 1
        all_kw_ok = True
        kw_failures = []
        for kw, req_count in rules["required_keyword_counts"].items():
            matches = len(re.findall(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE))
            if matches < req_count:
                all_kw_ok = False
                kw_failures.append(f"'{kw}': {matches}/{req_count}")
        if all_kw_ok:
            passed += 1
            details.append("PASS: All required keyword frequency counts satisfied")
        else:
            details.append(f"FAIL: Keyword frequency insufficient: {kw_failures}")

    # Rule: header_count and markdown_header_level
    if "header_count" in rules:
        total += 1
        expected = rules["header_count"]
        lvl = rules.get("markdown_header_level", "## ")
        hdr_lines = [l for l in text.splitlines() if l.strip().startswith(lvl)]
        if len(hdr_lines) == expected:
            passed += 1
            details.append(f"PASS: Header count {len(hdr_lines)} == {expected} ('{lvl}')")
        else:
            details.append(f"FAIL: Header count {len(hdr_lines)} != expected {expected}")

    # Rule: min_bold_highlights
    if "min_bold_highlights" in rules:
        total += 1
        req_bold = rules["min_bold_highlights"]
        bolds = re.findall(r"\*\*[^*]+\*\*", text)
        if len(bolds) >= req_bold:
            passed += 1
            details.append(f"PASS: Bold highlights count {len(bolds)} >= {req_bold}")
        else:
            details.append(f"FAIL: Bold highlights count {len(bolds)} < {req_bold}")

    # Rule: sentence_count
    if "sentence_count" in rules:
        total += 1
        expected = rules["sentence_count"]
        # Split by sentence-ending punctuation followed by space or end-of-string
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text_clean) if s.strip()]
        if len(sents) == expected:
            passed += 1
            details.append(f"PASS: Sentence count {len(sents)} == {expected}")
        else:
            details.append(f"FAIL: Sentence count {len(sents)} != expected {expected}")

    # Rule: ends_with_question
    if "ends_with_question" in rules:
        total += 1
        if text_clean.endswith("?"):
            passed += 1
            details.append("PASS: Ends with question mark '?'")
        else:
            details.append("FAIL: Does not end with question mark '?'")

    all_passed = (passed == total) if total > 0 else True
    return all_passed, passed, total, details


# ==============================================================================
# 2. Multiple Choice Answer Extraction (MedQA & MMLU History)
# ==============================================================================

def extract_multiple_choice_answer(text: str) -> Optional[str]:
    """
    Deterministically extracts option letter (A, B, C, D) from LLM output.
    Uses multi-stage regex prioritizing explicit conclusion declarations over
    casual elimination mentions (e.g. 'Option A is incorrect').
    """
    text_clean = text.strip()
    if not text_clean:
        return None

    def is_negative_mention(line: str, letter: str) -> bool:
        neg_pat = rf"\b(?:option\s+|choice\s+)?{letter}\b(?:\s*\([^)]*\))?[^\.\n]*?\b(?:incorrect|false|ruled\s+out|unlikely|wrong|not\s+the|not\s+correct)\b"
        return bool(re.search(neg_pat, line, re.IGNORECASE))

    # Priority 1: Explicit final answer declarations
    strong_patterns = [
        r"(?:the\s+)?(?:correct\s+)?(?:answer|choice|option|diagnosis)\s+is(?:\s+[a-zA-Z]+)?\s*[:\*\s]*\(?([A-D])\b",
        r"(?:^|\n)\s*(?:final\s+)?(?:answer|conclusion)\s*[:\*\s]+\(?([A-D])\b",
        r"\b(?:option|choice)?\s*([A-D])\b(?:\s*\([^)]*\))?\s+is(?:\s+[a-zA-Z]+)?\s+(?:the\s+)?(?:correct|best|right)\b",
        r"(?:therefore|thus|hence|in\s+conclusion)[^\.\n]*?\b(?:answer|choice|option)\s+is(?:\s+[a-zA-Z]+)?\s*[:\*\s]*\(?([A-D])\b",
    ]
    for pat in strong_patterns:
        matches = list(re.finditer(pat, text_clean, re.IGNORECASE))
        if matches:
            # Use the last conclusive declaration
            return matches[-1].group(1).upper()

    # Priority 2: Check lines from bottom to top for explicit answer declaration or isolated letter
    lines = [l.strip() for l in text_clean.splitlines() if l.strip()]
    for line in reversed(lines):
        if re.search(r"\b(?:answer|conclusion|choice|verdict)\b", line, re.IGNORECASE):
            m = re.search(r"\b([A-D])\b", line)
            if m and not is_negative_mention(line, m.group(1)):
                return m.group(1).upper()
        clean = line.strip().strip("*._- ")
        if re.match(r"^\(?([A-D])\)?$", clean, re.IGNORECASE):
            return clean.strip("()").upper()

    # Priority 3: Bold / Boxed answer declaration
    boxed = list(re.finditer(r"(?:\*\*|\[|\()([A-D])(?:\*\*|\]|\))", text_clean))
    if boxed:
        for b in reversed(boxed):
            span_start = max(0, b.start() - 30)
            span_end = min(len(text_clean), b.end() + 30)
            context = text_clean[span_start:span_end]
            if not re.search(r"\b(?:incorrect|wrong|false)\b", context, re.IGNORECASE):
                return b.group(1).upper()

    # Priority 4: First line starting with letter (e.g. 'A. Acute pancreatitis')
    if lines:
        first_line = lines[0]
        m = re.match(r"^([A-D])(?:\.|\:|\))\s", first_line)
        if m:
            return m.group(1).upper()

    return None


# ==============================================================================
# 3. PubMedQA Biomedical Decision Extraction
# ==============================================================================

def extract_pubmedqa_decision(text: str) -> Optional[str]:
    """
    Extracts binary/ternary decision ('yes', 'no', 'maybe') from PubMedQA response.
    """
    text_lower = text.lower().strip()

    # Pattern 1: Explicit conclusion/answer declaration
    match = re.search(r"(?:final\s+decision|answer|conclusion|decision|verdict)\s*[:\*\s]*(yes|no|maybe)\b", text_lower)
    if match:
        return match.group(1)

    # Pattern 2: First word of response is yes/no/maybe
    first_word_match = re.match(r"^(yes|no|maybe)[\b\.\,\!\:\;\s]", text_lower)
    if first_word_match:
        return first_word_match.group(1)

    # Pattern 3: Look at the last sentence / conclusion
    sentences = [s.strip() for s in re.split(r"[.\n]+", text_lower) if s.strip()]
    if sentences:
        last_sentence = sentences[-1]
        match = re.search(r"\b(yes|no|maybe)\b", last_sentence)
        if match:
            return match.group(1)

    # Pattern 4: Any isolated yes / no / maybe in the text
    match = re.search(r"\b(yes|no|maybe)\b", text_lower)
    if match:
        return match.group(1)

    return None


# ==============================================================================
# 4. GSM8K Numeric Solution Extraction
# ==============================================================================

def extract_gsm8k_numeric_answer(text: str) -> Optional[str]:
    """
    Extracts final numeric solution from GSM8K output.
    Looks for standard '#### <num>', 'The answer is <num>', or final number.
    """
    text_clean = text.strip()

    # Standard GSM8K delimiter: '#### 72' or '#### $10'
    hash_match = re.search(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text_clean)
    if hash_match:
        return hash_match.group(1).replace(",", "")

    # Explicit answer declaration
    explicit_match = re.search(
        r"(?:the\s+answer\s+is|answer\s+is|the\s+final\s+answer\s+is|result\s+is|equals?|=)\s*[:\*\s]*\$?(-?[\d,]+(?:\.\d+)?)\b",
        text_clean,
        re.IGNORECASE,
    )
    if explicit_match:
        return explicit_match.group(1).replace(",", "")

    # Trailing boxed number or last line number
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text_clean)
    if numbers:
        # Return the last number formatted
        return numbers[-1].replace(",", "")

    return None
