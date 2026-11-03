#!/usr/bin/env python3
"""
tests/test_domain_benchmarks.py
Unit tests verifying domain benchmark engines:
- IFEval rule verification
- USMLE / MedQA answer extraction
- PubMedQA decision extraction
- GSM8K numeric answer extraction
- Dataset loading integrity
"""

import unittest
from domain_benchmarks import (
    verify_ifeval_rules,
    extract_multiple_choice_answer,
    extract_pubmedqa_decision,
    extract_gsm8k_numeric_answer,
    load_benchmark_dataset,
)


class TestDomainBenchmarkDataLoading(unittest.TestCase):
    def test_datasets_exist_and_parse(self):
        usmle = load_benchmark_dataset("medical/usmle_clinical.json")
        self.assertGreaterEqual(len(usmle), 5)
        for item in usmle:
            self.assertIn("question", item)
            self.assertIn("options", item)
            self.assertIn(item["answer"], ["A", "B", "C", "D"])

        pubmed = load_benchmark_dataset("medical/biomedical_pubmed.json")
        self.assertGreaterEqual(len(pubmed), 5)
        for item in pubmed:
            self.assertIn("question", item)
            self.assertIn("context", item)
            self.assertIn(item["final_decision"], ["yes", "no", "maybe"])

        gsm8k = load_benchmark_dataset("math/gsm8k_reasoning.json")
        self.assertGreaterEqual(len(gsm8k), 5)
        for item in gsm8k:
            self.assertIn("question", item)
            self.assertIn("answer", item)

        history = load_benchmark_dataset("history/history_causality.json")
        self.assertGreaterEqual(len(history), 5)
        for item in history:
            self.assertIn("question", item)
            self.assertIn("options", item)
            self.assertIn(item["answer"], ["A", "B", "C", "D"])

        ifeval = load_benchmark_dataset("writing/ifeval_instructions.json")
        self.assertGreaterEqual(len(ifeval), 5)
        for item in ifeval:
            self.assertIn("prompt", item)
            self.assertIn("rules", item)

        crux = load_benchmark_dataset("reasoning/cruxeval_programs.json")
        self.assertGreaterEqual(len(crux), 2)
        for item in crux:
            self.assertIn("prompt", item)
            self.assertIn("expected_output", item)

        aime = load_benchmark_dataset("math/aime_olympiad.json")
        self.assertGreaterEqual(len(aime), 3)
        for item in aime:
            self.assertIn("question", item)
            self.assertIn("answer", item)

        gpqa = load_benchmark_dataset("reasoning/gpqa_diamond.json")
        self.assertGreaterEqual(len(gpqa), 5)
        for item in gpqa:
            self.assertIn("question", item)
            self.assertIn("answer", item)


class TestIFEvalVerification(unittest.TestCase):
    def test_ifeval_word_count_and_forbidden_words(self):
        rules = {
            "min_words": 10,
            "max_words": 20,
            "forbidden_words": ["bad", "wrong"],
        }
        valid_text = "This is a clean and compliant test sentence with exactly eleven words here."
        all_ok, passed, total, _ = verify_ifeval_rules(valid_text, rules)
        self.assertTrue(all_ok)
        self.assertEqual(passed, 3)

        # Fail forbidden word
        fail_text = "This is a bad sentence with enough words to satisfy length constraints."
        all_ok, passed, total, _ = verify_ifeval_rules(fail_text, rules)
        self.assertFalse(all_ok)
        self.assertEqual(passed, 2)

    def test_ifeval_bullet_and_boundary(self):
        rules = {
            "starts_with": "Hello world.",
            "ends_with": "Goodbye world.",
            "bullet_count": 2,
            "bullet_symbol": "-",
        }
        text = "Hello world.\n- Item one\n- Item two\nGoodbye world."
        all_ok, passed, total, details = verify_ifeval_rules(text, rules)
        self.assertTrue(all_ok)
        self.assertEqual(passed, 3)

    def test_ifeval_no_commas_and_keywords(self):
        rules = {
            "forbidden_chars": [","],
            "required_keyword_counts": {"deliverable": 2},
        }
        text = "The first deliverable is ready. The second deliverable will follow soon."
        all_ok, passed, total, _ = verify_ifeval_rules(text, rules)
        self.assertTrue(all_ok)
        self.assertEqual(passed, 2)

        # Has comma
        text_with_comma = "The first deliverable is ready, and the second deliverable is late."
        all_ok, passed, total, _ = verify_ifeval_rules(text_with_comma, rules)
        self.assertFalse(all_ok)
        self.assertEqual(passed, 1)


class TestAnswerExtractors(unittest.TestCase):
    def test_multiple_choice_extractor(self):
        self.assertEqual(extract_multiple_choice_answer("The correct answer is: B"), "B")
        self.assertEqual(extract_multiple_choice_answer("Option (C) is clearly the best choice."), "C")
        self.assertEqual(extract_multiple_choice_answer("**A**"), "A")
        self.assertEqual(extract_multiple_choice_answer("D. Acute pancreatitis"), "D")

    def test_multiple_choice_elimination_reasoning(self):
        # Must not falsely extract eliminated options like 'Option A is incorrect'
        t1 = "Option A (Treaty of Utrecht) is incorrect because it was signed in 1713. Option B (Peace of Westphalia) is the correct answer."
        self.assertEqual(extract_multiple_choice_answer(t1), "B")

        t2 = "We can eliminate options A and C as they are wrong. Therefore, the answer is D."
        self.assertEqual(extract_multiple_choice_answer(t2), "D")

        t3 = "A is ruled out. B is unlikely. Hence, option C is correct."
        self.assertEqual(extract_multiple_choice_answer(t3), "C")

    def test_pubmedqa_extractor(self):
        self.assertEqual(extract_pubmedqa_decision("Conclusion: yes"), "yes")
        self.assertEqual(extract_pubmedqa_decision("The final decision is **no** based on clinical trials."), "no")
        self.assertEqual(extract_pubmedqa_decision("Overall evidence suggests maybe."), "maybe")
        self.assertEqual(extract_pubmedqa_decision("Based on the landmark trials, yes, metformin reduces mortality."), "yes")

    def test_gsm8k_extractor(self):
        self.assertEqual(extract_gsm8k_numeric_answer("Natalia sold 48 + 24 = 72 clips. #### 72"), "72")
        self.assertEqual(extract_gsm8k_numeric_answer("Natalia made profit. #### $10"), "10")
        self.assertEqual(extract_gsm8k_numeric_answer("Therefore, the answer is $10."), "10")
        self.assertEqual(extract_gsm8k_numeric_answer("The remaining passengers equal 60."), "60")


if __name__ == "__main__":
    unittest.main()
