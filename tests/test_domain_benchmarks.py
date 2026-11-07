#!/usr/bin/env python3
"""
tests/test_domain_benchmarks.py
Unit tests verifying domain benchmark engines:
- USMLE / MedQA answer extraction
- PubMedQA decision extraction
- Dataset loading integrity
"""

import unittest
from domain_benchmarks import (
    extract_multiple_choice_answer,
    extract_pubmedqa_decision,
)


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


if __name__ == "__main__":
    unittest.main()
