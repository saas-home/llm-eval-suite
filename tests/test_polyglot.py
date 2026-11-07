#!/usr/bin/env python3
"""
Tests for the polyglot compile-and-run graders (polyglot.py, data/polyglot/).

Pack-integrity and assembly tests run anywhere. Tests that compile code are
skipped unless the language's toolchain is installed (they run inside the
Docker image); the full discrimination check is scripts/verify_polyglot_packs.py.
"""

import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import languages
import polyglot

TESTS = ["surgical_git_diff", "multifile_context", "unit_test_synthesis", "strict_typing", "refactor_invariants"]
LANGS = ["typescript", "go", "java", "rust", "csharp"]


class TestPackIntegrity(unittest.TestCase):
    """Every non-Python language has a complete, self-consistent pack."""

    def test_every_pack_exists_with_reference_and_negative(self):
        for test_key in TESTS:
            for lang in LANGS:
                with self.subTest(test=test_key, lang=lang):
                    self.assertTrue(polyglot.has_pack(test_key, lang))
                    pack = polyglot.load_pack(test_key, lang)
                    self.assertIn(pack["mode"], ("program", "unit_tests"))
                    self.assertEqual(len(pack["reference"]), 1)
                    self.assertEqual(len(pack["negative"]), 1)
                    self.assertNotEqual(pack["reference"], pack["negative"])

    def test_mutants_apply_exactly_once(self):
        """A mutant whose target text is missing would silently test nothing."""
        for lang in LANGS:
            pack = polyglot.load_pack("unit_test_synthesis", lang)
            self.assertEqual(len(pack["mutants"]), 3)
            for mutant in pack["mutants"]:
                with self.subTest(lang=lang, mutant=mutant["name"]):
                    self.assertEqual(pack["support"][mutant["file"]].count(mutant["replace"][0]), 1)

    def test_python_mutants_apply_exactly_once(self):
        source = languages.UNIT_TEST_FIXTURES["python"]["source"]
        for mutant in languages.UNIT_TEST_FIXTURES["python"]["mutants"]:
            self.assertEqual(source.count(mutant["replace"][0]), 1, mutant["name"])

    def test_prompts_contain_the_code_the_harness_compiles(self):
        """Prompts are generated from the packs, so they cannot drift apart."""
        for lang in LANGS:
            with self.subTest(lang=lang):
                for name, body in polyglot.load_pack("multifile_context", lang)["support"].items():
                    self.assertIn(body.strip(), languages.MULTIFILE_FIXTURES[lang]["prompt"], name)
                self.assertIn(languages.UNIT_TEST_FIXTURES[lang]["source"], languages.UNIT_TEST_FIXTURES[lang]["prompt"])
                self.assertIn(languages.REFACTOR_FIXTURES[lang]["legacy"], languages.REFACTOR_FIXTURES[lang]["prompt"])

    def test_refactor_prompts_include_legacy_code(self):
        """Previously the non-Python refactor prompts contained no code at all."""
        for lang in LANGS:
            self.assertNotIn("{ ... }", languages.REFACTOR_FIXTURES[lang]["prompt"], lang)
            self.assertIn("```", languages.REFACTOR_FIXTURES[lang]["prompt"], lang)

    def test_diff_negative_is_the_original_buggy_file(self):
        for lang in LANGS:
            pack = polyglot.load_pack("surgical_git_diff", lang)
            self.assertEqual(next(iter(pack["negative"].values())).strip(),
                             languages.DIFF_FIXTURES[lang]["code"].strip(), lang)


class TestAssembly(unittest.TestCase):
    def test_go_package_is_normalized(self):
        pack = {"go_package": "authsvc"}
        self.assertTrue(polyglot.normalize_answer(pack, "package main\n\nfunc F() {}").startswith("package authsvc"))

    def test_java_package_line_is_stripped(self):
        out = polyglot.normalize_answer({"strip_java_package": True}, "package com.x;\npublic class A {}")
        self.assertNotIn("package", out)

    def test_support_skipped_when_answer_defines_the_type(self):
        pack = {"support": {"Models.cs": "public record User(string Id);"}, "harness": {},
                "answer_file": "A.cs", "skip_support_if_defined": {"Models.cs": r"record\s+User\b"}}
        self.assertNotIn("Models.cs", polyglot._assemble(pack, "public record User(string Id);"))
        self.assertIn("Models.cs", polyglot._assemble(pack, "public class A {}"))

    def test_rust_tests_appended_unless_whole_file_returned(self):
        pack = polyglot.load_pack("unit_test_synthesis", "rust")
        only_tests = polyglot._assemble(pack, "#[cfg(test)] mod tests {}")["lib.rs"]
        self.assertIn("impl<'a, G: StripeGateway", only_tests)
        whole = pack["support"]["payment.rs"] + "\n#[cfg(test)] mod tests {}"
        self.assertEqual(polyglot._assemble(pack, whole)["lib.rs"].count("pub struct PaymentProcessor"), 1)

    def test_csharp_namespace_placeholder(self):
        pack = polyglot.load_pack("refactor_invariants", "csharp")
        with_ns = polyglot._assemble(pack, "namespace Orders;\npublic record Record();")["Program.cs"]
        self.assertIn("Orders.Record", with_ns)
        without_ns = polyglot._assemble(pack, "public record Record();")["Program.cs"]
        self.assertIn("global::Record", without_ns)

    def test_forbidden_ignores_go_type_constraints_and_comments(self):
        pack = polyglot.load_pack("strict_typing", "go")
        ok = "type P[T any] struct{}\n// any comment mentioning any\nfunc M[T any, U any]() {}"
        self.assertEqual(polyglot.forbidden_hits(pack, ok), [])
        self.assertTrue(polyglot.forbidden_hits(pack, "func F(x interface{}) {}"))

    def test_missing_toolchain_is_reported_not_passed(self):
        with mock.patch.object(polyglot, "toolchain_available", return_value=False):
            res = polyglot.run_program("go", {})
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage"], "unavailable")


class TestXunitReport(unittest.TestCase):
    """C# unit tests are counted from the xUnit v3 XML report."""

    def _report(self, body):
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(body)
        self.addCleanup(os.remove, path)
        return path

    def test_counts_summed_over_assemblies(self):
        path = self._report('<assemblies><assembly total="3" passed="3" errors="0"/>'
                            '<assembly total="2" passed="1" errors="0"/></assemblies>')
        self.assertEqual(polyglot._xunit_counts(path), (5, 4))

    def test_errors_fail_the_run(self):
        """A fixture that throws reports errors, not failed tests; it must not pass."""
        path = self._report('<assemblies><assembly total="3" passed="3" errors="1"/></assemblies>')
        ran, passed = polyglot._xunit_counts(path)
        self.assertNotEqual(ran, passed)

    def test_missing_or_broken_report_is_none(self):
        self.assertIsNone(polyglot._xunit_counts("/nonexistent/report.xml"))
        self.assertIsNone(polyglot._xunit_counts(self._report("<assemblies>")))
        self.assertIsNone(polyglot._xunit_counts(self._report("<assemblies/>")))


class TestGoModVersion(unittest.TestCase):
    """go.mod declares the installed Go version, so newer language features compile."""

    def setUp(self):
        polyglot._go_language_version.cache_clear()
        self.addCleanup(polyglot._go_language_version.cache_clear)

    def test_uses_installed_go_version(self):
        done = mock.Mock(stdout="go1.27.3\n")
        with mock.patch.object(polyglot.subprocess, "run", return_value=done):
            self.assertEqual(polyglot._go_mod(), "module evalpkg\n\ngo 1.27\n")

    def test_falls_back_when_go_is_missing(self):
        with mock.patch.object(polyglot.subprocess, "run", side_effect=OSError("no go")):
            self.assertIn("go 1.22", polyglot._go_mod())


class TestPythonMutationTesting(unittest.TestCase):
    """Test 23 (Python): passing tests that miss seeded bugs are not a PASS."""

    SOURCE = languages.UNIT_TEST_FIXTURES["python"]["source"]
    HEADER = ("import unittest\nfrom unittest.mock import MagicMock\n"
              "class T(unittest.TestCase):\n")

    def test_weak_suite_is_partial(self):
        weak = self.HEADER + (
            "    def test_ok(self):\n"
            "        g = MagicMock(); g.charge.return_value = {'status': 'succeeded', 'id': 'c'}\n"
            "        self.assertTrue(PaymentProcessor(g, MagicMock()).process_order('e', 5, 't'))\n"
            "    def test_neg(self):\n"
            "        with self.assertRaises(ValueError): PaymentProcessor(MagicMock(), MagicMock()).process_order('e', -1, 't')\n"
            "    def test_trivial(self):\n"
            "        self.assertTrue(True)\n")
        v = languages.verify_unit_test(weak, "python", self.SOURCE)
        self.assertEqual(v["unit_tests_passed_count"], 3)
        self.assertLess(v["mutants_killed"], v["mutants_total"])
        self.assertEqual(v["status"], "PARTIAL")

    def test_any_import_module_name_is_served(self):
        """Seen live: `from production_code import PaymentProcessor` crashed
        the run (0 tests) although the prompt never named the module."""
        code = self.HEADER.replace("import unittest\n", "import unittest\nfrom production_code import PaymentProcessor\n") + (
            "    def test_neg(self):\n"
            "        with self.assertRaises(ValueError): PaymentProcessor(MagicMock(), MagicMock()).process_order('e', -1, 't')\n")
        ran, passed = languages._run_python_unit_tests(code, self.SOURCE)
        self.assertEqual((ran, passed), (1, 1))

    def test_strong_suite_passes(self):
        strong = self.HEADER + (
            "    def setUp(self):\n"
            "        self.g, self.n = MagicMock(), MagicMock()\n"
            "        self.p = PaymentProcessor(self.g, self.n)\n"
            "    def test_success(self):\n"
            "        self.g.charge.return_value = {'status': 'succeeded', 'id': 'ch_1'}\n"
            "        self.assertTrue(self.p.process_order('a@x.io', 5, 'tok'))\n"
            "        self.n.send_receipt.assert_called_once_with('a@x.io', 'ch_1')\n"
            "    def test_declined(self):\n"
            "        self.g.charge.return_value = {'status': 'declined', 'id': 'ch_2'}\n"
            "        self.assertFalse(self.p.process_order('a@x.io', 5, 'tok'))\n"
            "        self.n.send_receipt.assert_not_called()\n"
            "    def test_negative(self):\n"
            "        with self.assertRaises(ValueError): self.p.process_order('a@x.io', -1, 'tok')\n"
            "        self.g.charge.assert_not_called()\n")
        v = languages.verify_unit_test(strong, "python", self.SOURCE)
        self.assertEqual((v["mutants_killed"], v["mutants_total"]), (3, 3))
        self.assertEqual(v["status"], "PASS")


class TestPythonRefactorDifferential(unittest.TestCase):
    def test_dropping_rounding_is_caught(self):
        code = ("def process_order_data(records):\n"
                "    if records is None: return []\n"
                "    out = []\n"
                "    for r in records:\n"
                "        if not r or 'qty' not in r or 'price' not in r: continue\n"
                "        if r['qty'] <= 0 or r['price'] < 0: continue\n"
                "        t = r['qty'] * r['price']\n"
                "        if 'discount' in r and 0 < r['discount'] <= 1: t *= 1 - r['discount']\n"
                "        out.append({'id': r.get('id', 'N/A'), 'total': t})\n"
                "    return out\n")
        v = languages.verify_refactoring(code, "python")
        self.assertFalse(v["invariants_preserved"])
        self.assertEqual(v["status"], "PARTIAL")


@unittest.skipUnless(all(polyglot.toolchain_available(l) for l in LANGS), "polyglot toolchains not installed")
class TestPacksDiscriminate(unittest.TestCase):
    """Inside the Docker image: references pass and negatives fail."""

    def test_reference_and_negative(self):
        for test_key in TESTS:
            for lang in LANGS:
                pack = polyglot.load_pack(test_key, lang)
                good = polyglot.grade_answer(test_key, lang, next(iter(pack["reference"].values())))
                bad = polyglot.grade_answer(test_key, lang, next(iter(pack["negative"].values())))
                with self.subTest(test=test_key, lang=lang):
                    if test_key == "unit_test_synthesis":
                        self.assertEqual(len(good["mutants_killed"]), good["mutants_total"])
                        self.assertLess(len(bad["mutants_killed"]), bad["mutants_total"])
                    else:
                        self.assertTrue(good["ok"], good)
                        self.assertFalse(bad["ok"], bad)


if __name__ == "__main__":
    unittest.main()
