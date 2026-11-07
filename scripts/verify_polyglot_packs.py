#!/usr/bin/env python3
"""
Self-check for the polyglot test packs (data/polyglot/<test>/<lang>/).

For every pack, the known-good reference answer must be graded as passing and
the known-bad negative answer as failing; for unit-test packs, the reference
suite must also kill every seeded mutant while the negative suite must not
earn a pass. A harness that cannot tell good from bad would silently produce
wrong benchmark numbers, so run this after changing any pack:

    docker run --rm -v "$PWD:/app" --entrypoint python3 llm-eval-suite:latest \\
        scripts/verify_polyglot_packs.py [test_key ...]

Exits non-zero if any pack fails its self-check or a toolchain is missing.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polyglot  # noqa: E402


def _passes(test_key, res):
    if test_key == "unit_test_synthesis":
        return res["ok"] and len(res["mutants_killed"]) == res["mutants_total"]
    return res["ok"]


def main(argv):
    tests = argv or sorted(os.listdir(polyglot.FIXTURE_ROOT))
    failures = 0
    for test_key in tests:
        for lang in sorted(os.listdir(os.path.join(polyglot.FIXTURE_ROOT, test_key))):
            if not polyglot.has_pack(test_key, lang):
                continue
            pack = polyglot.load_pack(test_key, lang)
            for kind, expect_pass in (("reference", True), ("negative", False)):
                answers = pack[kind]
                if not answers:
                    print(f"MISSING  {test_key}/{lang} {kind}")
                    failures += 1
                    continue
                code = next(iter(answers.values()))
                t0 = time.time()
                res = polyglot.grade_answer(test_key, lang, code)
                got = _passes(test_key, res)
                good = got == expect_pass and res["stage"] != "unavailable"
                failures += not good
                extra = ""
                if test_key == "unit_test_synthesis":
                    extra = f" mutants {len(res['mutants_killed'])}/{res['mutants_total']}"
                print(f"{'OK  ' if good else 'BAD '}  {test_key}/{lang:<10} {kind:<9} "
                      f"graded={'PASS' if got else 'FAIL'} ({res['stage']}: {res['detail'][:90]}){extra} "
                      f"[{time.time() - t0:.1f}s]", flush=True)
    print(f"\n{'ALL PACKS DISCRIMINATE CORRECTLY' if not failures else f'{failures} SELF-CHECK FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
