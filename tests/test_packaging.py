#!/usr/bin/env python3
"""Tests for source and installed resource paths."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import client  # noqa: E402
import languages  # noqa: E402
import polyglot  # noqa: E402
import report_md  # noqa: E402
import suites  # noqa: E402


class TestResourcePaths(unittest.TestCase):
    def test_source_modules_share_the_packaged_data_root(self):
        self.assertEqual(client.DATA_DIR, os.path.join(ROOT, "data"))
        self.assertEqual(suites.SUITE_DIR, os.path.join(client.DATA_DIR, "suites"))
        self.assertEqual(polyglot.FIXTURE_ROOT, os.path.join(client.DATA_DIR, "polyglot"))
        self.assertEqual(languages._POLYGLOT_DIR, polyglot.FIXTURE_ROOT)
        self.assertEqual(report_md.DEFAULT_REFERENCE, os.path.join(client.DATA_DIR, "reference", "published_scores.json"))

    def test_installed_import_needs_no_write_access_to_site_packages(self):
        with tempfile.TemporaryDirectory() as td:
            site = os.path.join(td, "site")
            work = os.path.join(td, "work")
            os.makedirs(os.path.join(site, "llm_eval_data"))
            os.makedirs(os.path.join(site, "llm_eval_assets"))
            os.makedirs(work)
            for name in ("client.py", "env_loader.py"):
                shutil.copyfile(os.path.join(ROOT, name), os.path.join(site, name))
            image = os.path.join(site, "llm_eval_assets", "invoice.png")
            with open(image, "wb") as f:
                f.write(b"image")
            script = (
                "import json, os, sys\n"
                "sys.dont_write_bytecode = True\n"
                f"sys.path.insert(0, {site!r})\n"
                "def forbid(*args, **kwargs):\n"
                "    raise PermissionError('import must not create output directories')\n"
                "os.makedirs = forbid\n"
                "import client, env_loader\n"
                "print(json.dumps([client.DATA_DIR, client.IMAGE_PATH, client.DEFAULT_RESULTS_DIR, "
                "client.DEFAULT_LOGS_DIR, env_loader.DEFAULT_CONFIG_PATH]))\n"
            )
            result = subprocess.run([sys.executable, "-I", "-c", script], cwd=work,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [
                os.path.join(site, "llm_eval_data"), image, os.path.join(work, "results"),
                os.path.join(work, "logs"), os.path.join(work, "config.json"),
            ])


if __name__ == "__main__":
    unittest.main()
