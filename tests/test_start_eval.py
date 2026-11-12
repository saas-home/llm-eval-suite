#!/usr/bin/env python3
"""Regression tests for the interactive Docker launcher, without Docker calls."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_NAMES = (
    "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY", "MAX_CONTEXT",
    "JUDGE_ENDPOINT", "JUDGE_MODEL", "JUDGE_API_KEY",
)


@unittest.skipUnless(shutil.which("bash"), "needs Bash")
class TestStartEval(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo with spaces"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        self.launcher = scripts / "start_eval.sh"
        shutil.copyfile(ROOT / "scripts" / "start_eval.sh", self.launcher)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.log = self.base / "docker.jsonl"
        docker = self.bin / "docker"
        docker.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            f"names = {ENV_NAMES!r}\n"
            "with open(os.environ['FAKE_DOCKER_LOG'], 'a') as f:\n"
            "    f.write(json.dumps({'args': args, 'env': "
            "{name: os.environ.get(name) for name in names}}) + '\\n')\n"
            "command = 'probe' if '--probe' in args else args[0]\n"
            "if args[:2] == ['image', 'inspect'] and os.getenv('FAKE_IMAGE_MISSING'):\n"
            "    sys.exit(1)\n"
            "if os.getenv('FAKE_FAIL') == command:\n"
            "    print('Docker failed: ' + command, file=sys.stderr)\n"
            "    sys.exit(3)\n",
            encoding="utf-8",
        )
        docker.chmod(0o755)

    def launcher_env(self, env=None):
        child_env = dict(os.environ)
        for name in ENV_NAMES + ("FAKE_IMAGE_MISSING", "FAKE_FAIL"):
            child_env.pop(name, None)
        child_env.update({
            "PATH": str(self.bin) + os.pathsep + child_env.get("PATH", ""),
            "FAKE_DOCKER_LOG": str(self.log),
        })
        child_env.update(env or {})
        return child_env

    def run_launcher(self, answers=(), env=None, args=(), trace=False):
        command = [shutil.which("bash")]
        if trace:
            command.append("-x")
        command.extend([str(self.launcher), *args])
        result = subprocess.run(
            command, input="\n".join(answers) + ("\n" if answers else ""),
            cwd=str(self.base), env=self.launcher_env(env),
            capture_output=True, text=True, timeout=20,
        )
        calls = []
        if self.log.exists():
            calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, calls

    def run_calls(self, calls):
        return [call for call in calls if call["args"][0] == "run"]

    def test_defaults_probe_then_run_all_languages_with_persistent_outputs(self):
        result, calls = self.run_launcher([""] * 11 + ["y"])
        self.assertEqual(result.returncode, 0, result.stderr)
        probe, run = self.run_calls(calls)
        self.assertEqual(probe["args"][-2:], ["llm-eval-suite:latest", "--probe"])
        self.assertEqual(run["args"][-5:], [
            "llm-eval-suite:latest", "--suite", "all", "--language", "polyglot",
        ])
        self.assertEqual(run["env"]["OPENAI_BASE_URL"], "http://127.0.0.1:8080/v1")
        self.assertEqual(run["env"]["OPENAI_MODEL"], "")
        self.assertIn(str(self.repo / "results") + ":/app/results", run["args"])
        self.assertIn(str(self.repo / "logs") + ":/app/logs", run["args"])
        self.assertTrue((self.repo / "results").is_dir())
        self.assertTrue((self.repo / "logs").is_dir())
        self.assertNotIn(str(self.repo) + ":/app", run["args"])

    def test_custom_details_and_secrets_use_environment_not_arguments_or_output(self):
        model = "model $(touch should-not-exist)"
        key = "test-target-key with spaces $HOME"
        judge_key = "test-judge-key"
        answers = [
            "https://model.example/v1", model, key, "262144", "8",
            "all", "http://judge.example/v1", "independent judge", judge_key, "2",
            "rust", "2x GPU", "120", "400", "y",
        ]
        result, calls = self.run_launcher(answers, trace=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        run = self.run_calls(calls)[-1]
        self.assertEqual(run["env"], {
            "OPENAI_BASE_URL": "https://model.example/v1",
            "OPENAI_MODEL": model,
            "OPENAI_API_KEY": key,
            "MAX_CONTEXT": "262144",
            "JUDGE_ENDPOINT": "http://judge.example/v1",
            "JUDGE_MODEL": "independent judge",
            "JUDGE_API_KEY": judge_key,
        })
        for name in ENV_NAMES:
            index = run["args"].index(name)
            self.assertEqual(run["args"][index - 1], "--env")
        self.assertEqual(run["args"][-15:], [
            "llm-eval-suite:latest", "--suite", "all", "--language", "rust",
            "--hardware", "2x GPU",
            "--parallel", "8", "--judge-parallel", "2",
            "--workload-ttft", "120", "--workload-complete", "400",
        ])
        for secret in (key, judge_key):
            self.assertNotIn(secret, result.stdout + result.stderr)
            self.assertTrue(all(secret not in arg for call in calls for arg in call["args"]))
        self.assertFalse((self.base / "should-not-exist").exists())
        self.assertFalse((self.repo / "config.json").exists())
        self.assertFalse((self.repo / ".env").exists())

    @unittest.skipUnless(os.name == "posix", "needs a POSIX terminal")
    def test_api_key_entry_disables_terminal_echo(self):
        import errno
        import pty
        import select
        import termios
        import time

        master, slave = pty.openpty()
        process = subprocess.Popen(
            [shutil.which("bash"), str(self.launcher)], cwd=str(self.base),
            env=self.launcher_env(), stdin=slave, stdout=slave, stderr=slave,
        )
        os.close(slave)
        transcript = b""
        secret = "test-secret-not-echoed"
        try:
            answers = [
                ("Model endpoint", ""),
                ("Model ID", ""),
                ("Model API key", secret),
                ("Context window", ""),
                ("Model concurrency", ""),
                ("Test mode", ""),
                ("Independent judge endpoint", ""),
                ("Coding language", ""),
                ("GPU hardware", ""),
                ("Workload Test 46 max TTFT", ""),
                ("Workload Test 46 max completion", ""),
                ("Start evaluation?", "n"),
            ]
            for label, answer in answers:
                deadline = time.monotonic() + 5
                while label.encode() not in transcript:
                    remaining = deadline - time.monotonic()
                    self.assertGreater(remaining, 0, transcript.decode(errors="replace"))
                    ready, _, _ = select.select([master], [], [], remaining)
                    self.assertTrue(ready, transcript.decode(errors="replace"))
                    transcript += os.read(master, 65536)
                if answer == secret:
                    while termios.tcgetattr(master)[3] & termios.ECHO:
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.001)
                os.write(master, (answer + "\n").encode())
            self.assertEqual(process.wait(timeout=5), 0)
            while select.select([master], [], [], 0)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
                if not chunk:
                    break
                transcript += chunk
            self.assertNotIn(secret.encode(), transcript)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)

    def test_blank_answers_clear_inherited_model_and_judge_credentials(self):
        result, calls = self.run_launcher(
            [""] * 11 + ["y"], env={name: "old-setting" for name in ENV_NAMES},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        env = self.run_calls(calls)[-1]["env"]
        self.assertEqual(env["OPENAI_BASE_URL"], "http://127.0.0.1:8080/v1")
        self.assertTrue(all(env[name] == "" for name in ENV_NAMES[1:]))

    def test_invalid_endpoint_context_and_choices_are_reprompted(self):
        answers = [
            "server:8000", "http://server/v1", "", "", "-1", "abc", "8192",
            "-2", "12",
            "unknown", "core", "ruby", "python", "", "-5", "300", "-10", "600", "maybe", "yes",
        ]
        result, calls = self.run_launcher(answers)
        self.assertEqual(result.returncode, 0, result.stderr)
        run = self.run_calls(calls)[-1]
        self.assertEqual(run["env"]["MAX_CONTEXT"], "8192")
        self.assertEqual(run["args"][-10:], [
            "--suite", "core", "--language", "python",
            "--parallel", "12", "--workload-ttft", "300", "--workload-complete", "600",
        ])
        self.assertIn("Enter an http:// or https:// endpoint", result.stderr)
        self.assertIn("positive whole number", result.stderr)
        self.assertIn("Choose one of:", result.stderr)
        self.assertIn("Enter y or n.", result.stderr)

    def test_cancel_does_not_send_requests_or_create_outputs(self):
        result, calls = self.run_launcher([""] * 12)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Cancelled", result.stdout)
        self.assertFalse(self.run_calls(calls))
        self.assertFalse((self.repo / "results").exists())

    def test_eof_is_an_explicit_failure_without_starting_evaluation(self):
        result, calls = self.run_launcher(["http://server/v1"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input ended", result.stderr)
        self.assertFalse(self.run_calls(calls))

    def test_docker_daemon_failure_stops_before_prompting(self):
        result, calls = self.run_launcher(env={"FAKE_FAIL": "info"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("Cannot reach Docker", result.stderr)
        self.assertEqual([call["args"] for call in calls], [["info"]])

    def test_missing_image_is_built_only_after_approval(self):
        result, calls = self.run_launcher(
            ["y"] + [""] * 11 + ["y"], env={"FAKE_IMAGE_MISSING": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        builds = [call["args"] for call in calls if call["args"][0] == "build"]
        self.assertEqual(builds, [["build", "--tag", "llm-eval-suite:latest", str(self.repo)]])
        self.assertEqual(len(self.run_calls(calls)), 2)

    def test_declining_missing_image_build_does_not_run(self):
        result, calls = self.run_launcher(["n"], env={"FAKE_IMAGE_MISSING": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(call["args"][0] in ("build", "run") for call in calls))

    def test_failed_probe_prevents_full_run(self):
        result, calls = self.run_launcher([""] * 11 + ["y"], env={"FAKE_FAIL": "probe"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("Endpoint probe failed", result.stderr)
        self.assertEqual(len(self.run_calls(calls)), 1)
        self.assertFalse((self.repo / "results").exists())

    def test_evaluation_failure_exit_code_is_preserved(self):
        result, calls = self.run_launcher([""] * 11 + ["y"], env={"FAKE_FAIL": "run"})
        self.assertEqual(result.returncode, 3)
        self.assertEqual(len(self.run_calls(calls)), 2)

    def test_help_and_invalid_arguments_do_not_contact_docker(self):
        result, calls = self.run_launcher(args=("--help",))
        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)
        self.assertFalse(calls)
        result, calls = self.run_launcher(args=("--unknown",))
        self.assertEqual(result.returncode, 1)
        self.assertIn("No arguments are accepted", result.stderr)
        self.assertFalse(calls)

    def test_subsequent_run_remembers_previous_values_as_defaults(self):
        answers = [
            "http://172.16.16.16:8080/v1", "custom-model", "custom-key", "131072", "12",
            "all", "http://172.16.16.43:8888/v1", "judge-model", "judge-key", "4",
            "rust", "1x GPU", "300", "600", "y",
        ]
        result1, calls1 = self.run_launcher(answers)
        self.assertEqual(result1.returncode, 0, result1.stderr)
        run1 = self.run_calls(calls1)[-1]
        self.assertEqual(run1["env"]["OPENAI_BASE_URL"], "http://172.16.16.16:8080/v1")
        self.assertEqual(run1["env"]["OPENAI_MODEL"], "custom-model")
        self.assertEqual(run1["env"]["OPENAI_API_KEY"], "custom-key")
        self.assertEqual(run1["env"]["MAX_CONTEXT"], "131072")
        self.assertEqual(run1["env"]["JUDGE_ENDPOINT"], "http://172.16.16.43:8888/v1")
        self.assertEqual(run1["env"]["JUDGE_MODEL"], "judge-model")
        self.assertEqual(run1["env"]["JUDGE_API_KEY"], "judge-key")
        self.assertEqual(run1["args"][-15:], [
            "llm-eval-suite:latest", "--suite", "all", "--language", "rust",
            "--hardware", "1x GPU",
            "--parallel", "12", "--judge-parallel", "4",
            "--workload-ttft", "300", "--workload-complete", "600",
        ])

        cache_file = self.repo / ".start_eval.cache"
        self.assertTrue(cache_file.exists())
        if os.name == "posix":
            self.assertEqual(cache_file.stat().st_mode & 0o777, 0o600)
        cache_text = cache_file.read_text()
        self.assertNotIn("custom-key", cache_text)
        self.assertNotIn("judge-key", cache_text)
        self.assertNotIn("CACHED_API_KEY", cache_text)
        self.assertNotIn("CACHED_JUDGE_API_KEY", cache_text)
        self.assertIn("CACHED_PARALLEL=12", cache_text)
        self.assertIn("CACHED_JUDGE_PARALLEL=4", cache_text)
        self.assertIn("CACHED_WORKLOAD_TTFT=300", cache_text)
        self.assertIn("CACHED_WORKLOAD_COMPLETE=600", cache_text)

        # Second run: press Enter for all prompts (API keys are never saved, so blank means no authentication)
        result2, calls2 = self.run_launcher([""] * 14 + ["y"])
        self.assertEqual(result2.returncode, 0, result2.stderr)
        run2 = self.run_calls(calls2)[-1]
        self.assertEqual(run2["env"]["OPENAI_BASE_URL"], "http://172.16.16.16:8080/v1")
        self.assertEqual(run2["env"]["OPENAI_MODEL"], "custom-model")
        self.assertEqual(run2["env"]["OPENAI_API_KEY"], "")
        self.assertEqual(run2["env"]["MAX_CONTEXT"], "131072")
        self.assertEqual(run2["env"]["JUDGE_ENDPOINT"], "http://172.16.16.43:8888/v1")
        self.assertEqual(run2["env"]["JUDGE_MODEL"], "judge-model")
        self.assertEqual(run2["env"]["JUDGE_API_KEY"], "")
        self.assertEqual(run2["args"][-15:], [
            "llm-eval-suite:latest", "--suite", "all", "--language", "rust",
            "--hardware", "1x GPU",
            "--parallel", "12", "--judge-parallel", "4",
            "--workload-ttft", "300", "--workload-complete", "600",
        ])

    def test_reset_clears_cached_defaults(self):
        answers = [
            "http://172.16.16.16:8080/v1", "custom-model", "", "", "",
            "quick", "python", "", "", "", "y",
        ]
        self.run_launcher(answers)
        cache_file = self.repo / ".start_eval.cache"
        self.assertTrue(cache_file.exists())

        res_reset, _ = self.run_launcher(args=("--reset",))
        self.assertEqual(res_reset.returncode, 0)
        self.assertIn("Cleared remembered evaluation defaults", res_reset.stdout)
        self.assertFalse(cache_file.exists())

        # Next run uses clean defaults again
        res3, calls3 = self.run_launcher([""] * 11 + ["y"])
        self.assertEqual(res3.returncode, 0)
        run3 = self.run_calls(calls3)[-1]
        self.assertEqual(run3["env"]["OPENAI_BASE_URL"], "http://127.0.0.1:8080/v1")
        self.assertEqual(run3["env"]["OPENAI_MODEL"], "")
        self.assertEqual(run3["args"][-4:], ["--suite", "all", "--language", "polyglot"])

    def test_quick_and_core_mode_skips_judge_prompts(self):
        answers = [
            "http://172.16.16.25:8080/v1", "quick-model", "", "", "8",
            "quick", "python", "2x GPU", "300", "600", "y",
        ]
        result, calls = self.run_launcher(answers)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Independent judge endpoint", result.stderr)
        run = self.run_calls(calls)[-1]
        self.assertEqual(run["env"]["JUDGE_ENDPOINT"], "")
        self.assertEqual(run["args"][-14:], [
            "llm-eval-suite:latest", "--suite", "core", "--language", "python",
            "--quick", "--hardware", "2x GPU",
            "--parallel", "8",
            "--workload-ttft", "300", "--workload-complete", "600",
        ])


if __name__ == "__main__":
    unittest.main()
