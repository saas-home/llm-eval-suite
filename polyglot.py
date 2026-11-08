#!/usr/bin/env python3
"""
polyglot.py -- compile-and-run graders for the non-Python SWE tests (20-25).

Each (test, language) pair has a test pack under data/polyglot/<test>/<lang>/:

    manifest.json   how to grade (mode, answer file name, strictness, mutants)
    support/        files written next to the model's answer (types, sources)
    harness/        verification program / test files written by us
    reference/      a known-good answer   (self-check: must PASS)
    negative/       a known-bad answer    (self-check: must FAIL)

Modes:
    program     support + harness + answer are built and run; the harness
                exits 0 and prints "POLYGLOT_OK <run id>" only if every
                behavior holds. The run id is random per run and reaches the
                harness in the EVAL_RUN_ID variable, so an answer that
                prints the marker itself does not pass.
    unit_tests  the model's answer IS a test suite: it must pass against the
                real source and fail against each seeded mutant (mutation
                testing), proving the tests actually assert behavior.

Toolchains come from the Docker image (scripts/setup_polyglot_toolchains.sh);
everything here is Python standard library only.
"""

import functools
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from client import DATA_DIR, RUN_ID_VAR, model_code_env, run_model_code

POLYGLOT_HOME = os.environ.get("POLYGLOT_HOME", "/opt/polyglot")
FIXTURE_ROOT = os.path.join(DATA_DIR, "polyglot")
OK_MARKER = "POLYGLOT_OK"
COMPILE_TIMEOUT = 240
RUN_TIMEOUT = 90

_TOOLS = {
    "typescript": ["node"],
    "go": ["go"],
    "java": ["javac", "java"],
    "rust": ["rustc"],
    "csharp": ["dotnet"],
}


# Rust edition for every compile. 2024 is the edition `cargo new` uses.
RUST_EDITION = "2024"


@functools.lru_cache(maxsize=1)
def _go_language_version() -> str:
    """The installed Go's language version (for example "1.27"), so go.mod
    allows every feature of the toolchain in the image."""
    try:
        out = subprocess.run(["go", "env", "GOVERSION"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    m = re.match(r"go(\d+\.\d+)", out.strip())
    return m.group(1) if m else "1.22"


def _go_mod() -> str:
    return f"module evalpkg\n\ngo {_go_language_version()}\n"


def toolchain_available(lang: str) -> bool:
    """True when the compilers and offline test frameworks for `lang` exist."""
    if lang not in _TOOLS or not all(shutil.which(t) for t in _TOOLS[lang]):
        return False
    needs = {
        "typescript": os.path.join(POLYGLOT_HOME, "ts", "node_modules", ".bin", "vitest"),
        "java": os.path.join(POLYGLOT_HOME, "java", "lib"),
        "csharp": os.path.join(POLYGLOT_HOME, "csharp", "tests", "Tests.csproj"),
    }.get(lang)
    return needs is None or os.path.exists(needs)


# ----------------------------------------------------------------------------
# Process helpers
# ----------------------------------------------------------------------------
def _exec(cmd: List[str], cwd: str, timeout: int, env: Optional[Dict[str, str]] = None):
    """Runs a command with client.run_model_code (own process group, killed
    with its children at the end; output capped). Returns (returncode,
    combined output, timed_out). A timeout or a missing executable gives
    returncode -1 with an explanatory message."""
    try:
        proc = run_model_code(cmd, timeout, cwd=cwd, env=env if env is not None else model_code_env())
        return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), False
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout}s (non-terminating code or deadlock)", True
    except OSError as e:
        return -1, f"could not execute {cmd[0]}: {e}", False


def _run(cmd: List[str], cwd: str, timeout: int, env: Optional[Dict[str, str]] = None):
    """_exec without the timed_out flag: (returncode, combined output)."""
    rc, out, _ = _exec(cmd, cwd, timeout, env)
    return rc, out


# Build steps that timed out in this thread: a compiler that needs more than
# COMPILE_TIMEOUT on a correct or wrong answer alike points at a loaded
# machine, not at the model. eval.py's infra guard reads and clears them and
# marks the test as an infra failure, like a failed request.
_INFRA = threading.local()


def take_infra_events() -> List[str]:
    """Returns and clears this thread's build timeouts."""
    events = getattr(_INFRA, "events", [])
    _INFRA.events = []
    return events


def _compile(cmd: List[str], cwd: str, env: Dict[str, str], counts: bool = False) -> Optional[Dict[str, Any]]:
    """Runs one build step (compiler, restore). None when it succeeds; else
    the failure result: stage "compile" for a build error (the model's), or
    stage "infra" when the step timed out. `counts` adds 0 tests ran/passed.
    Only pure build steps go through here: `go test` and Vitest also run the
    code, so their timeouts stay the model's."""
    rc, out, timed_out = _exec(cmd, cwd, COMPILE_TIMEOUT, env)
    if rc == 0:
        return None
    zero = (0, 0) if counts else (None, None)
    if timed_out:
        reason = f"{os.path.basename(cmd[0])} build step timed out after {COMPILE_TIMEOUT}s (machine overloaded?)"
        _INFRA.events = getattr(_INFRA, "events", []) + [reason]
        return _result(False, "infra", reason, *zero)
    return _fail("compile", out, *zero)


def _first_error(output: str, limit: int = 300) -> str:
    """The most informative line(s) of a failed build/run, for the report."""
    lines = [l.strip() for l in output.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        # Rust prints the panic location, then the message on the next line.
        if "panicked at" in line and i + 1 < len(lines):
            return f"{line} {lines[i + 1]}"[:limit]
    for pat in (r"_test\.go:\d+:", r"^\w*(?:Error|Exception)\b.*?:\s", r"error", r"Error", r"panic", r"FAIL",
                r"Exception", r"assert"):
        hits = [l for l in lines if re.search(pat, l)]
        if hits:
            return hits[0][:limit]
    return (lines[-1] if lines else "no output")[:limit]


FEEDBACK_LINES = 40
FEEDBACK_CHARS = 2000


def _error_excerpt(output: str) -> str:
    """The build or run output from its first informative line on (at most
    FEEDBACK_LINES lines / FEEDBACK_CHARS characters): what a developer would
    read, sent back to the model when it gets a retry."""
    lines = [l.rstrip() for l in output.splitlines() if l.strip()]
    first = _first_error(output, limit=10**6)
    start = next((i for i, l in enumerate(lines) if first.startswith(l.strip()[:80])), 0)
    return "\n".join(lines[start:start + FEEDBACK_LINES])[:FEEDBACK_CHARS]


def _fail(stage: str, output: str, ran: Optional[int] = None, passed: Optional[int] = None) -> Dict[str, Any]:
    res = _result(False, stage, _first_error(output), ran, passed)
    res["output"] = _error_excerpt(output)
    return res


def _write_files(root: str, files: Dict[str, str]) -> None:
    for name, body in files.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)


def _env(tmp: str) -> Dict[str, str]:
    env = model_code_env()  # no API keys for the model's code
    env.update({
        "HOME": tmp,
        "GOPATH": os.path.join(tmp, "gopath"),
        "GOCACHE": os.environ.get("POLYGLOT_GOCACHE", "/tmp/polyglot-gocache"),
        "GOFLAGS": "-mod=mod",
        "GOPROXY": "off",
        "GOTOOLCHAIN": "local",
        "DOTNET_CLI_HOME": tmp,
        "DOTNET_NOLOGO": "1",
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "NUGET_PACKAGES": os.path.join(POLYGLOT_HOME, "nuget"),
        "NO_COLOR": "1",
        RUN_ID_VAR: secrets.token_hex(16),  # one id for every step of this run (see _passed_run)
    })
    return env


def _passed_run(rc: int, out: str, env: Dict[str, str]) -> bool:
    """The harness exited 0 and printed the marker with this run's id."""
    return rc == 0 and f"{OK_MARKER} {env[RUN_ID_VAR]}" in out.splitlines()


def _link_node_modules(tmp: str) -> None:
    """tmp/node_modules as a real folder of links to the shared packages.
    Vitest and Vite write their caches into <root>/node_modules; with a
    folder of our own those writes stay in this run's temp folder, so
    TypeScript runs need no lock."""
    shared = os.path.join(POLYGLOT_HOME, "ts", "node_modules")
    own = os.path.join(tmp, "node_modules")
    os.mkdir(own)
    for name in os.listdir(shared):
        if not name.startswith((".vite", ".vitest")):
            os.symlink(os.path.join(shared, name), os.path.join(own, name))


def _result(ok: bool, stage: str, detail: str, ran: Optional[int] = None, passed: Optional[int] = None) -> Dict[str, Any]:
    return {"ok": ok, "stage": stage, "detail": detail, "tests_ran": ran, "tests_passed": passed}


# ----------------------------------------------------------------------------
# Program mode: build, then run the harness entry point
# ----------------------------------------------------------------------------
_TSCONFIG = {
    "compilerOptions": {
        "strict": True, "target": "ES2022", "module": "commonjs", "outDir": "out", "rootDir": ".",
        "types": ["node"], "esModuleInterop": True, "skipLibCheck": True, "noEmitOnError": True,
    },
    "include": ["*.ts"],
}


# Toolchains whose builds share on-disk state and so must not build at the
# same time: `dotnet restore` writes the shared NuGet folder. The lock covers
# the build steps only; the built program runs outside it. Go, Rust and Java
# build in their own temp directories (Go's build cache is safe for
# concurrent use), and TypeScript runs get their own node_modules folder
# (_link_node_modules).
_SHARED_STATE_LOCKS = {"csharp": threading.Lock()}


def _build_lock(lang: str):
    return _SHARED_STATE_LOCKS.get(lang) or nullcontext()


def _dotnet_build(tmp: str, env: Dict[str, str], build_args: List[str], counts: bool = False) -> Optional[Dict[str, Any]]:
    """`dotnet restore` + `dotnet build` under the C# build lock. None when
    both succeed, else the failure result (see _compile). No build servers:
    the compiler server detaches from the process group, so it would outlive
    the grade (one idle server per build), and a later build cannot reuse it."""
    with _build_lock("csharp"):
        return (_compile(["dotnet", "restore", "--source", os.path.join(POLYGLOT_HOME, "nuget"), "-v", "q"],
                         tmp, env, counts)
                or _compile(["dotnet", "build", "--no-restore", "--disable-build-servers", "-v", "q", "--nologo",
                             "-o", "bin_out"] + build_args, tmp, env, counts))


def run_program(lang: str, files: Dict[str, str], strict: bool = False, main_class: str = "Main") -> Dict[str, Any]:
    """Builds `files` and runs the harness. ok only if the build succeeds, the
    run exits 0 and prints POLYGLOT_OK. `strict` turns warnings into errors."""
    if not toolchain_available(lang):
        return _result(False, "unavailable", f"{lang} toolchain not installed (run inside the Docker image)")
    with tempfile.TemporaryDirectory(prefix=f"polyglot-{lang}-") as tmp:
        env = _env(tmp)
        if lang == "typescript":
            _write_files(tmp, files)
            _write_files(tmp, {"tsconfig.json": json.dumps(_TSCONFIG)})
            _link_node_modules(tmp)
            failed = _compile([os.path.join(tmp, "node_modules", ".bin", "tsc"), "-p", "tsconfig.json"], tmp, env)
            if failed:
                return failed
            rc, out = _run(["node", os.path.join("out", "main.js")], tmp, RUN_TIMEOUT, env)
        elif lang == "go":
            _write_files(tmp, files)
            _write_files(tmp, {"go.mod": _go_mod()})
            if strict:
                failed = _compile(["go", "vet", "./..."], tmp, env)
                if failed:
                    return failed
            rc, out = _run(["go", "test", "-count=1", f"-timeout={RUN_TIMEOUT}s", "./..."], tmp, COMPILE_TIMEOUT, env)
            if rc != 0:
                stage = "compile" if ("[build failed]" in out or "[setup failed]" in out) else "run"
                return _fail(stage, out)
            return _result(True, "ok", "verified by execution")
        elif lang == "java":
            _write_files(tmp, files)
            sources = [n for n in files if n.endswith(".java")]
            cmd = ["javac", "-d", "out"] + (["-Xlint:all", "-Werror"] if strict else []) + sources
            failed = _compile(cmd, tmp, env)
            if failed:
                return failed
            rc, out = _run(["java", "-ea", "-cp", "out", main_class], tmp, RUN_TIMEOUT, env)
        elif lang == "rust":
            _write_files(tmp, files)
            cmd = ["rustc", "--edition", RUST_EDITION, "-o", "prog"] + (["-D", "warnings"] if strict else []) + ["main.rs"]
            failed = _compile(cmd, tmp, env)
            if failed:
                return failed
            rc, out = _run([os.path.join(tmp, "prog")], tmp, RUN_TIMEOUT, env)
        elif lang == "csharp":
            shutil.copytree(os.path.join(POLYGLOT_HOME, "csharp", "app"), tmp, dirs_exist_ok=True)
            _write_files(tmp, files)
            failed = _dotnet_build(tmp, env, ["-warnaserror"] if strict else [])
            if failed:
                return failed
            rc, out = _run(["dotnet", os.path.join("bin_out", "App.dll")], tmp, RUN_TIMEOUT, env)
        else:
            return _result(False, "unavailable", f"no runner for {lang}")
        if _passed_run(rc, out, env):
            return _result(True, "ok", "verified by execution")
        return _fail("run", out)


# ----------------------------------------------------------------------------
# Unit-test mode: run a test suite with the language's framework
# ----------------------------------------------------------------------------
def run_tests(lang: str, files: Dict[str, str]) -> Dict[str, Any]:
    """Runs the test suite in `files`. Returns tests_ran / tests_passed parsed
    from the framework's own report; a build failure means 0 ran."""
    if not toolchain_available(lang):
        return _result(False, "unavailable", f"{lang} toolchain not installed (run inside the Docker image)", 0, 0)
    with tempfile.TemporaryDirectory(prefix=f"polyglot-tests-{lang}-") as tmp:
        env = _env(tmp)
        failures = ""  # the failed tests' messages, when the framework reports them apart from `out`
        if lang == "typescript":
            _write_files(tmp, files)
            _link_node_modules(tmp)
            rc, out = _run([os.path.join(tmp, "node_modules", ".bin", "vitest"), "run", "--reporter=json",
                            "--outputFile=report.json", "--root", tmp], tmp, COMPILE_TIMEOUT, env)
            try:
                with open(os.path.join(tmp, "report.json")) as f:
                    rep = json.load(f)
                ran, passed = rep.get("numTotalTests", 0), rep.get("numPassedTests", 0)
                failures = "\n".join(f"FAIL {a.get('fullName', '')}: {_own_frames(m)}"
                                     for t in rep.get("testResults", []) for a in t.get("assertionResults", [])
                                     if a.get("status") == "failed" for m in a.get("failureMessages", []))
            except (OSError, ValueError):
                return _fail("compile", out, 0, 0)
        elif lang == "go":
            _write_files(tmp, files)
            _write_files(tmp, {"go.mod": _go_mod()})
            rc, out = _run(["go", "test", "-count=1", "-json", f"-timeout={RUN_TIMEOUT}s", "./..."], tmp, COMPILE_TIMEOUT, env)
            ran = passed = 0
            go_output = []
            for line in out.splitlines():
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("Action") == "output" and ev.get("Test"):
                    go_output.append(ev.get("Output", ""))
                name = ev.get("Test")
                if not name or "/" in name or ev.get("Action") not in ("pass", "fail"):
                    continue
                ran += 1
                passed += ev["Action"] == "pass"
            if ran == 0:
                return _fail("compile", out, 0, 0)
            failures = "".join(go_output)
        elif lang == "java":
            _write_files(tmp, files)
            lib = os.path.join(POLYGLOT_HOME, "java", "lib")
            jars = sorted(os.path.join(lib, j) for j in os.listdir(lib) if j.endswith(".jar"))
            cp = os.pathsep.join(j for j in jars if "console-standalone" not in j)
            launcher = next(j for j in jars if "console-standalone" in j)
            sources = [n for n in files if n.endswith(".java")]
            failed = _compile(["javac", "-cp", launcher + os.pathsep + cp, "-d", "out"] + sources, tmp, env, counts=True)
            if failed:
                return failed
            rc, out = _run(["java", "-XX:+EnableDynamicAgentLoading", "-jar", launcher, "execute",
                            "--class-path", "out" + os.pathsep + cp, "--scan-class-path", "--disable-banner",
                            "--disable-ansi-colors", "--details=summary"], tmp, RUN_TIMEOUT, env)
            found = re.search(r"(\d+)\s+tests found", out)
            ok_n = re.search(r"(\d+)\s+tests successful", out)
            ran = int(found.group(1)) if found else 0
            passed = int(ok_n.group(1)) if ok_n else 0
        elif lang == "rust":
            _write_files(tmp, files)
            failed = _compile(["rustc", "--edition", RUST_EDITION, "--test", "-o", "tests", "lib.rs"], tmp, env, counts=True)
            if failed:
                return failed
            rc, out = _run([os.path.join(tmp, "tests")], tmp, RUN_TIMEOUT, env)
            m = re.search(r"test result: \w+\. (\d+) passed; (\d+) failed", out)
            passed = int(m.group(1)) if m else 0
            ran = passed + (int(m.group(2)) if m else 0)
        elif lang == "csharp":
            shutil.copytree(os.path.join(POLYGLOT_HOME, "csharp", "tests"), tmp, dirs_exist_ok=True)
            _write_files(tmp, files)
            # xUnit v3 test projects are executables. Build, run the in-process
            # runner, and read its XML report (counts per assembly).
            failed = _dotnet_build(tmp, env, [], counts=True)
            if failed:
                return failed
            report = os.path.join(tmp, "xunit-report.xml")
            rc, out = _run(["dotnet", os.path.join("bin_out", "Tests.dll"), "-xml", report], tmp, RUN_TIMEOUT, env)
            counts = _xunit_counts(report)
            if counts is None:
                return _fail("run", out, 0, 0)
            ran, passed = counts
            failures = _xunit_failures(report)
        else:
            return _result(False, "unavailable", f"no runner for {lang}", 0, 0)
        ok = ran > 0 and passed == ran
        res = _result(ok, "ok" if ok else "run", f"{passed}/{ran} tests passed" if ran else _first_error(out), ran, passed)
        if not ok:
            res["output"] = (_error_excerpt(failures) if lang == "go" else failures[:FEEDBACK_CHARS]) \
                if failures.strip() else _error_excerpt(out)
        return res


def _own_frames(message: str) -> str:
    """A failure message without the test framework's own stack frames."""
    return "\n".join(l for l in message.splitlines() if "node_modules" not in l and "<anonymous>" not in l)


def _xunit_failures(report_path: str) -> str:
    """Name, message and first stack lines of each failed test in an xUnit
    v3 XML report; '' when there is none or the report cannot be read."""
    try:
        root = ET.parse(report_path).getroot()
    except (OSError, ET.ParseError):
        return ""
    lines = []
    for test in root.iter("test"):
        if test.get("result") != "Fail":
            continue
        message = (test.findtext("failure/message") or "").strip()
        stack = (test.findtext("failure/stack-trace") or "").strip().splitlines()[:3]
        lines.append(f"FAIL {test.get('name', '')}: {message}" + "".join("\n    " + l.strip() for l in stack))
    return "\n".join(lines)


def _xunit_counts(report_path: str):
    """(ran, passed) from an xUnit v3 XML report, or None if it is missing or
    unreadable. Errors (for example, a fixture that throws) count as ran and
    not passed, so they fail the run."""
    try:
        root = ET.parse(report_path).getroot()
    except (OSError, ET.ParseError):
        return None
    assemblies = root.findall("assembly")
    if not assemblies:
        return None
    ran = passed = 0
    for a in assemblies:
        ran += int(a.get("total", 0)) + int(a.get("errors", 0))
        passed += int(a.get("passed", 0))
    return ran, passed


# ----------------------------------------------------------------------------
# Test packs
# ----------------------------------------------------------------------------
def _read_dir(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isdir(path):
        return out
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            with open(full, encoding="utf-8") as f:
                out[name] = f.read()
    return out


def has_pack(test_key: str, lang: str) -> bool:
    return os.path.isfile(os.path.join(FIXTURE_ROOT, test_key, lang, "manifest.json"))


def load_pack(test_key: str, lang: str) -> Dict[str, Any]:
    base = os.path.join(FIXTURE_ROOT, test_key, lang)
    with open(os.path.join(base, "manifest.json"), encoding="utf-8") as f:
        pack = json.load(f)
    for part in ("support", "harness", "reference", "negative"):
        pack[part] = _read_dir(os.path.join(base, part))
    return pack


def normalize_answer(pack: Dict[str, Any], code: str) -> str:
    """Benign, documented normalizations so a correct answer is not rejected
    for packaging details: Go package clause, Java package line."""
    code = code.strip() + "\n"
    if pack.get("go_package"):
        code = re.sub(r"(?m)^package\s+\w+", f"package {pack['go_package']}", code, count=1)
    if pack.get("strip_java_package"):
        code = re.sub(r"(?m)^package\s+[\w.]+;\s*$", "", code)
    return code


def _assemble(pack: Dict[str, Any], code: str) -> Dict[str, str]:
    """support files (minus those whose types the answer already defines) +
    harness + the answer."""
    files: Dict[str, str] = {}
    skip = pack.get("skip_support_if_defined", {})
    for name, body in pack["support"].items():
        if name in skip and re.search(skip[name], code):
            continue
        files[name] = body
    files.update(pack["harness"])
    if pack.get("prepend_support"):
        # The answer is meant to be appended to a source file (Rust test
        # modules); if the model returned the whole file, use it as-is.
        source = pack["support"][pack["prepend_support"]]
        already_included = pack.get("prepend_support_unless") and re.search(pack["prepend_support_unless"], code)
        files[pack["answer_file"]] = code if already_included else source + "\n" + code
        files.pop(pack["prepend_support"], None)
    else:
        files[pack["answer_file"]] = code
    # C# answers may declare a namespace; let the harness see its types,
    # either via a using directive or by qualifying `__NS__.` placeholders
    # (needed when legacy and refactored types share names).
    ns = re.search(r"(?m)^\s*namespace\s+([\w.]+)", code)
    if pack.get("csharp_namespace_using") and ns and pack["csharp_namespace_using"] in files:
        target = pack["csharp_namespace_using"]
        files[target] = f"using {ns.group(1)};\n" + files[target]
    if pack.get("csharp_namespace_placeholder") in files:
        target = pack["csharp_namespace_placeholder"]
        files[target] = files[target].replace("__NS__.", f"{ns.group(1)}." if ns else "global::")
    return files


def forbidden_hits(pack: Dict[str, Any], code: str) -> List[str]:
    """Patterns the answer must not contain (e.g. `any` in strict typing).
    Comments are ignored; square-bracketed type-parameter lists are removed
    first when the pack says so (Go's `[T any]` constraint is allowed)."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.DOTALL)
    if pack.get("forbidden_ignore_brackets"):
        text = re.sub(r"\[[^\[\]]*\]", "", text)
    return [p for p in pack.get("forbidden_patterns", []) if re.search(p, text)]


def grade_answer(test_key: str, lang: str, code: str) -> Dict[str, Any]:
    """Grades a model answer with the test pack. Returns ok / stage / detail
    plus mutation results for unit-test packs."""
    pack = load_pack(test_key, lang)
    code = normalize_answer(pack, code)
    if pack["mode"] == "program":
        hits = forbidden_hits(pack, code)
        res = run_program(lang, _assemble(pack, code), strict=pack.get("strict", False),
                          main_class=pack.get("main_class", "Main"))
        if hits and res["ok"]:
            res = _result(False, "strictness", f"forbidden construct(s): {hits}")
        res["forbidden"] = hits
        return res

    # unit_tests: must pass on the real source and kill every mutant
    res = run_tests(lang, _assemble(pack, code))
    mutants = pack.get("mutants", [])
    killed: List[str] = []
    if res["ok"]:
        for mutant in mutants:
            mutated = dict(pack)
            mutated["support"] = dict(pack["support"])
            fname, old, new = mutant["file"], mutant["replace"][0], mutant["replace"][1]
            mutated["support"][fname] = mutated["support"][fname].replace(old, new, 1)
            m_res = run_tests(lang, _assemble(mutated, code))
            if m_res["stage"] == "infra":  # not checked, so not killed: the grade is void
                return dict(m_res, mutants_total=len(mutants), mutants_killed=killed)
            if not m_res["ok"]:
                killed.append(mutant["name"])
    res["mutants_total"] = len(mutants)
    res["mutants_killed"] = killed
    return res


# ----------------------------------------------------------------------------
# MultiPL-E: one self-contained program per problem (prompt + body + tests)
# ----------------------------------------------------------------------------
_MULTIPL_E_TSCONFIG = {
    # The official MultiPL-E TypeScript tests declare `require` themselves and
    # are written for non-strict compilation.
    "compilerOptions": {"target": "ES2022", "module": "commonjs", "outDir": "out", "types": [],
                        "skipLibCheck": True, "strict": False, "noEmitOnError": True},
    "include": ["*.ts"],
}


def run_multipl_e(lang: str, program: str, timeout: int = 60) -> Dict[str, Any]:
    """Builds and runs one MultiPL-E program; ok when it exits 0 (every
    assertion in the official test block held)."""
    import sys
    if lang != "python" and not toolchain_available(lang):
        return _result(False, "unavailable", f"{lang} toolchain not installed (run inside the Docker image)")
    with tempfile.TemporaryDirectory(prefix=f"multipl-e-{lang}-") as tmp:
        env = _env(tmp)
        if lang == "python":
            _write_files(tmp, {"main.py": program})
            rc, out = _run([sys.executable, "main.py"], tmp, timeout, env)
        elif lang == "typescript":
            _write_files(tmp, {"main.ts": program, "tsconfig.json": json.dumps(_MULTIPL_E_TSCONFIG)})
            _link_node_modules(tmp)
            failed = _compile([os.path.join(tmp, "node_modules", ".bin", "tsc"), "-p", "tsconfig.json"], tmp, env)
            if failed:
                return failed
            rc, out = _run(["node", os.path.join("out", "main.js")], tmp, timeout, env)
        elif lang == "go":
            _write_files(tmp, {"humaneval_test.go": program, "go.mod": _go_mod()})
            rc, out = _run(["go", "test", "-count=1", f"-timeout={timeout}s", "./..."], tmp, COMPILE_TIMEOUT, env)
            if rc != 0 and ("[build failed]" in out or "[setup failed]" in out):
                return _fail("compile", out)
        elif lang == "java":
            # MultiPL-E Java prompts import org.javatuples (Pair, Triplet, ...).
            _write_files(tmp, {"Problem.java": program})
            tuples = os.path.join(POLYGLOT_HOME, "java", "lib", "javatuples-1.2.jar")
            failed = _compile(["javac", "-cp", tuples, "-d", "out", "Problem.java"], tmp, env)
            if failed:
                return failed
            rc, out = _run(["java", "-ea", "-cp", "out" + os.pathsep + tuples, "Problem"], tmp, timeout, env)
        elif lang == "rust":
            _write_files(tmp, {"main.rs": program})
            failed = _compile(["rustc", "--edition", RUST_EDITION, "-o", "prog", "main.rs"], tmp, env)
            if failed:
                return failed
            rc, out = _run([os.path.join(tmp, "prog")], tmp, timeout, env)
        elif lang == "csharp":
            shutil.copytree(os.path.join(POLYGLOT_HOME, "csharp", "app"), tmp, dirs_exist_ok=True)
            _write_files(tmp, {"Program.cs": program})
            # Debug configuration: the official tests use Debug.Assert.
            failed = _dotnet_build(tmp, env, ["-c", "Debug"])
            if failed:
                return failed
            rc, out = _run(["dotnet", os.path.join("bin_out", "App.dll")], tmp, timeout, env)
        else:
            return _result(False, "unavailable", f"no MultiPL-E runner for {lang}")
        if rc == 0:
            return _result(True, "ok", "all official tests passed")
        return _fail("run", out)
