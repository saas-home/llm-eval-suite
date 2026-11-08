# AGENTS.md

Rules for AI coding agents and people who work in this repository. The [README](README.md) describes what the suite tests and how to run it.

## Zero external dependencies

The suite runs on the Python 3.8+ standard library only (`urllib`, `json`, `threading`, `concurrent.futures`, `socket`, `subprocess`). Do **not** add a runtime dependency. Dev tools (ruff, mypy, pytest) are in the optional `dev` extra. The runtime never imports them.

## Layout

| File | Role |
|------|------|
| `eval.py` | Main program: core tests, scheduler, scoring, CLI |
| `compare.py` | Compares two or more reports, or two live endpoints |
| `judge_eval.py` | Code-quality scoring by an LLM judge |
| `client.py` | Shared OpenAI-compatible HTTP client: retries, request limit, logging, server metrics |
| `domain_benchmarks.py` | Shared answer extraction for graders (final answer text, multiple choice, yes/no) |
| `reporting.py` | Metric extraction, executive summary, scorecards, cross-model comparison |
| `env_loader.py` | Loads and saves config.json and .env |
| `languages.py` | Prompts and graders for Tests 20-25 in all six languages |
| `polyglot.py` | Compile-and-run graders for TypeScript, Go, Java, Rust and C# (Tests 20-25, 32) |
| `data/polyglot/` | Packs per test and language: support files, harness, reference answer (must pass), negative answer (must fail), mutants |
| `scripts/verify_polyglot_packs.py` | Checks that every pack passes good code and fails bad code (run it in the Docker image) |
| `suites.py` | Qualification suites, Tests 32-45 and 47-48 (coding, medical, academic, writing, personal, legal, contract review, summarisation): samples, repeats, 95% confidence intervals, pass^k |
| `workload.py` | Test 46: the deployment's workload (concurrent users, long new documents) against speed gates |
| `usecases.py` | Maps test results to use-case verdicts (Recommended / Usable with human review / Not recommended) |
| `report_md.py` | The Markdown decision report written next to every JSON report |
| `judge_validation.py` | Checks the judge against expert-rated public data (SummEval, LLMBar) |
| `data/reference/` | Vendor-published scores used in the comparison section, with source and retrieval date |
| `judge.py` | LLM judge that answers yes/no checklists (`--judge-endpoint`). Without one, or when it goes down, judged tests are graded objectively; the tested model never grades itself. |
| `ifeval.py` | Full Google IFEval verifier (standard-library port, Apache-2.0) |
| `data/suites/` | Seeded samples of official benchmarks and original tasks. Licenses are in `data/suites/SOURCES.md`. |
| `scripts/fetch_benchmark_data.py` | Downloads the official samples again (standard library, Hugging Face datasets-server API) |
| `benchmarks/` | Standalone probe scripts. They are used; do not delete them as dead code. |
| `tests/` | `unittest` suite (standard library, no pytest) |

Shared logic is in `client.py` and `reporting.py`. Do **not** copy protocol, scoring or formatting code into `eval.py` or `compare.py`. Use the shared modules.

## Testing

- Run the tests with `python -m unittest discover -s tests -v`. Use `unittest`, **not** pytest.
- Test every behavior. Each test must catch a specific defect. Do not add a test that no regression can break.
- After a behavior change, run the full suite. All tests must pass.

## Code style

- Write explicit code. Keep diffs to the size of the change. Use the plain, readable implementation.
- Handle edge cases fully, even if the run gets slower. A wrong number is worse than a slow run.
- Terminal output obeys `NO_COLOR` and non-TTY output. Use the color constants in `client.py`. Do not write ANSI codes directly.
- Keep infrastructure failures (network, timeout, 5xx, 429) separate from model failures. The completion rate depends on this separation.

## CI (`.github/workflows/ci.yml`)

1. `python -m py_compile` on every module, including `benchmarks/*.py`.
2. `python -m unittest discover -s tests -v`.
3. Smoke test of the CLIs: `eval.py --help`, `compare.py --help`, `judge_eval.py --help`.
4. Lint and type check (ruff, mypy). This job does not block a merge. Old code can have warnings. New code must be clean.

## Polyglot test packs

The Docker image compiles and runs the non-Python answers for Tests 20-25. It uses offline Vitest, JUnit 6 + Mockito, xUnit v3 + Moq, `go test` and `rustc --test`. Toolchain versions are build arguments at the top of the `Dockerfile`; framework versions are in `scripts/setup_polyglot_toolchains.sh`. After a version change, rebuild the image and run the pack check below. After you change a file under `data/polyglot/`, run:

    docker run --rm --network none -v "$PWD:/app" --entrypoint python3 \
        llm-eval-suite:latest scripts/verify_polyglot_packs.py

Every reference answer must PASS. Every negative answer must FAIL. If the toolchains are missing, the graders use a structural check, and its result is PARTIAL at most. An unverified answer is never a PASS.

Run model-written code only through `client.run_model_code` (own process group, output cap, no credentials). As root, it runs the code as the `evalcode` user, which owns only `cwd`. So give each run its own folder from `tempfile.TemporaryDirectory()`, and put every file the code must read in that folder. A harness signals success by printing its marker followed by the run id from `EVAL_RUN_ID` (`POLYGLOT_OK <id>`; in Python, `print('MARKER', __import__('os').environ['EVAL_RUN_ID'])`), checked with `client.harness_passed`. A bare marker is what an answer could print itself.

## Scheduling (phased runs)

- `ISOLATED_TESTS` in `eval.py` lists the tests that run alone before the parallel phase. Add a test to it if the test measures latency, throughput, cache state or memory, or needs an idle server.
- Test 46 (workload) is isolated: it sends its own concurrent users and measures what each one waits for.
- Every model request goes through `LLMClient.call`, which obeys `request_limiter`. Do not send raw HTTP requests in model-quality tests. They bypass the limit and can overload the server in the parallel phase.
- Log only with `client.log`. It buffers output per thread in the parallel phase. A bare `print` mixes lines from different tests.
- If the builds of a toolchain share files on disk, add the toolchain to `polyglot._SHARED_STATE_LOCKS`.
- When the tested server is unreachable, `LLMClient.call` sets `server_down` and sends no more requests. The scheduler then starts no new test, writes the report with `"incomplete"`, and exits with `EXIT_INCOMPLETE` (3). Before that, a refused or unreachable connection is retried only after the short waits in `client.UNREACHABLE_RETRY_DELAYS` (a restart or model reload). Do not add other retries or waits for a server that is down.

## Conventions

- Results go to `results/`, which git ignores. Results can contain internal IP addresses and API keys. Do not commit result files.
- git ignores `config.json` because it can contain API keys. `config.json.example` is the committed template.
- Record notable changes in `CHANGELOG.md`.
