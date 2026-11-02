# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-arch Docker containerization (`Dockerfile`, `compose.yaml`, `.dockerignore`):
  - Zero-host-impact containerized evaluation with isolated multi-arch toolchains (Ubuntu 24.04 base for `x86_64` and `aarch64`).
  - Bundled LTS runtimes: Microsoft .NET 10 SDK (`10.0.403`), OpenJDK 21 LTS (`21.0.12.1`), Node.js 22 Active LTS (`v22.23.4`) + TypeScript (`7.0.2`), Python 3.12 LTS, Go (`1.22.2`), Rust (`1.75.0`), and Git (`2.43.0`).
  - Dual volume architecture mounting `./results:/app/results` for JSON reports and `./logs:/app/logs` for real-time diagnostic execution traces.
- Diagnostic file logging subsystem in `client.py`:
  - Added `init_file_logger()` and `close_file_logger()` with automated `atexit` cleanup and `--log-file` / `--no-log-file` CLI flags.
- Polyglot SWE Benchmark Suite (`languages.py`):
  - Full support for the 6 major industry software engineering ecosystems: **Python, TypeScript, Go, Java, Rust, and C#**.
  - New `--language` / `-l` CLI parameter (`python`, `typescript`, `go`, `java`, `rust`, `csharp`, `polyglot`; default: `python` for 100% backward compatibility).
  - Polyglot Language Matrix mode (`--language polyglot`) running SWE tests 20-25 across all 6 toolchains and producing an executive polyglot cross-language scorecard.
  - Adaptive dual-mode verification: dynamic test execution in subprocess when host toolchains are available, with graceful zero-dependency static structural/AST/regex verification fallback.
  - Polyglot markdown code extractor (`extract_code_by_language`) supporting Python, TypeScript, Go, Java, Rust, and C# fences and truncated generations.
- 7 enterprise software engineering (SWE) & safety defense benchmarks (expanding suite from 19 to 26 tests):
  - Test 20: Surgical Unified Diff syntax & `git apply --check` patch verification.
  - Test 21: Multi-file cross-reference & constructor dependency injection (AST validated).
  - Test 22: Security code review & OWASP vulnerability audit (SQLi & race condition detection, 0 false alarms on safe code).
  - Test 23: Unit test synthesis with mocks and test doubles (`unittest.mock.MagicMock` / `patch`).
  - Test 24: Strict type compliance and static analysis (`mypy --strict` with AST fallback).
  - Test 25: Safe refactoring with behavioral invariant preservation against regression test suites.
  - Test 26: Prompt injection and delimiter defense protocol against untrusted code docstring overrides.
- 5 capability pillar suites (`--suite serving`, `--suite contract`, `--suite context`, `--suite swe`, `--suite reasoning`, `--suite all`) with full alias support.
- N-way cross-model efficiency & effectiveness comparison matrix supporting 3 or more models simultaneously in both terminal and Markdown reports.
- Multi-client concurrency & continuous batching stress suite measuring prefill TTFT and decode throughput across 1, 2, 3, and 4 concurrent clients and 2K, 4K, 8K, 16K, 32K context horizons.
- Test exclusion flags (`--exclude`, `--skip`, `--exclude-test`, `--skip-test`) allowing selective omission of tests, ranges, aliases, or suites.
- GPQA Diamond graduate science reasoning benchmark (Test 19).
- Centralized URL normalization helper (`normalize_endpoint`).
- Hoisting of `from __future__ import ...` in dynamically assembled test scripts.

### Changed
- Streamlined and grouped CLI option layout in `eval.py` and `compare.py`:
  - Organized `eval.py --help` into 6 clear logical argument groups: Target Endpoint, Test Selection, Execution & Safety, Output & Logging, Utilities & Analysis, and Configuration.
  - Suppressed internal micro-tuning (`--adversarial-depth`, `--context-ratio`, etc.) and cross-tool arena options from the primary help display, reducing visual noise from 33 scattered flags to 14 high-value options while preserving 100% backward compatibility.
  - Grouped `compare.py --help` into Live Arena Mode, Offline Report Comparison, and Output & Configuration.
- Refactored `compare.py` to eliminate duplicate raw urllib streaming SSE logic by delegating to `client.LLMClient` with real-time chunk callbacks.
- Expanded `TOTAL_TESTS = 26` across catalog and scoring reporting.
- Replaced automatic mutating of `config.json` with dynamic endpoint discovery.
- Scoped configuration loading to matching endpoints, preventing cross-server parameter bleed.
- Removed `changelog.d/` fragment workflow and `towncrier` dependency in favor of direct `CHANGELOG.md` maintenance.

### Fixed
- Fixed N-way comparison matrix formatting bug in `reporting.py` where unnested `{col_w}` inside f-strings caused literal `:<` format specifiers to be printed.
- Fixed ANSI terminal alignment in `reporting.py`: unified `fmt_status` padding to compute whitespace based on visible character count, eliminating column drift between color TTY and `NO_COLOR` / plain-text environments.
- Hardened `verify_git_diff` in `languages.py` to prevent false-positive PASS marks on unexpected subprocess/git execution crashes.
- Added `extract_unified_diff` in `languages.py` to strip conversational model prose and ensure clean patch application with `git apply --check`.
- Fixed file logger descriptor leaks by closing existing handles before re-initialization.
- Fixed concurrency parallel levels bug where `--parallel >= 5` (e.g. 6) was capped at 4 clients; now evaluates `[1, 2, 4, parallel]`.
- Fixed prefix caching cold/warm test salting to ensure independent cold baseline per round without false cache hits from prior rounds.
- Fixed stop sequence false-positive validation by verifying numbers past the stop sequence are suppressed.
- Fixed model selection when pointing CLI to remote endpoints with different model names than local defaults.
- Fixed infrastructure failure classification (`_classify_infra`) to properly inspect error lists in composite concurrency tests.
- Hardened `.gitignore` to protect generated dumps in `results/`, environment files (`.env.*`), and editor artifacts.

## [1.0.0] — 2026-10-05

### Added
- Enterprise LLM server benchmark & evaluation suite (22 tests + 14 benchmark tasks).
- Native Arena Mode: live concurrent head-to-head comparison of two endpoints.
- Cross-model efficiency & effectiveness comparison engine.
- Judge-based code-quality evaluation (`judge_eval.py`).
