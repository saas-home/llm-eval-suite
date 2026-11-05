# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **`medical_research` (Test 28) false-negative `extracted: null` on reasoning models.** The USMLE and PubMedQA calls used `max_tokens=500`/`300`, which was fully consumed by the model's reasoning trace before it emitted the `Answer: <letter>` / `Conclusion: <word>` line. The harness recorded a null extraction and a `PARTIAL` (7/10) even though the model's clinical reasoning was correct. Raised the per-call budget to `max_tokens=2048` so reasoning models can finish thinking and emit the final answer line. Verified live: `medical_research` now scores `10/10 (100.0%) -> PASS`.

### Added
- **Multi-Adapter Routing, Dynamic Slot Migration & Cross-Talk Isolation (Test 31)**:
  - Added Test 31 to `eval.py` and `reporting.py` for certifying multi-LoRA and multi-adapter serving runtimes (e.g. vLLM EMR Merging, S-LoRA).
  - Validates direct adapter routing, concurrent multi-adapter execution (dynamic slot contention), and byte-level activation/memory containment.
  - Gracefully skips with zero penalty on single-model endpoints.
- **Prometheus Server Metrics Integration (`/metrics`)**:
  - Added `fetch_server_metrics()` in `client.py` using pure standard library (`urllib.request`).
  - Automatically captures process resident memory (RSS MB), active running requests, waiting requests, EMR LoRA deferred queue count, KV cache usage percentage, and engine preemptions.
  - Displays runtime metrics in the scorecard and exports them in the JSON benchmark report.
- **Full Declarative Configuration Support (`config.json`)**:
  - `env_loader.py` and `eval.py` now support declarative benchmark execution from `config.json`: `tests`, `suite`, `exclude`, `parallel`, `quick`, `language`, `max_model_len`, `max_context`, and `adapters`.
  - Enables zero-CLI-variance, fully reproducible CI/CD benchmark runs.

### Fixed
- **Filtered False Positives in Prompt Injection Defense (Test 26)**:
  - Disambiguated reasoning-trace reflection and docstring quoting from genuine prompt hijacking. If a model generates the genuine implementation and does not abandon the task, prompt reflection is not misclassified as a security failure.
- **Calibrated Slot Recovery Threshold (Test 8)**:
  - Adjusted client socket abort follow-up threshold from 5.0s to 12.0s to account for standard single-stream prefill/decode latency on dense 27B+ parameter models.

### Added
- Decoupled **Pure Cognitive Capability & Intelligence Grading** from hardware serving speed in `reporting.py`:
  - `pure_capability_score_pct` strictly evaluates cognitive problem solving across all intelligence benchmarks (Tests 6-30: SWE-bench, CruxEval, AIME Olympiad, GPQA Diamond, IFEval, polyglot coding, etc.).
  - Added enterprise cognitive tier grading (`pure_capability_grade`): `A+ (Frontier Intelligence)`, `A (High Enterprise Capability)`, `B+ (Competent Enterprise Model)`, `B (Moderate Capability / Guardrails Required)`, `C (Marginal Capability / High Supervision)`, `F (Unsuitable)`.
  - CLI summary scorecard now prominently presents separate **`MODEL COGNITIVE CAPABILITY & INTELLIGENCE SCORECARD`** and **`HOSTING INFRASTRUCTURE & SERVING PERFORMANCE SCORECARD`**.
- Multi-problem datasets for CruxEval and AIME Olympiad in `data/`:
  - Added `data/reasoning/cruxeval_programs.json` for multi-item mental code execution tracing.
  - Added `data/math/aime_olympiad.json` for multi-item competition mathematical deduction (modular arithmetic CRT, Diophantine equations, and combinatorics).
- Cross-model head-to-head comparison (`reporting.py`):
  - Added `adv_cap` to compare pure cognitive capability advantage alongside task effectiveness and throughput.

### Fixed
- Fixed client request payload re-serialization in `client.py`: moved `json.dumps(payload)` inside the retry loop so automatic parameter fallbacks (e.g., stripping `reasoning_effort` on 400/422 validation errors) take immediate effect on re-attempts.
- Fixed IFEval reasoning parameter in `eval.py`: standardized on `reasoning_effort="none"` to properly suppress reasoning thought tokens during strictly constrained copywriting benchmarks on reasoning servers.
- Fixed CruxEval program 2 ground truth in `data/reasoning/cruxeval_programs.json` (`"AnTnI"`).
- Fixed AIME Olympiad answer extraction in `eval.py` to inspect the full model generation output (`full_text`) rather than only partial content chunks, properly capturing solutions from reasoning models.
- Fixed regex escape syntax in `eval.py` ensuring full compatibility with Python 3.12+ and 3.14 regex parsers without `bad escape \e` PatternErrors.

### Added
- 4 enterprise benchmark tests expanding suite from 26 to 30 tests:
  - Test 27: Google IFEval verifiable instruction following (word count bounds, markdown structure, forbidden words, starters/closers).
  - Test 28: Clinical & Medical Research (MedQA / USMLE board vignettes and PubMedQA biomedical literature consensus).
  - Test 29: Multi-step elementary & algebra math (OpenAI GSM8K word problems with step-by-step reasoning).
  - Test 30: Historical causality & chronology analysis (MMLU World, European & US History).
- Open benchmark dataset repository (`data/`):
  - Bundled verified test banks: `data/writing/ifeval_instructions.json`, `data/medical/usmle_clinical.json`, `data/medical/biomedical_pubmed.json`, `data/math/gsm8k_reasoning.json`, and `data/history/history_causality.json`.
  - Zero-dependency dataset downloader utility (`scripts/download_open_benchmarks.py`) for pulling full open datasets directly from official GitHub / Hugging Face sources.
- Domain benchmark evaluation engine (`domain_benchmarks.py`):
  - Deterministic IFEval constraint verification engine.
  - Conclusion-prioritized multiple choice answer extractor with elimination reasoning protection.
  - PubMedQA consensus decision extractor (`yes`/`no`/`maybe`).
  - GSM8K numeric extractor with float equivalence tolerance.
- Expanded Model Workload Qualification Matrix (`reporting.py`):
  - Added Clinical & Medical Life-Sciences Research domain.
  - Added Verifiable Instruction & Structured Writing domain.
  - Added Historical Causality & Humanities Research domain.
  - Updated Deep Reasoning domain to incorporate GSM8K math alongside AIME and GPQA Diamond.
- Added test aliases for tests 27-30 (`--test ifeval`, `--test medical`, `--test gsm8k`, `--test history`).
- Model Workload Suitability & Enterprise Purpose Qualification Engine (`reporting.py`):
  - Automatically assesses benchmark results against 6 enterprise workload domains: Autonomous Agents & Tool Execution, Deep Reasoning & Math, Long-Context Retrieval & RAG, Multimodal Vision & Document OCR, High-Throughput Serving & Real-Time APIs, and Software Engineering & Code Intelligence.
  - Generates domain qualification ratings (`EXCELLENT`, `COMPETENT`, `MARGINAL`, `NOT RECOMMENDED`), deployment recommendations, and specific operational cautions in terminal scorecards and JSON reports.
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
- Fixed Unit Test Synthesis (Test 23) prompt fixture in `languages.py`: provided complete production implementation of `PaymentProcessor.process_order` in the prompt instead of stubbed ellipsis (`...`), eliminating false-positive mock argument casing mismatches (`"usd"` vs `"USD"`).
- Fixed reasoning model scratchpad leak in IFEval (Test 27) and `client.py`:
  - Ensured `client.py` does not fallback empty `content` to `reasoning_content` when models exhaust token limits inside reasoning blocks.
  - Added graceful 400/422 retry when sending `reasoning_effort="off"` to non-reasoning endpoints.
  - Stripped `<think>.*?</think>` tags in `domain_benchmarks.verify_ifeval_rules` to prevent scratchpad planning thoughts from polluting constraint evaluations.
  - Increased IFEval `max_tokens` ceiling from 600 to 1,500 tokens.
- Fixed Test 4 (Architecture 4-Task Suite) token ceiling and balance factor matching in `eval.py`: increased `max_tokens` to 2000 and added support for `"balance factor"` / `"balanced"`.
- Fixed multiple-choice answer extraction in `domain_benchmarks.py`: added adverb support (`"The answer is clearly B"`) and restricted line-scanning fallbacks to avoid misidentifying option definition listings as answers.
- Fixed dead code in `benchmarks/vision.py` where streaming response handling was indented inside an `except Exception:` block after `continue`.
- Fixed single-server evaluation scoring in `judge_eval.py`: added single-mode keys (`score_10`, `score`, `total`) and tightened fallback regex to prevent literal `_10` in `score_10` coercing all scores to `10.0`.
- Fixed `SyntaxError` in `languages.py` for unit tests and refactoring tests using `from __future__ import annotations` by hoisting via `assemble_python_script`.
- Fixed `UnboundLocalError` in `eval.py` when `cfg_max_context` was referenced without definition.
- Guarded `prompt_tokens_details` in `benchmarks/context.py` and `benchmarks/reasoning.py` against `NoneType` attribute errors.
- Bounded default `adv_depth` by `max(milestones)` in `eval.py` to prevent prompt overflow on models with context limits < 32k.
- Fixed `run_test_client_abort` status mapping to mark unrecovered follow-ups as `FAIL` rather than `DEGRADED`.
- Added outermost balanced brace `{...}` fallback regex for `run_test_json_schema` when models output conversational prose without code fences.
- Resolved false negatives in `verify_security_review` for phrases like "not vulnerable" or "secure".
- Ignored `cls` alongside `self` in `verify_strict_typing` classmethod parameter inspection.
- Captured TTFT `t_first` and incremented token counts for streaming tool call chunks without text deltas in `client.py`.
- Parsed passing tests in `verify_unit_test` from `failures=` and `errors=` counts to accurately award partial credit instead of false zeros.
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
