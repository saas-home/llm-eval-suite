# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-client concurrency & continuous batching stress suite measuring prefill TTFT and decode throughput across 1, 2, 3, and 4 concurrent clients and 2K, 4K, 8K, 16K, 32K context horizons.
- Test exclusion flags (`--exclude`, `--skip`, `--exclude-test`, `--skip-test`) allowing selective omission of tests, ranges, aliases, or suites.
- GPQA Diamond graduate science reasoning benchmark (Test 19).
- Centralized URL normalization helper (`normalize_endpoint`).
- Hoisting of `from __future__ import ...` in dynamically assembled test scripts.

### Changed
- Streamlined test suite to 19 enterprise benchmark tests.
- Replaced automatic mutating of `config.json` with dynamic endpoint discovery.
- Scoped configuration loading to matching endpoints, preventing cross-server parameter bleed.
- Removed `changelog.d/` fragment workflow and `towncrier` dependency in favor of direct `CHANGELOG.md` maintenance.

### Fixed
- Fixed model selection when pointing CLI to remote endpoints with different model names than local defaults.
- Fixed infrastructure failure classification (`_classify_infra`) to properly inspect error lists in composite concurrency tests.
- Hardened `.gitignore` to protect generated dumps in `results/`, environment files (`.env.*`), and editor artifacts.

## [1.0.0] — 2026-10-05

### Added
- Enterprise LLM server benchmark & evaluation suite (22 tests + 14 benchmark tasks).
- Native Arena Mode: live concurrent head-to-head comparison of two endpoints.
- Cross-model efficiency & effectiveness comparison engine.
- Judge-based code-quality evaluation (`judge_eval.py`).
