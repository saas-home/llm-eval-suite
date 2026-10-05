# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changelog entries are generated from fragments in `changelog.d/` via
[towncrier](https://towncrier.readthedocs.io/). To add an entry, drop a file
like `changelog.d/<issue>.added.md` (see `changelog.d/README.md`).

<!-- towncrier release notes start -->

## [1.0.0] — 2026-10-05

### Added
- Enterprise LLM server benchmark & evaluation suite (22 tests + 14 benchmark tasks).
- Native Arena Mode: live concurrent head-to-head comparison of two endpoints.
- Cross-model efficiency & effectiveness comparison engine.
- Judge-based code-quality evaluation (`judge_eval.py`).
