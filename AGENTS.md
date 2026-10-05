# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## The one rule that matters most

**Zero external dependencies.** The entire suite runs on the Python 3.8+
standard library (`urllib`, `json`, `threading`, `concurrent.futures`,
`socket`, `subprocess`). Do **not** add a runtime dependency. Dev-only tooling
(ruff, mypy, towncrier, pytest) lives in the optional `dev` extra and is never
imported by the runtime.

## Layout

| File | Role |
|------|------|
| `eval.py` | Main orchestrator: 22 endpoint tests + scoring + CLI |
| `compare.py` | Head-to-head / live-arena comparison of two endpoints |
| `judge_eval.py` | LLM-judge code-quality evaluation |
| `client.py` | Shared OpenAI-compatible HTTP client + retry + formatting |
| `reporting.py` | Metric extraction, executive summary, cross-model compare |
| `env_loader.py` | config.json / .env loading & saving |
| `benchmarks/` | Standalone probe scripts (intentional, not dead code) |
| `tests/` | `unittest` suite (stdlib, no pytest) |

Shared logic lives in `client.py` / `reporting.py`. **Do not duplicate**
protocol, scoring, or formatting code across `eval.py` / `compare.py` — reuse
the shared modules.

## Testing

- Test runner: `python -m unittest discover -s tests -v` (stdlib `unittest`,
  **not** pytest).
- **Every behavior must be tested.** Do not add a test that no regression
  would break. Every test must correspond to a defect it would catch.
- After changing behavior, run the full suite and confirm it stays green.

## Code style

- **Explicit over clever.** Right-sized diffs. Prefer the boring, readable
  implementation.
- **Edge cases:** thorough handling over speed. This is a benchmarking tool —
  a wrong number is worse than a slow run.
- Terminal output respects `NO_COLOR` and non-TTY (see `client.py` color
  constants). Reuse those; don't hardcode ANSI codes.
- Infrastructure failures (network / timeout / 5xx / 429) are distinguished
  from model-quality failures. Keep that separation — it powers the
  completion-rate scoring.

## CI expectations (`.github/workflows/ci.yml`)

1. `python -m py_compile` on every module (including `benchmarks/*.py`).
2. `python -m unittest discover -s tests -v`.
3. Smoke-test CLIs: `eval.py --help`, `compare.py --help`, `judge_eval.py --help`.
4. Non-blocking lint/type job (ruff + mypy) — legacy debt is tolerated, but new
   code should be clean.

## Conventions

- Results write to `results/` (gitignored — they may contain internal IPs and
  API keys). Never commit result dumps.
- `config.json` is gitignored (may hold API keys). `config.json.example` is the
  committed template.
- Changelog: add a fragment to `changelog.d/` (see its README) instead of
  editing `CHANGELOG.md` directly.
