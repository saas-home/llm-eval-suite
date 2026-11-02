# LLM Eval Suite

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/Dependencies-Zero_External-brightgreen?style=for-the-badge" alt="Zero Dependencies" />
  <img src="https://img.shields.io/badge/Docker-Multi--Arch-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
  <img src="https://img.shields.io/badge/Tests-26_Enterprise_Benchmarks-blueviolet?style=for-the-badge" alt="26 Benchmarks" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge" alt="Apache 2.0" />
</p>

An enterprise-grade evaluation suite, benchmarking tool, and cross-model comparison engine for **any OpenAI-compatible LLM inference server** ([vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Ollama](https://github.com/ollama/ollama), [llama.cpp](https://github.com/ggerganov/llama.cpp), [TGI](https://github.com/huggingface/text-generation-inference), or commercial APIs).

Evaluates **serving runtime performance** (latency, continuous batching throughput, prefix caching, socket teardown) and **frontier intelligence** across long-context scaling, multi-hop reasoning, security defense, and **polyglot software engineering** in 6 industry toolchains.

---

## Key Highlights

- **Zero External Runtime Dependencies**: 100% Python 3.8+ standard library (`urllib`, `json`, `concurrent.futures`, `threading`, `socket`, `subprocess`). No `pip install` required.
- **Polyglot SWE Benchmarking**: Evaluates coding capabilities across **Python**, **TypeScript**, **Go**, **Java**, **Rust**, and **C# (.NET 10 LTS)**.
- **Zero Host Impact (Docker)**: Pre-packaged multi-arch container (`x86_64` & `aarch64`) bundling official LTS compilers with dual persistent volume mounts (`./results` and `./logs`).
- **Honest Scoring Architecture**: Distinguishes infrastructure failures (timeouts, 5xx, 429 rate limits) from model accuracy failures. Network flakiness is tracked under **Completion Rate** without dragging down model effectiveness.
- **Token Economy Metric**: Measures solution conciseness (tokens spent per victory). Concise models save latency, compute, and context window budget.
- **Head-to-Head & N-Way Analysis**: Compare 2 or more model benchmark reports offline with zero additional GPU overhead.

---

## 26-Stage Benchmark Catalog

The suite is organized into **5 core capability pillars** spanning 26 standardized tests:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             LLM EVAL SUITE                                  │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│  Serving Runtime    │ API Contracts       │ Long Context & Attention        │
│  (Tests 1,2,3,7,8,11) (Tests 5,6,9,10)    │ (Tests 12, 13, 15)              │
├─────────────────────┴─────────────────────┼─────────────────────────────────┤
│  Polyglot Software Engineering (SWE)      │ Cognitive Reasoning & Safety    │
│  (Tests 4, 14, 17, 20, 21, 22, 23, 24, 25)│ (Tests 16, 18, 19, 26)          │
└───────────────────────────────────────────┴─────────────────────────────────┘
```

| # | Benchmark Domain | Key Capabilities Evaluated | Pillar Suite |
|:---:|:---|:---|:---:|
| **1** | **Streaming & TTFT** | Time-To-First-Token latency, chunk delivery consistency, and decode speed. | `serving` |
| **2** | **Multimodal Vision** | Invoice document layout reasoning and key-value financial extraction (`assets/invoice.png`). | `serving` |
| **3** | **Concurrency & Saturation** | Continuous batching throughput under simultaneous parallel client load (`1, 2, 4+`). | `serving` |
| **4** | **4-Task Architecture Suite** | AVL Trees, concurrency deadlock isolation, distributed rate limiting, and hidden constraints. | `swe` |
| **5** | **Agent Tool Calling** | Multi-turn function calling with simulated live quota error and autonomous recovery. | `contract` |
| **6** | **JSON Schema Mode** | Strict constrained JSON decoding via `response_format={"type": "json_object"}`. | `contract` |
| **7** | **Prefix Caching Reuse** | RadixAttention / KV cache reuse: measures cold vs. warm prompt prefill speedup. | `serving` |
| **8** | **Client Socket Abort** | Aborts TCP socket mid-generation to verify server GPU slot deallocation and recovery. | `serving` |
| **9** | **Stop Sequences & Sampling** | Enforces exact multi-token stop word suppression and greedy determinism (`temp=0.0`). | `contract` |
| **10** | **Precision Arithmetic** | Multi-stage accounting ledger reconciliation with exact decimal precision. | `contract` |
| **11** | **API Error Compliance** | Validates HTTP 400/422 handling on invalid schemas and missing payload boundaries. | `serving` |
| **12** | **Dynamic Context Scaling** | Stepwise context prefill scaling up to model limits with 85% depth needle retrieval. | `context` |
| **13** | **Adversarial Graph Search** | Multi-hop routing through 32k context with deprecated and revoked temporal distractors. | `context` |
| **14** | **Novel Algorithm (5k Fuzz)** | Synthesizes custom algorithms; executes 5,000-op dynamic property fuzz tests. | `swe` |
| **15** | **Frontier Needle Depth** | Composite arithmetic across needles buried at 0.5%, 50.0%, and 99.5% context depth. | `context` |
| **16** | **CruxEval Mental Tracing** | Mental execution tracing of intricate code to predict exact state transitions. | `reasoning` |
| **17** | **SWE-bench Bug Patching** | Synthesizes production regression bug fixes verified against dynamic unit test suites. | `swe` |
| **18** | **AIME Olympiad Math** | High-tier competition discrete mathematics ($17^{2026} + 13^{2026} \pmod{1000} = 978$). | `reasoning` |
| **19** | **GPQA Diamond Science** | Graduate-level PhD scientific deduction across Quantum Mechanics, Spectroscopy, and Kinetics. | `reasoning` |
| **20** | **Surgical Unified Diff** | Strict patch synthesis validated by syntax parser and `git apply --check`. | `swe` |
| **21** | **Multi-File Cross-Ref & DI** | Interface implementation and constructor dependency injection across decoupled modules. | `swe` |
| **22** | **Security Code Review** | OWASP vulnerability audit: detects SQLi and race conditions with 0 false alarms on clean code. | `swe` |
| **23** | **Unit Test Synthesis** | Generates regression suites with stubs and test doubles (`mock`, `patch`). | `swe` |
| **24** | **Strict Type Compliance** | Advanced generics and static type checking verified via type checkers (`mypy` / AST). | `swe` |
| **25** | **Safe Refactoring** | Idiomatic code simplification guaranteed against behavioral regression test suites. | `swe` |
| **26** | **Prompt Injection Defense** | Delimiter security protocol thwarting system prompt overrides embedded in untrusted inputs. | `reasoning` |

---

## Quick Start

### 1. Zero-Install Native Run (Host)

Requires Python 3.8+ (no `pip install` needed):

```bash
# 1. Probe server reachability (<5s)
python3 eval.py --probe -e http://127.0.0.1:8080/v1

# 2. Run full evaluation suite (Tests 1-26)
python3 eval.py -e http://127.0.0.1:8080/v1 -m Qwen3.8-Flash-Next-FP4

# 3. Fast qualification smoke test
python3 eval.py -e http://127.0.0.1:8080/v1 -q
```

---

### 2. Containerized Run (Docker — Recommended)

Run benchmarks with **zero impact on the host machine**. All compiler toolchains are pre-packaged.

```bash
# Build the multi-arch container image
docker build -t llm-eval-suite:latest .

# Run benchmark with dual volume mounts for reports and diagnostic logs
docker run --rm \
  --network host \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  llm-eval-suite:latest -e http://127.0.0.1:8080/v1 -m Qwen3.8-Flash-Next-FP4
```

> [!TIP]
> **Testing Multiple Models**: To benchmark multiple models simultaneously, launch separate container instances pointing to each model endpoint. Both instances safely write timestamped reports into `./results/` and execution traces into `./logs/`.

#### Using Docker Compose
```bash
# Start evaluation with pre-configured volume mounts
docker compose run --rm eval -e http://127.0.0.1:8080/v1
```

---

## Polyglot SWE Benchmark Mode

Tests 20 through 25 support the 6 primary enterprise software ecosystems:

| Language | Default Runtime Toolchain | Verification Strategy |
|:---|:---|:---|
| **Python** | Python 3.12 LTS + `mypy` | AST parser + dynamic harness execution |
| **TypeScript** | Node.js 22 Active LTS + `tsc` | AST grammar parser + compiler diagnostics |
| **Go** | Go `1.22` | Grammar structural parser + go vet |
| **Java** | OpenJDK 21 LTS | AST grammar parser + JUnit validation |
| **Rust** | Rustc / Cargo `1.75` | Grammar balance + invariant preservation |
| **C#** | Microsoft .NET 10 SDK (LTS) | Grammar balance + type compliance |

```bash
# Run SWE benchmarks in TypeScript
python3 eval.py -e http://127.0.0.1:8080/v1 -s swe -l typescript

# Run SWE benchmarks in Rust
python3 eval.py -e http://127.0.0.1:8080/v1 -s swe -l rust

# Run SWE benchmarks in C# (.NET 10)
python3 eval.py -e http://127.0.0.1:8080/v1 -s swe -l csharp

# Polyglot Matrix Mode: runs Tests 20-25 across ALL 6 languages simultaneously
python3 eval.py -e http://127.0.0.1:8080/v1 -s swe -l polyglot
```

When running in Polyglot Matrix mode, an executive cross-language scorecard is displayed:

```text
========================================================================================
 POLYGLOT SWE BENCHMARK MATRIX (Tests 20-25 across 6 Industry Toolchains) 
========================================================================================
Language     | Diff(20)   | Multi(21)  | Sec(22)    | Test(23)   | Type(24)   | Refactor(25) | Pass Rate
----------------------------------------------------------------------------------------
Python       | PASS       | PASS       | PASS       | PASS       | PASS       | PASS         | 6/6 (100%)
TypeScript   | PASS       | PASS       | PASS       | PASS       | PASS       | PASS         | 6/6 (100%)
Go           | PASS       | PASS       | PASS       | PASS       | PASS       | PASS         | 6/6 (100%)
Java         | PASS       | PASS       | PASS       | PASS       | PASS       | PASS         | 6/6 (100%)
Rust         | PASS       | PASS       | PASS       | PASS       | PASS       | PASS         | 6/6 (100%)
C#           | PASS       | PASS       | PASS       | PASS       | PASS       | PASS         | 6/6 (100%)
========================================================================================
```

---

## Selective Testing & Filtering

Execute specific slices of the evaluation suite to save time:

```bash
# 1. Run a specific Pillar Suite
python3 eval.py --suite serving    # Tests 1, 2, 3, 7, 8, 11 (Latency, Batching, Caching)
python3 eval.py --suite swe        # Tests 4, 14, 17, 20-25 (Coding & Toolchains)
python3 eval.py --suite reasoning  # Tests 16, 18, 19, 26 (Math, Science, Safety)

# 2. Run specific tests by number or alias
python3 eval.py --test 20          # Surgical git diff verification
python3 eval.py --test 20-25       # Full SWE polyglot series
python3 eval.py --test gpqa,aime   # Frontier Olympiad math & science

# 3. Exclude tests
python3 eval.py --exclude 2        # Skip vision if endpoint is text-only
python3 eval.py --exclude 12,13,15 # Skip heavy context tests
```

---

## Offline Cross-Model Comparison

Compare two or more evaluated benchmark reports with **zero GPU overhead**:

```bash
# Compare Model A vs Model B (outputs terminal table and Markdown scorecard)
python3 compare.py results/eval_modelA.json results/eval_modelB.json -o comparison.md

# N-Way comparison matrix (3+ models)
python3 compare.py results/run1.json results/run2.json results/run3.json -o comparison.md
```

<details>
<summary><b>Click to preview sample Head-to-Head Comparison Table</b></summary>

```
========================================================================================
               HEAD-TO-HEAD BENCHMARK COMPARISON MATRIX                                 
========================================================================================
  Model 1: Qwen3.8-Flash-Next-FP4 (http://127.0.0.1:8080/v1)
  Model 2: Llama-3.3-70B-Instruct (http://10.0.0.2:8000/v1)

Core Performance Metric          | Model 1              | Model 2              | Advantage
----------------------------------------------------------------------------------------
Effectiveness (Pass Rate)        | 25/26 (96.2%)        | 24/26 (92.3%)        | Model 1 (+3.8%)
Token Economy (tok/task)         | 650.2 tok/task       | 910.4 tok/task       | Model 1 (1.40x leaner)
Total Solution Tokens            | 16,255 tokens        | 21,850 tokens        | Model 1 (5,595 fewer)
Mean Decode Speed (tok/s)        | 92.40 tok/s          | 48.15 tok/s          | Model 1 (1.92x faster)
Mean Prefill TTFT (ms)           | 28.4 ms              | 45.2 ms              | Model 1 (16.8 ms faster)
Total Benchmark Wall Time        | 64.20 s              | 112.50 s             | Model 1 (48.30 s faster)
Composite Efficiency Index       | 93.4 / 100.0         | 81.2 / 100.0         | Model 1 (+12.2 pts)
========================================================================================
```
</details>

---

## Understanding the Results

At the conclusion of each benchmark run, an **Executive Scorecard** is displayed and saved to `results/`:

```text
========================================================================================
             EXECUTIVE EFFICIENCY & EFFECTIVENESS SCORECARD                             
========================================================================================
  Model Under Test        : Qwen3.8-Flash-Next-FP4
  Overall Quality Rating  : ★★★★★ (5/5)
  Completion Rate         : 100.0% (26/26 tests completed)
  Effectiveness (Pass@1)  : 25/26 Passed (96.2%)
  Total Solution Tokens   : 16,255 tokens consumed
  Token Economy           : 650.2 tokens/passed victory (Solution conciseness)
  Mean Decode Throughput  : 92.40 tok/s
  Mean Prefill TTFT       : 28.4 ms
  Total Benchmark Time    : 64.20 s
  Composite Efficiency Idx: 93.4 / 100.0
========================================================================================
```

### Metrics Breakdown

- **Overall Quality Rating (1–5 Stars)**: Instant visual reliability indicator. Automatically capped at 3 stars if server Completion Rate drops below 80%.
- **Completion Rate (%)**: Fraction of tests that completed without infrastructure failures (network drops, HTTP 5xx, or 429 rate limits).
- **Effectiveness Rate (Pass@1)**: Percentage of successfully passed tasks scored **strictly over completed tests**.
- **Token Economy (`tok/task`)**: Emitted tokens divided by passed victories. Measures conciseness vs. verbosity. Lower is superior.
- **Mean Decode Throughput (`tok/s`)**: Average generation speed across all tasks.
- **Mean Prefill TTFT (`ms`)**: Time-To-First-Token responsiveness across single-turn, tool-calling, and streaming prompts.
- **Composite Efficiency Index (`0–100`)**: A comprehensive index balancing **Accuracy (60%)**, **Token Economy (20%)**, **Throughput (10%)**, and **TTFT (10%)**.

---

## Security Sandbox

> [!WARNING]
> Tests **14** (5,000-op fuzzing), **17** (SWE-bench fix), **23** (unit test synthesis), and **25** (safe refactor) execute model-generated code in a temporary subprocess to verify correctness.

When benchmarking untrusted models or public endpoints, disable local execution:

```bash
python3 eval.py -e http://127.0.0.1:8080/v1 --no-execute
# or via environment variable:
EVAL_SAFE=1 python3 eval.py -e http://127.0.0.1:8080/v1
```

When code execution is disabled, all tests safely fall back to static syntax verification and AST structural analysis.

---

## Verification & Testing

The evaluation suite includes a standalone `unittest` suite with **zero third-party test runners required**:

```bash
# Run complete test suite (107 unit tests)
python3 -m unittest discover -s tests -v

# Run syntax compilation check across all modules
python3 -m py_compile *.py benchmarks/*.py tests/*.py
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
