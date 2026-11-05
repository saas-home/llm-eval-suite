# LLM Eval Suite

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/Dependencies-Zero_External-brightgreen?style=for-the-badge" alt="Zero Dependencies" />
  <img src="https://img.shields.io/badge/Docker-Multi--Arch-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
  <img src="https://img.shields.io/badge/Tests-31_Enterprise_Benchmarks-blueviolet?style=for-the-badge" alt="31 Benchmarks" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge" alt="Apache 2.0" />
</p>

An enterprise-grade evaluation suite, benchmarking tool, and cross-model comparison engine for **any OpenAI-compatible LLM inference server** ([vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Ollama](https://github.com/ollama/ollama), [llama.cpp](https://github.com/ggerganov/llama.cpp), [TGI](https://github.com/huggingface/text-generation-inference), or commercial APIs).

Evaluates **serving runtime performance** (latency, continuous batching throughput, prefix caching, socket teardown, multi-adapter / multi-LoRA routing & memory containment) and **frontier intelligence** across long-context scaling, multi-hop reasoning, security defense, **verifiable instruction following (Google IFEval)**, **medical diagnosis & biomedical literature (MedQA / PubMedQA)**, **multi-step mathematics (GSM8K)**, **historical analysis (MMLU History)**, and **polyglot software engineering** in 6 industry toolchains.

---

## Key Highlights

- **Zero External Runtime Dependencies**: 100% Python 3.8+ standard library (`urllib`, `json`, `concurrent.futures`, `threading`, `socket`, `subprocess`). No `pip install` required.
- **Decoupled Cognitive & Serving Scorecards**: Separates pure model intelligence & cognitive capability (independent of GPU decode speed) from hosting infrastructure throughput & serving efficiency.
- **Multi-Adapter & Multi-LoRA Certification**: Validates dynamic adapter routing, concurrent multi-adapter execution under slot contention, and bit-level activation/memory containment (e.g. for vLLM EMR Merging or S-LoRA).
- **Live Prometheus Server Telemetry (`/metrics`)**: Auto-harvests server resident memory (RSS MB), active running requests, waiting queue depth, KV cache utilization %, EMR LoRA deferred queues, and engine preemptions directly from server Prometheus endpoints.
- **Declarative Configuration (`config.json`)**: Configure benchmarks declaratively via JSON or `.env` files for zero-CLI-variance CI/CD pipelines.
- **Model Purpose Qualification**: Automatically maps benchmark performance to an enterprise **Task Suitability Matrix**, determining which production workloads (Coding, Agents, Math, Clinical, Writing, High-Throughput Serving) the model is ready for.
- **Industry-Standard Open Datasets**: Built-in canonical test banks from **Google IFEval**, **OpenAI GSM8K**, **USMLE / MedQA**, **PubMedQA**, **CruxEval**, **AIME Olympiad**, **GPQA Diamond**, and **MMLU History**. Runs 100% offline out-of-the-box.
- **Polyglot SWE Benchmarking**: Evaluates coding capabilities across **Python**, **TypeScript**, **Go**, **Java**, **Rust**, and **C# (.NET 10 LTS)**.
- **Zero Host Impact (Docker)**: Pre-packaged multi-arch container (`x86_64` & `aarch64`) bundling official LTS compilers with dual persistent volume mounts (`./results` and `./logs`).
- **Honest Scoring Architecture**: Distinguishes infrastructure failures (timeouts, 5xx, 429 rate limits) from model accuracy failures. Network flakiness is tracked under **Completion Rate** without dragging down model effectiveness.
- **Token Economy Metric**: Measures solution conciseness (tokens spent per victory). Concise models save latency, compute, and context window budget.
- **Head-to-Head & N-Way Analysis**: Compare 2 or more model benchmark reports offline or run real-time side-by-side Live Arena benchmarks.

---

## 🏆 Latest Certified Result

**`Qwen3.8-Flash-Next-FP4`** served by vLLM `v0.33.0rc1.dev848` on an NVIDIA DGX Spark (GB10, 128 GB) achieved a **perfect 30/30 (100%)** run — **Grade A+ (Frontier Intelligence)**, ⭐⭐⭐⭐⭐, **100% pure capability**, with all 9 enterprise workload domains rated **EXCELLENT**.

👉 **[Full Certification Report →](results/EVAL_REPORT.md)** · [Machine-readable JSON →](results/enterprise_eval_127.0.0.1_8080_20261105_141137.json)

---

## 31-Stage Benchmark Catalog

The suite is organized into **5 core capability pillars** spanning 31 standardized tests:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             LLM EVAL SUITE                                  │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│  Serving Runtime    │ API Contracts & IF  │ Long Context & Attention        │
│  (Tests 1, 2, 3, 7, │ (Tests 5,6,9,10, 27)│ (Tests 12, 13, 15)              │
│   8, 11, 31)        │                     │                                 │
├─────────────────────┴─────────────────────┼─────────────────────────────────┤
│  Polyglot Software Engineering (SWE)      │ Cognitive Reasoning, STEM & Life│
│  (Tests 4, 14, 17, 20, 21, 22, 23, 24, 25)│ (Tests 16, 18, 19, 26, 28, 29,30)│
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
| **27** | **Google IFEval Writing** | Deterministic verifiable instruction following (word counts, bullet formatting, start/end anchors, forbidden words). | `contract` |
| **28** | **Clinical & Medical Research** | USMLE clinical board diagnosis (MedQA) & biomedical research literature consensus (PubMedQA `yes`/`no`/`maybe`). | `reasoning` |
| **29** | **Multi-Step Math (GSM8K)** | Multi-step word problem deduction and calculation verification from OpenAI GSM8K. | `reasoning` |
| **30** | **Historical Causality (MMLU)** | Historical cause-and-effect deduction, treaty sovereignty, and anti-hallucination analysis from MMLU History. | `reasoning` |
| **31** | **Multi-Adapter Routing & Isolation** | Multi-LoRA concurrent execution, dynamic slot migration contention, and cross-talk activation/memory containment. | `serving` |

---

## Quick Start

### 1. Zero-Install Native Run (Host)

Requires Python 3.8+ (no `pip install` needed):

```bash
# 1. Probe server reachability (<5s)
python3 eval.py --probe -e http://127.0.0.1:8080/v1

# 2. Run full evaluation suite (Tests 1-31)
python3 eval.py -e http://127.0.0.1:8080/v1 -m Qwen3.8-Flash-Next-FP4

# 3. Fast qualification smoke test
python3 eval.py -e http://127.0.0.1:8080/v1 -q

# 4. Declarative run with config.json
python3 eval.py --config config.json
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
python3 eval.py --suite serving       # Tests 1, 2, 3, 7, 8, 11 (Latency, Batching, Caching)
python3 eval.py --suite contract      # Tests 5, 6, 9, 10, 27 (Tools, Schema, Stops, IFEval)
python3 eval.py --suite swe           # Tests 4, 14, 17, 20-25 (Coding & Toolchains)
python3 eval.py --suite reasoning     # Tests 16, 18, 19, 26, 28, 29, 30 (Math, Science, Medicine, History)
python3 eval.py --suite multi_adapter # Test 31 (Multi-LoRA & multi-adapter routing & isolation)

# 2. Run specific tests by number or alias
python3 eval.py --test multi_adapter  # Multi-adapter routing & memory isolation (Test 31)
python3 eval.py --test ifeval         # Google IFEval verifiable instruction following (Test 27)
python3 eval.py --test medical        # MedQA (USMLE) & PubMedQA clinical reasoning (Test 28)
python3 eval.py --test gsm8k          # OpenAI GSM8K multi-step algebra math (Test 29)
python3 eval.py --test history        # MMLU History causal deduction (Test 30)
python3 eval.py --test 20-25          # Full SWE polyglot series
python3 eval.py --test gpqa,aime      # Frontier Olympiad math & science

# 3. Exclude tests
python3 eval.py --exclude 2        # Skip vision if endpoint is text-only
python3 eval.py --exclude 12,13,15 # Skip heavy context tests
```

---

## Declarative Configuration (`config.json` & `.env`)

Execute reproducible benchmark runs without CLI flag sprawl using a declarative `config.json` or `.env` file:

```json
{
  "_comment": "llm-eval-suite enterprise certification configuration",
  "endpoint": "http://127.0.0.1:8000/v1",
  "model": "Qwen3.8-27B",
  "max_context": 32768,
  "max_model_len": 32768,
  "parallel": 4,
  "suite": "all",
  "exclude": "2",
  "adapters": ["finance-lora", "legal-lora", "medical-lora"]
}
```

```bash
# Execute using default config.json or custom path
python3 eval.py --config config.json

# Override any config parameter on-the-fly from the CLI
python3 eval.py --config config.json --quick --suite serving
```

---

## Model Workload Suitability & Purpose Qualification

Rather than presenting raw scores alone, `llm-eval-suite` automatically projects performance onto an enterprise **Workload Qualification Matrix** to determine where a model is production-ready and where human oversight or guardrails are mandatory:

```text
========================================================================================
             MODEL WORKLOAD SUITABILITY & PURPOSE QUALIFICATION                         
========================================================================================
  Workload Domain                             | Score    | Rating       | Deployment Status
  --------------------------------------------------------------------------------------
  Deep Reasoning, Science & Math              | 100.0%   | EXCELLENT    | Highly Recommended
  High-Throughput Serving & Low-Latency APIs  | 100.0%   | EXCELLENT    | Highly Recommended
  Software Engineering & Code Intelligence    |  88.9%   | EXCELLENT    | Highly Recommended
  Autonomous Agents & Tool Execution          |  75.0%   | COMPETENT    | Recommended with Guardrails
  Long-Context Retrieval & RAG Pipelines      |  75.0%   | COMPETENT    | Recommended with Guardrails
  Verifiable Instruction & Structured Writing |  50.0%   | MARGINAL     | Human Review Required
  Clinical & Medical Life-Sciences Research   |  50.0%   | MARGINAL     | Human Review Required
  Historical Causality & Humanities Research  |   0.0%   | NOT RECOMMENDED | Not Recommended
========================================================================================
```

Each evaluated domain provides explicit recommendations:
* **Recommended Enterprise Purposes (Best Suited For):** Identifies tasks where the model consistently succeeds (e.g. multi-step algebra, code refactoring, SQLi detection, streaming APIs).
* **Operational Limitations & Workload Cautions:** Pinpoints specific failure modes (e.g. medical hallucination risks, word-count violations, or chronological conflations).

---

## Open Benchmark Datasets & Offline Evaluation

All evaluation data is bundled directly under the [`data/`](data/) directory in lightweight JSON format:
* **`data/writing/ifeval_instructions.json`**: Standardized prompts with deterministic regex rules (word limits, bullet counts, forbidden words, starters/closers).
* **`data/medical/usmle_clinical.json`**: Verified clinical board examination vignettes with 4-choice diagnostic options.
* **`data/medical/biomedical_pubmed.json`**: Clinical research questions based on PubMed abstracts with gold-standard decisions (`yes`, `no`, `maybe`).
* **`data/math/gsm8k_reasoning.json`**: Multi-step word problems from OpenAI Grade School Math with verified numeric solutions.
* **`data/history/history_causality.json`**: Historical questions evaluating causal relationships and treaty sovereignty from MMLU History.

To download additional raw test sets from official repositories, use the zero-dependency utility script:
```bash
# Download official open datasets directly from GitHub / Hugging Face
python3 scripts/download_open_benchmarks.py --all
```

---

## Cross-Model Comparison & Live Arena

Compare two or more evaluated benchmark reports offline with zero additional GPU overhead, or benchmark two active servers concurrently in real time:

```bash
# 1. Live Arena Mode: concurrent side-by-side benchmark of two live servers
python3 compare.py --live \
  --endpoint1 http://127.0.0.1:8080/v1 --model1 Qwen3.8-Flash-Next-FP4 \
  --endpoint2 http://10.0.0.2:8000/v1 --model2 Llama-3.3-70B-Instruct \
  -o arena.md

# 2. Offline Report Comparison: Model A vs Model B (outputs terminal table and Markdown report)
python3 compare.py results/eval_modelA.json results/eval_modelB.json -o comparison.md

# 3. N-Way Comparison Matrix (3+ models)
python3 compare.py results/run1.json results/run2.json results/run3.json -o comparison.md
```

<details>
<summary><b>Click to preview sample Head-to-Head Comparison Table</b></summary>

```text
====================================================================================================
          CROSS-MODEL HEAD-TO-HEAD EFFICIENCY & EFFECTIVENESS BENCHMARK REPORT          
====================================================================================================
  Model A : Qwen3.8-Flash-Next-FP4 (http://127.0.0.1:8080/v1)
  Model B : Llama-3.3-70B-Instruct (http://10.0.0.2:8000/v1)

Core Performance Metric             | Model A (Qwen3.8-Flash-N)  | Model B (Llama-3.3-70B-I)  | Advantage
----------------------------------------------------------------------------------------------------
Pure Cognitive Capability           | 96.0% (A+)                 | 88.0% (A )                 | Model A (+8.0%)
Overall Task Effectiveness          | 29/30 (96.7%)              | 27/30 (90.0%)              | Model A (+6.7%)
Token Economy (tok/task)            | 650.2 tok/task             | 910.4 tok/task             | Model A (1.40x leaner)
Total Solution Tokens Consumed      | 18,855 tokens              | 24,580 tokens              | Model A (5,725 fewer)
Mean Decode Speed (tok/s)           | 92.40 tok/s                | 48.15 tok/s                | Model A (1.92x faster)
Mean Prefill TTFT (ms)              | 28.4 ms                    | 45.2 ms                    | Model A (16.8 ms faster)
Total Benchmark Wall Time           | 64.20 s                    | 112.50 s                   | Model A (48.30 s faster)
Hardware Serving Efficiency Index   | 93.4 / 100.0               | 81.2 / 100.0               | Model A (+12.2 pts)
====================================================================================================
```
</details>

---

## Understanding the Results

At the conclusion of each benchmark run, distinct **Cognitive Capability** and **Hosting Infrastructure** scorecards are displayed and saved to `results/`:

```text
========================================================================================
             MODEL COGNITIVE CAPABILITY & INTELLIGENCE SCORECARD                        
========================================================================================
  Model Under Test        : Qwen3.8-Flash-Next-FP4
  Intelligence Grade      : A+ (Frontier Intelligence)
  Pure Capability Score   : 100.0%  [24/24 Tasks Solved (100.0%)]
  Overall Task Accuracy   : 30/30 Passed (100.0%)
  Solution Conciseness    : 1573.4 tokens/passed task (Token Economy)
  Evaluation Methodology  : Open Benchmark Standard (GPQA / AIME / SWE / IFEval / CruxEval)
========================================================================================

========================================================================================
             HOSTING INFRASTRUCTURE & SERVING PERFORMANCE SCORECARD                     
========================================================================================
  Target Serving Endpoint : http://127.0.0.1:8080/v1
  Mean Decode Throughput  : 49.14 tok/s
  Mean Prefill TTFT       : 5633.0 ms
  Concurrent Clients      : 4 active parallel stream(s)
  Total Suite Wall Time   : 1703.92 s
  Total Solution Tokens   : 47,203 tokens consumed
  Serving Efficiency Idx  : 88.0 / 100.0 (Throughput & conciseness rating)
  Live Engine Telemetry   : RSS Memory 18.2 GB | Running: 4 | Waiting: 0 | KV Cache: 12.4%
========================================================================================
```

### Metrics Breakdown

- **Intelligence Grade & Pure Capability Score (%)**: Evaluates pure cognitive problem solving across all intelligence benchmarks (Tests 6-30: SWE-bench, CruxEval, AIME Olympiad, GPQA Diamond, IFEval, polyglot coding, USMLE, PubMedQA, GSM8K, and MMLU History) completely decoupled from hardware decode speed.
  - Grades: `A+` (Frontier, $\ge 90\%$), `A` (High Enterprise, $\ge 80\%$), `B+` (Competent, $\ge 70\%$), `B` (Moderate, $\ge 60\%$), `C` (Marginal, $\ge 50\%$), `F` (Unsuitable).
- **Completion Rate (%)**: Fraction of tests that completed without infrastructure failures (network drops, HTTP 5xx, or 429 rate limits).
- **Overall Task Accuracy (Pass@1)**: Percentage of successfully passed tasks scored **strictly over completed tests**.
- **Token Economy (`tok/task`)**: Emitted tokens divided by passed victories. Measures conciseness vs. verbosity. Lower is superior.
- **Mean Decode Throughput (`tok/s`)**: Average generation speed across all tasks.
- **Mean Prefill TTFT (`ms`)**: Time-To-First-Token responsiveness across single-turn, tool-calling, and streaming prompts.
- **Serving Efficiency Index (`0–100`)**: A hardware infrastructure index balancing **Accuracy (60%)**, **Token Economy (20%)**, **Throughput (10%)**, and **TTFT (10%)**.
- **Live Server Telemetry**: When Prometheus `/metrics` is exposed by the server (e.g. vLLM or SGLang), resident set size (RSS MB), active running requests, waiting queue depth, KV cache utilization %, EMR LoRA deferred queue count, and engine preemptions are captured and reported.
- **Multi-Adapter Isolation**: Evaluates bit-level activation containment and concurrent throughput without cross-adapter interference (Test 31).

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
# Run complete test suite (133 unit tests)
python3 -m unittest discover -s tests -v

# Run syntax compilation check across all modules
python3 -m py_compile *.py benchmarks/*.py tests/*.py
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
