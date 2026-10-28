# LLM Eval Suite

A zero-dependency, enterprise-grade evaluation suite and cross-model comparison engine for **any OpenAI-compatible LLM inference server** (vLLM, SGLang, Ollama, llama.cpp, TGI, commercial APIs).

---

## Purpose

This suite is built to benchmark, qualify, and select models for daily software engineering and reasoning workloads. It evaluates both **serving runtime performance** (latency, throughput, concurrency, prefix caching) and **frontier model intelligence** (long-context scaling, automated fuzz testing, bug patching, math competition logic, and graduate-level scientific deduction).

### Core Principles
- **Zero External Dependencies**: 100% Python 3.8+ standard library (`urllib`, `json`, `concurrent.futures`, `threading`, `subprocess`). No `pip install` required.
- **Honest Scoring**: Distinguishes infrastructure failures (timeouts, 5xx, 429 rate limits) from model-quality failures. If a server flakes, it is reported under **Completion Rate** rather than dragging down the model's accuracy score.
- **Token Economy**: Tracks solution conciseness (tokens spent per victory). In production, a concise model is faster, cheaper, and consumes less context.

---

## 19-Stage Evaluation Catalog

| # | Test Name | Key Capabilities Tested | Category |
|:---:|:---|:---|:---|
| **1** | **Streaming & TTFT** | Time-To-First-Token latency, chunk delivery, and decode throughput. | Flagship (1–12) |
| **2** | **Multimodal Vision** | Invoice document layout and key financial field extraction (`assets/invoice.png`). | Flagship |
| **3** | **Concurrency & Batching** | Continuous batching throughput under simultaneous client load (`threading.Barrier`). | Flagship |
| **4** | **4-Task Architecture Suite** | AVL Trees, concurrency bug isolation, distributed rate limiting, and hidden constraints. | Flagship |
| **5** | **Agent Tool Calling** | Multi-turn function calling with simulated live quota error and autonomous recovery. | Flagship |
| **6** | **JSON Schema Mode** | Strict constrained JSON decoding via `response_format={"type": "json_object"}`. | Flagship |
| **7** | **Prefix Caching Reuse** | RadixAttention / KV cache reuse: measures cold vs. warm prompt prefill speedup. | Flagship |
| **8** | **Client Socket Abort** | Aborts TCP socket mid-generation to verify server GPU slot deallocation. | Flagship |
| **9** | **Stop Sequences & Sampling** | Enforces exact multi-token stop word suppression and greedy determinism (`temp=0.0`). | Flagship |
| **10** | **Extreme Precision Arithmetic** | Multi-stage accounting ledger reconciliation with exact decimal precision. | Flagship |
| **11** | **API Error Compliance** | Validates HTTP 400/422 handling on missing payloads and graceful handling of invalid model IDs. | Flagship |
| **12** | **Dynamic Context Scaling** | Stepwise context prefill scaling up to model limits, with 85% depth needle retrieval. | Flagship |
| **13** | **Adversarial Graph Search** | Multi-hop routing through 32k context with deprecated and revoked temporal distractors. | Adversarial (13–15) |
| **14** | **Novel Algorithm (5k Fuzz)** | Synthesizes custom data structures; dynamically executes 5,000-op property fuzz tests. | Adversarial |
| **15** | **Frontier Needle Depth** | Composite arithmetic across needles buried at 0.5%, 50.0%, and 99.5% context depth. | Adversarial |
| **16** | **CruxEval Code Simulation** | Mental execution tracing of intricate Python code to predict exact state transitions. | Frontier (16–19) |
| **17** | **SWE-bench Bug Patching** | Synthesizes production regression bug fixes verified against dynamic unit test suites. | Frontier |
| **18** | **AIME Olympiad Math** | Solves high-tier competition discrete mathematics ($17^{2026} + 13^{2026} \pmod{1000} = 978$). | Frontier |
| **19** | **GPQA Diamond Science** | Graduate-level PhD scientific deduction across Quantum Mechanics, Spectroscopy, and Enzyme Kinetics. | Frontier |

---

## How to Run

### 1. Interactive Run (Default)
```bash
python3 eval.py
```
Discovers local endpoints (`http://127.0.0.1:8000/v1` or `config.json`), queries available models, detects context ceiling and parallel slots, and runs the benchmark.

### 2. Direct CLI Run
```bash
# Run full suite against a specific endpoint
python3 eval.py --endpoint http://127.0.0.1:8000/v1 --model meta-llama/Llama-3.1-70B-Instruct

# Non-interactive mode (ideal for scripts and CI)
python3 eval.py --auto
```

### 3. Selective Runs, Suites & Exclusions
```bash
# Run only Frontier Reasoning tests (16-19: CruxEval, SWE-bench, AIME, GPQA)
python3 eval.py --suite frontier

# Run specific tests by number or alias
python3 eval.py --test 3,12,19
python3 eval.py --test context_scaling,gpqa

# Exclude specific tests or ranges (alias: --skip)
python3 eval.py --exclude 14,17
python3 eval.py --skip 2-5
python3 eval.py --suite flagship --exclude vision
```

### 4. Fast Smoke Qualification
```bash
# Fast mode: compact context milestones, skips heavy prefill
python3 eval.py --quick
```

### 5. Probe & Diagnostics
```bash
# Quick <5s reachability and capability check
python3 eval.py --probe --endpoint http://127.0.0.1:8000/v1

# View history of past runs and leaderboard
python3 eval.py --history
python3 eval.py --leaderboard
```

### 6. Head-to-Head Comparison
```bash
# Compare two evaluated runs offline (zero GPU overhead)
python3 compare.py results/eval_modelA.json results/eval_modelB.json --out results/comparison.md

# Live Arena Mode: evaluate two servers side-by-side in real time
python3 compare.py --live --endpoint1 http://127.0.0.1:8080/v1 --model1 modelA \
                          --endpoint2 http://127.0.0.1:8000/v1 --model2 modelB
```

---

## Understanding the Results

At the end of a benchmark run, an **Executive Scorecard** is printed to the terminal and exported to `results/*.json`:

```text
========================================================================================
             EXECUTIVE EFFICIENCY & EFFECTIVENESS SCORECARD                             
========================================================================================
  Model Under Test        : Qwen3.8-Flash-Next-FP4
  Overall Quality Rating  : ★★★★☆ (4/5)
  Completion Rate         : 100.0% (19/19 tests completed)
  Effectiveness Rate      : 94.7% (18/19 tasks passed)
  Token Economy           : 720.4 tokens/passed task
  Total Solution Tokens   : 12,968 tokens
  Mean Decode Throughput  : 84.15 tok/s
  Mean TTFT               : 32.1 ms
  Composite Efficiency    : 91.2 / 100
========================================================================================
```

### Metric Definitions
- **Quality Rating (1–5 Stars)**: Fast visual indicator of model reliability. Capped at 3 stars if Completion Rate is under 80% (an unstable server prevents trustworthy scoring).
- **Completion Rate (%)**: Fraction of tests that completed without infrastructure failures (network drops, HTTP 5xx, or 429 rate limits).
- **Effectiveness Rate (%)**: Percentage of successfully passed tasks scored **only over completed tests**.
- **Token Economy (`tokens/passed task`)**: Total tokens emitted divided by passed tests. Measures verbosity vs. conciseness. Lower is better.
- **Mean Decode Throughput (`tok/s`)**: Average generation speed across all evaluated tasks.
- **Mean TTFT (`ms`)**: Time-To-First-Token responsiveness across single-turn, tool-calling, and streaming prompts.
- **Composite Efficiency Index (`0–100`)**: A balanced metric combining **Accuracy (60%)**, **Token Conciseness (20%)**, **Throughput (10%)**, and **TTFT (10%)**.

---

## Security Note

Tests **14** (5,000-op property fuzzing) and **17** (SWE-bench patch validation) execute model-generated Python code in a local subprocess to verify correctness.

When testing untrusted models or public endpoints, disable code execution:
```bash
python3 eval.py --no-execute
# or via environment variable:
EVAL_SAFE=1 python3 eval.py
```
When disabled, code execution tests gracefully fall back to AST syntax verification.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
