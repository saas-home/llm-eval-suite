# `llm-eval-suite`: Universal LLM Server Benchmark & Efficiency Evaluation Harness

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://python.org)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Stdlib)-success.svg)](requirements.txt)
[![Cross-Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-blueviolet.svg)](#)

A zero-dependency, enterprise-grade evaluation suite and cross-model comparison engine for **any OpenAI-compatible LLM inference server**.

Point it at any endpoint—**vLLM**, **SGLang**, **Ollama**, **llama.cpp**, **ExLlamaV3**, **TGI**, **LM Studio**, or commercial APIs (**OpenAI**, **DeepSeek**, **Groq**, **Mistral**)—to stress-test architectural capabilities, probe latency/throughput limits, measure token economy, and run head-to-head model comparisons.

---

## Key Highlights

- **Zero External Dependencies**: Implemented 100% using Python's standard library (`urllib`, `json`, `concurrent.futures`, `socket`, `subprocess`). No `pip install` required.
- **Universal Backend Compatibility**: Automatically queries `/v1/models` to discover model IDs, context limits, and capabilities, with seamless fallbacks for minimal or proxy gateways.
- **22 Enterprise Evaluation Stages**:
  - **Flagship Core**: Streaming latency, multimodal invoice parsing, continuous batching concurrency, code synthesis, agent tool recovery, strict JSON schema validation, prefix caching speedup, socket abort resilience, greedy reproducibility, high-entropy key-value recall, financial ledger math, dynamic code execution, HTTP error compliance, and dynamic context scaling (up to 200k+ tokens).
  - **Adversarial Hardening**: Multi-hop graph distractor retrieval, novel algorithmic fuzzing, combinatorial anti-constraints (IFEval tier), counterfactual axiomatic algebra, and frontier multi-needle depth precision.
  - **Frontier Reasoning**: CruxEval execution simulation, SWE-bench bug patch synthesis, and AIME Olympiad mathematics.
- **Head-to-Head Comparison Engine (`compare.py` / `--compare`)**: Compare two or more saved benchmark reports offline with zero GPU overhead. Directly measures **Effectiveness (Accuracy)** vs. **Token Economy (Solution Conciseness)**.
- **Environment Variable Auto-Discovery**: Automatically recognizes `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.

---

## Repository Layout

```
llm-eval-suite/
├── eval.py                   # Flagship 22-stage evaluation harness & comparison CLI
├── compare.py                # Cross-model head-to-head comparison CLI (offline, zero-GPU)
├── benchmarks/               # Individual standalone test probes & stress tests
│   ├── context.py            # Long-context scaling (8k to 200k+ tokens)
│   ├── concurrency.py        # Multi-client continuous batching stress test
│   ├── streaming.py          # TTFT and streaming latency evaluation
│   ├── vision.py             # Multimodal document extraction benchmark
│   ├── adversarial.py        # Multi-turn adversarial agent & recovery protocol
│   ├── reasoning.py          # Deep parallel tree reasoning & logic search
│   ├── precision.py          # High-entropy ledger reconciliation
│   ├── code_synthesis.py     # Dynamic execution & fuzz testing on generated code
│   └── needle_recall.py      # High-entropy key-value needle retrieval
├── assets/
│   └── invoice.png           # High-resolution invoice document for vision evaluation
├── results/                  # Evaluated benchmark JSON reports and Markdown comparisons
├── pyproject.toml            # Optional packaging metadata
├── requirements.txt          # Notes zero dependencies (100% Python stdlib)
├── LICENSE                   # Apache 2.0 License
└── README.md                 # Full documentation
```

---

## Quickstart

### 1. Run the Full 22-Stage Benchmark

Run against a local server (defaults to `http://127.0.0.1:8000/v1` or prompts interactively):
```bash
python3 eval.py
```

Run against a specific endpoint and model:
```bash
python3 eval.py --endpoint http://127.0.0.1:8000/v1 --model meta-llama/Llama-3.1-70B-Instruct
```

Run non-interactively with environment variables:
```bash
export OPENAI_BASE_URL="http://10.0.0.5:8000/v1"
export OPENAI_MODEL="qwen2.5-coder-32b-instruct"
export OPENAI_API_KEY="sk-..."

python3 eval.py --auto
```

### 2. Fast Smoke Qualification Mode

Skip heavy context prefill to quickly test basic capabilities:
```bash
python3 eval.py --quick
```

### 3. Run Specific Test Suites or Tests

```bash
# Run only Frontier Reasoning tests (Tests 20-22: CruxEval, SWE-bench, AIME)
python3 eval.py --suite frontier

# Run only Adversarial Hardening tests (Tests 15-19)
python3 eval.py --suite adversarial

# Run specific tests by number or alias
python3 eval.py --test 14,20,22
python3 eval.py --test context_scaling,aime
```

### 4. Run Standalone Specialized Probes

Each benchmark probe in `benchmarks/` can also be executed independently:

```bash
# Long context prefill & decode scaling (8k up to 200k+)
python3 benchmarks/context.py --tokens 8000 32000 64000 128000 200000

# Continuous batching and concurrency stress test
python3 benchmarks/concurrency.py --parallel 8 --tokens 256

# Streaming TTFT & decode throughput
python3 benchmarks/streaming.py

# Multimodal document extraction
python3 benchmarks/vision.py

# Multi-turn adversarial agent & error recovery
python3 benchmarks/adversarial.py

# Deep parallel tree reasoning & logic search
python3 benchmarks/reasoning.py
```

---

## Cross-Model Comparison Engine

Evaluate how efficiently Model A solves problems compared to Model B using saved JSON reports:

```bash
python3 compare.py results/eval_modelA.json results/eval_modelB.json --out results/comparison.md
```
*(Or via `python3 eval.py --compare results/eval_modelA.json results/eval_modelB.json`)*

### Sample Comparison Matrix

```text
====================================================================================================
                        CROSS-MODEL EFFICIENCY & EFFECTIVENESS COMPARISON
====================================================================================================
  Dimension / Metric            | Model A: qwen3.8-27b        | Model B: llama-3.1-70b        | Relative Advantage
----------------------------------------------------------------------------------------------------
  Effectiveness (Pass Rate)     | 21/22 (95.5%)               | 18/22 (81.8%)                 | Model A (+13.6%)
  Token Economy (Tokens/Victory)| 1,148.2 tokens/task         | 2,055.6 tokens/task           | Model A (1.8x more concise)
  Total Solution Tokens         | 24,113 tokens               | 36,990 tokens                 | Model A (12,877 fewer tokens)
  Mean Decode Speed             | 44.12 tok/s                 | 28.40 tok/s                   | Model A (+15.72 tok/s)
  Mean Time-to-First-Token      | 48.2 ms                     | 62.4 ms                       | Model A (-14.2 ms)
  Composite Efficiency Index    | 88.4 / 100                  | 74.6 / 100                    | Model A (+13.8 pts)
====================================================================================================
```

### Understanding the Efficiency Metrics

- **Token Economy (`tokens_per_passed_task`)**: Measures the average number of generated tokens spent to successfully solve a task. In production, a model that produces correct code in 400 tokens is vastly cheaper, faster, and more context-efficient than one that requires 2,500 tokens of rambling derivation for the same outcome.
- **Composite Efficiency Index (`0-100`)**: A weighted index balancing task accuracy (60%), token conciseness (20%), generation throughput (10%), and TTFT responsiveness (10%).

---

## 22-Stage Evaluation Catalog

| # | Test Name | Target Capability & Verification Methodology |
|:---:|:---|:---|
| **1** | **Streaming Latency** | Time-To-First-Token (TTFT) and incremental chunk delivery verification. |
| **2** | **Multimodal Vision** | Invoice document layout and field extraction from `assets/invoice.png`. |
| **3** | **Concurrency & Batching** | Parallel continuous batching throughput and FIFO request queueing. |
| **4** | **4-Task Architecture** | AVL tree, concurrent bounded buffer, event-driven ledger, distributed tracing. |
| **5** | **Agent Tool Calling** | Multi-turn agent loop, schema validation, and autonomous error recovery. |
| **6** | **Strict JSON Schema** | Constrained decoding via `response_format={"type": "json_object"}`. |
| **7** | **Prefix Caching Reuse** | RadixAttention / KV cache reuse: measures cold vs. warm prompt prefill speedup. |
| **8** | **Client Socket Abort** | Simulates client disconnection mid-generation; verifies GPU worker slot release. |
| **9** | **Stop Sequences & Sampling** | Enforces exact multi-token stop word truncation and deterministic greedy decoding. |
| **10** | **High-Entropy KV Recall** | Recalls 10 high-entropy keys buried across thousands of distractor tokens. |
| **11** | **Precision Ledger Math** | Reconciles multi-entity transaction ledgers with strict decimal precision. |
| **12** | **Dynamic Code Testing** | Synthesizes an LRU Cache with TTL; dynamically executes unit tests in-memory. |
| **13** | **API Error Compliance** | Validates proper HTTP 400/422 handling for malformed or out-of-bounds requests. |
| **14** | **Context Window Scaling** | Stepwise context expansion (4k, 8k, 16k, 32k, 64k, 128k, 200k+) measuring prefill scaling. |
| **15** | **Adversarial Graph Search** | Multi-hop shortest path search through distractor-heavy topological graphs. |
| **16** | **Novel Algorithm Fuzzing** | Synthesizes non-standard algorithms; runs 5,000 in-memory property fuzz tests. |
| **17** | **Combinatorial Anti-Constraints** | IFEval-tier negative constraints (lipograms, strict word counts, syntax rules). |
| **18** | **Counterfactual Algebra** | Axiomatic re-definition of arithmetic operators; evaluates symbolic deduction. |
| **19** | **Frontier Needle Depth** | Deep needle retrieval at target context horizon under high information entropy. |
| **20** | **CruxEval Program Simulation** | Accurate mental execution tracing of intricate Python code to predict output. |
| **21** | **SWE-bench Bug Patching** | Synthesizes unified git diff patches resolving edge-case production bugs. |
| **22** | **AIME Olympiad Mathematics** | Solves high-tier competition math problems requiring multi-step discrete derivations. |

---

## Server Setup Examples

### vLLM
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
python3 eval.py -e http://localhost:8000/v1
```

### Ollama
```bash
ollama serve
python3 eval.py -e http://localhost:11434/v1 -m qwen2.5:32b
```

### llama.cpp
```bash
./llama-server -m model.gguf --port 8080
python3 eval.py -e http://localhost:8080/v1
```

### ExLlamaV3 / Simplex
```bash
./linux/simplex start
python3 eval.py -e http://127.0.0.1:8888/v1
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
