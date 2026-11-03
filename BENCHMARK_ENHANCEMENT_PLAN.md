# Industry-Standard Benchmark Enhancement Roadmap

## 1. Executive Summary & Core Philosophy

The goal of this roadmap is to elevate `llm-eval-suite` from a 26-test benchmark harness to an enterprise-grade evaluation suite on par with modern industry standards (**Google IFEval**, **Berkeley Function Calling Leaderboard v3**, **NVIDIA RULER**, and **GenAI-Perf**).

### Uncompromising Architectural Principles
* **Zero External Runtime Dependencies:** Pure Python 3.8+ standard library (`urllib`, `json`, `re`, `threading`, `concurrent.futures`, `random`, `math`, `subprocess`). No runtime bloat, no third-party package vulnerabilities.
* **Universal OpenAI-Compatible Interface:** Single client protocol running seamlessly against any modern inference runtime: **vLLM**, **SGLang**, **Ollama**, **LMDeploy**, **MLX**, **llama.cpp**, and **TGI**.
* **Deterministic & Reproducible Grading:** Programmatic verification (regex string matching, AST parsing, and isolated code execution) rather than subjective LLM-as-a-judge scoring.
* **Infrastructure vs. Intelligence Isolation:** Strict separation of networking/queue failures (5xx, 429, timeouts) from model capability failures.
* **Actionable Task Suitability:** Direct mapping of benchmark results to production deployment readiness (Coding, Agentic Tools, Long-Context, Serving Throughput, Reasoning).

---

## 2. Benchmark Gap Analysis & Industry Mapping

| Industry Benchmark | Current Suite Coverage | Target Enhancement |
| :--- | :--- | :--- |
| **Google IFEval** | Partial (JSON mode, stop seq) | **Full Verifiable Instruction Following Suite** (20+ rules) |
| **Berkeley BFCL v3** | Basic (2-turn single tool) | **Parallel, Nested Schema & Negative Abstention** |
| **NVIDIA RULER** | Basic NIAH (Needle in Hay) | **Multi-Hop Pointer Tracing & Distributed Aggregation** |
| **GenAI-Perf / vLLM** | Lockstep Batch Concurrency | **Poisson Burst Arrival & SLA Saturation Curve** |
| **SWE-bench Verified** | Single-file traceback patch | **Multi-file Repo Navigation & Diff Application** |
| **OpenAI SimpleQA** | None | **Factual Precision & Calibrated Abstention** |

---

## 3. Four Core Enhancement Pillars

### Pillar 1: Verifiable Instruction Adherence (IFEval Specification)

#### The Problem
In real-world applications, models frequently ignore explicit user constraints—exceeding length bounds, missing mandatory headers, using forbidden characters, or failing rigid formatting rules.

#### Proposed Architecture
Implement an automated constraint verification engine (`ifeval.py`) using pure standard library regex and token counters:

1. **Detectable Formatting Constraints:**
   - Exact bullet count validation (e.g., exactly 4 markdown bullet items).
   - Heading structure validation (e.g., must contain `# Executive Summary` and `## Analysis`).
   - Highlight section validation (e.g., exactly 3 phrases enclosed in `*italics*` or `**bold**`).
2. **Length & Density Constraints:**
   - Strict word count bounds ($N_{\min} \le \text{words} \le N_{\max}$).
   - Sentence count enforcement (counted via sentence-boundary regex).
   - Paragraph count enforcement (double newline delimiters `\n\n`).
3. **Linguistic & Negative Constraints:**
   - Forbidden character/punctuation rules (e.g., zero commas `,` across the entire response).
   - Frequency counters (e.g., the target word `"deterministic"` appears exactly 3 times).
   - Strict boundary anchoring (prompt must start with `"Result: "` and conclude with `"End of transmission."`).
4. **Scoring Metrics:**
   - **Prompt-Level Strict:** $1.0$ if all constraints within a prompt pass; $0.0$ if any single constraint fails.
   - **Instruction-Level Strict:** Fractional score based on percentage of individual rules satisfied.

---

### Pillar 2: Advanced Tool Calling & Agentic Guardrails (BFCL v3 Specification)

#### The Problem
In production agentic architectures, models fail when:
1. They need to trigger multiple tools in parallel (e.g., fetching 3 distinct APIs at once).
2. They encounter complex nested schemas (arrays of objects with enum constraints).
3. They hallucinate function calls on conversational queries where no tool is needed (negative abstention).

#### Proposed Architecture
Extend client tool handling and add a comprehensive BFCL-style multi-turn agent probe:

1. **Parallel Function Calling:**
   - Prompt: *"Fetch real-time stock quotes for AAPL, GOOG, and MSFT simultaneously."*
   - Verification: Verify the model returns a single assistant message containing 3 distinct items in `tool_calls`.
2. **Nested Object & Enum Schema Validation:**
   - Provide schemas with deep parameters:
     ```json
     {
       "filters": {
         "date_range": {"start": "2026-01-01", "end": "2026-12-31"},
         "status": ["PENDING", "APPROVED"]
       }
     }
     ```
   - Validate payload against expected types using Python standard library `json` AST without requiring external schema validators.
3. **Negative Tool Abstention:**
   - Provide 10 active tool declarations.
   - Supply a purely conceptual prompt: *"Explain the mathematical derivation of backpropagation."*
   - Verification: Strictly verify `tool_calls` is `None` or empty. If the model invokes any tool, award $0.0$.
4. **Stateful Multi-Turn Clarification:**
   - Omit a required parameter in the user turn; verify the model asks for the missing argument instead of inventing/hallucinating a default.

---

### Pillar 3: True Long-Context Multi-Hop & Aggregation (RULER Specification)

#### The Problem
Standard Needle-in-a-Haystack (NIAH) benchmarks only test superficial keyword retrieval. Models can score 100% on NIAH while suffering severe coreference collapse and reasoning degradation over 32k–128k context lengths.

#### Proposed Architecture
Enhance `benchmarks/needle_recall.py` and `eval.py` with RULER-style synthetic probes:

1. **Multi-Hop Pointer Tracing:**
   - Scatter sequential pointers across the context window at configured depths (e.g., 10%, 40%, 75%, 95%):
     - Needle 1: `"Node Alpha delegates authority to Node Beta"`
     - Needle 2: `"Node Beta assigns task to Agent Gamma"`
     - Needle 3: `"Agent Gamma holds security key: SECRET_DELTA_882"`
   - Prompt: *"Following the chain of delegation starting from Node Alpha, what security key is held?"*
   - Requires dynamic cross-context coreference resolution across tens of thousands of tokens.
2. **Distributed Frequency & Ledger Aggregation:**
   - Distribute 25 distinct transaction entries across a 64k text corpus containing noise.
   - Prompt: *"Sum the total transaction value for entity 'FinanceOrg' and identify the most frequent transaction type."*
   - Verifies holistic context scanning rather than single-location keyword lookup.
3. **Context Degradation Curve:**
   - Automatically execute the benchmark across scaling steps: 8k, 16k, 32k, 64k, 128k.
   - Plot the effective context retention boundary (where accuracy drops below 80%).

---

### Pillar 4: Poisson / Bursty Traffic & SLA Profiling (GenAI-Perf Specification)

#### The Problem
Current multi-client tests run in lockstep (workers fire in rigid loops). Real production workloads follow bursty, non-deterministic arrival rates (Poisson processes), causing KV-cache swapping and queue contention.

#### Proposed Architecture
Enhance `benchmarks/load.py` with stochastic traffic simulation:

1. **Poisson Arrival Engine:**
   - Calculate inter-request intervals via exponential distribution: `random.expovariate(target_rps)`.
2. **Dynamic RPS Saturation Sweep:**
   - Automatically scale traffic from 0.5 RPS to 20.0 RPS in stepped increments.
   - At each stage, sample 50 requests and compute:
     - $P_{50}, P_{90}, P_{99}$ Time-To-First-Token (TTFT)
     - Inter-Token Latency (ITL) variance
     - Request queue duration vs generation duration
3. **Automated Knee-Point Discovery:**
   - Flag the exact RPS saturation point where:
     - $P_{99}\text{ TTFT} > 2.0\text{ seconds}$, or
     - Server begins rejecting requests with HTTP 429 / 503 / 504.

---

## 4. Phase-by-Phase Implementation Plan

### Phase 1: Verifiable Instruction Adherence (IFEval Engine)
* **Target:** Create `ifeval.py`, add Test 27 to `eval.py`
* **Deliverables:**
  - Standardized rule validator covering word count, sentence count, bullet count, start/end tokens, and forbidden characters.
  - Comprehensive prompt bank with single and compound constraints.
  - Integration into `compute_task_suitability_matrix()` under `"Instruction Adherence"`.

### Phase 2: Advanced Tool Calling & Agentic Guardrails (BFCL v3)
* **Target:** Enhance `client.py`, update `eval.py`
* **Deliverables:**
  - Client support for parsing and aggregating parallel `tool_calls` in streaming mode.
  - Test case for parallel tool execution (3 concurrent tools in one turn).
  - Test case for negative tool abstention (zero tool calls on conversational query).
  - Test case for nested schema parsing with type verification.

### Phase 3: Long-Context Multi-Hop & Bursty Saturation Profiling
* **Target:** Enhance `benchmarks/needle_recall.py`, `benchmarks/load.py`
* **Deliverables:**
  - Pointer-chain generator for multi-hop coreference resolution up to 128k context.
  - Long-range aggregation probe across distributed transaction entries.
  - Poisson arrival distribution runner in `benchmarks/load.py` with automated SLA knee-point calculation.

### Phase 4: Reporting & Cross-Model Comparison Integration
* **Target:** Update `reporting.py`, `compare.py`
* **Deliverables:**
  - Add Instruction Adherence and Multi-Hop Reasoning to the Task Suitability Matrix.
  - Add latency SLA percentile visualizations ($P_{50}, P_{90}, P_{99}$) to markdown reports.
  - Update `unittest` suite to maintain 100% test coverage across all new validators.

---

## 5. Verification & Quality Gates

1. **Unit Testing:**
   - Every programmatic validator (regex, AST parser, counter) must have deterministic test cases in `tests/test_*.py`.
2. **Zero Dependency Check:**
   - Strict adherence to Python standard library; verify that no `requirements.txt` or third-party packages are required for execution.
3. **Cross-Server Validation:**
   - Validate that new tests run cleanly against local vLLM (`http://localhost:8888/v1`) and remote MLX (`http://172.16.16.29:8085/v1`) without protocol errors.
