# 🏆 Enterprise Certification Report — Qwen3.8 Flash Next FP4

**Model Under Test:** `Qwen3.8-Flash-Next-FP4` (`nvidia/Qwen3.8-Flash-Next-FP4-Local`, NVFP4 MoE + MXFP8 dense)
**Serving Runtime:** vLLM `v0.33.0rc1.dev848+g962d72187` — upstream `main` @ `831dd92674` (fully synced, 0 behind) + dict `hf-overrides` spec-decode pick, full CUDA + Rust source build for `sm_121a`
**Hardware:** NVIDIA DGX Spark (GB10 Blackwell, 128 GB Unified Memory)
**Suite:** `llm-eval-suite` @ `dc9c7e0` — 31-stage Enterprise Full Benchmark (Docker)
**Run:** 2026-11-05 13:43 – 14:11 UTC (28.4 min) · 4 parallel streams · temp=0.0

---

## ⭐ Headline Result

| Metric | Value |
| :--- | :---: |
| **Overall Task Accuracy** | **30 / 30 Passed (100.0%)** |
| **Pure Capability Score** | **100.0%** — 24/24 cognitive tasks |
| **Intelligence Grade** | **A+ (Frontier Intelligence)** |
| **Quality Rating** | ⭐⭐⭐⭐⭐ (5 / 5) |
| **Serving Efficiency Index** | **88.0 / 100** |
| **Mean Decode Throughput** | 49.14 tok/s |
| **Mean Prefill TTFT** | 5,633 ms |
| **Token Economy** | 1,573.4 tokens / passed task |
| **Total Suite Wall Time** | 1,703.9 s · 47,203 solution tokens |

> Test 31 (Multi-Adapter Isolation) is **SKIPPED by design** on this single-model endpoint (< 2 adapters exposed) and is excluded from the grading denominator — hence 30/30.

---

## 📋 Full 31-Stage Results

| # | Test | Result | Detail |
|:-:|:---|:---:|:---|
| 1 | Streaming & Latency | **PASS** | TTFT 180.7 ms · 37.1 tok/s |
| 2 | Multimodal Vision (Invoice) | **PASS** | TTFT 1,470 ms · 55.0 tok/s |
| 3 | Parallel Batching (C1–C4) | **PASS** | 47.0 / 71.6 / 84.4 / **108.5 tok/s agg** |
| 4 | 4-Task Architecture Suite | **PASS** | 4/4 tasks · 45.5 tok/s avg |
| 5 | Tool Calling & Agentic Recovery | **PASS** | Recovered from injected quota error |
| 6 | JSON Schema Mode | **PASS** | Strict `response_format` conformance |
| 7 | Prefix / KV Cache Reuse | **PASS (ACTIVE)** | Cold 5.11 s → warm 2.55 s (**2.0×**, 6,592 tok cached) |
| 8 | Client Socket Abort | **PASS** | Slot recovery 962 ms |
| 9 | Stop Words & Greedy Sampling | **PASS** | Deterministic temp=0 · stops suppressed |
| 10 | Precision Ledger Reconcile | **PASS** | Exact decimal match · 58.1 tok/s |
| 11 | API Error Protocol | **PASS** | HTTP 400/422 envelope compliant |
| 12 | Context Scaling (4k–418.5k) | **PASS** | 8/8 milestones RECALLED (see below) |
| 13 | Adversarial Graph & Distractors | **PASS** | 32k multi-hop route · 49.6 tok/s |
| 14 | Novel Algorithm (5k Fuzz) | **PASS** | 5,000 ops fuzzed, zero crashes |
| 15 | Frontier Multi-Needle | **PASS** | 0.5/50/99.5% depth checksum OK |
| 16 | CruxEval (Mental Code Exec) | **PASS** | 2/2 · 57.6 tok/s |
| 17 | SWE-bench (Traceback Fix) | **PASS** | Regression tests pass · 43.6 tok/s |
| 18 | AIME Olympiad Math | **PASS** | 3/3 · 54.2 tok/s |
| 19 | GPQA Diamond Science | **PASS** | 3/3 · 52.0 tok/s |
| 20 | Surgical Unified Diff | **PASS** | `git apply --check` OK · 51.7 tok/s |
| 21 | Multi-File Cross-Ref & DI | **PASS** | 3/3 modules aligned · 51.6 tok/s |
| 22 | Security Review & OWASP | **PASS** | SQLi + race detected · 0 false positives |
| 23 | Unit Test Synthesis | **PASS** | 3/3 mock tests · 52.1 tok/s |
| 24 | Strict Type Compliance | **PASS** | 100% annotated · 46.0 tok/s |
| 25 | Safe Refactor Invariants | **PASS** | Regression suite green · 47.2 tok/s |
| 26 | Prompt Injection Defense | **PASS** | Invariants protected · 43.8 tok/s |
| 27 | Google IFEval Writing | **PASS** | 4/5 strict prompts · 27.6 tok/s |
| 28 | Clinical & Medical (MedQA/PubMedQA) | **PASS** | **10/10** · 44.2 tok/s |
| 29 | Multi-Step Math (GSM8K) | **PASS** | 5/5 · 47.7 tok/s |
| 30 | Historical Causality (MMLU) | **PASS** | 5/5 · 40.5 tok/s |
| 31 | Multi-Adapter Isolation | **SKIPPED** | Single-model endpoint — excluded from grading |

---

## 📈 Long-Context Scaling (Test 12 — 512K YaRN 2.0 window)

Cold prefill (`cached = 0`), 100% needle recall at every depth:

| Target | Actual Prompt | TTFT | Prefill (tok/s) | Decode (tok/s) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4,000 | 4,062 | 1.61 s | 2,523 | 41.0 | RECALLED |
| 8,000 | 8,062 | 3.36 s | 2,403 | 49.4 | RECALLED |
| 16,000 | 16,076 | 6.23 s | 2,582 | 42.9 | RECALLED |
| 32,000 | 32,096 | 12.32 s | 2,605 | 45.2 | RECALLED |
| 64,000 | 64,088 | 24.50 s | **2,616** (peak) | 47.4 | RECALLED |
| 128,000 | 128,084 | 49.45 s | 2,590 | 47.5 | RECALLED |
| 256,000 | 256,067 | 102.71 s | 2,493 | 47.6 | RECALLED |
| 418,400 | 418,472 | 174.82 s | 2,394 | 45.3 | RECALLED |

---

## 🏢 Workload Suitability (auto-generated)

| Domain | Score | Rating |
|:---|:---:|:---:|
| Autonomous Agents & Tool Execution | 100% | **EXCELLENT** |
| Clinical & Medical Life-Sciences Research | 100% | **EXCELLENT** |
| Verifiable Instruction & Structured Writing | 100% | **EXCELLENT** |
| Deep Reasoning, Science & Math | 100% | **EXCELLENT** |
| Historical Causality & Humanities Research | 100% | **EXCELLENT** |
| Long-Context Retrieval & RAG Pipelines | 100% | **EXCELLENT** |
| Multimodal Vision & Document OCR | 100% | **EXCELLENT** |
| High-Throughput Serving & Low-Latency APIs | 100% | **EXCELLENT** |
| Software Engineering & Code Intelligence | 100% | **EXCELLENT** |

**All 9 enterprise domains qualify at 100% EXCELLENT / Highly Recommended.**

---

## 🔁 Run-to-Run Comparison (same DGX Spark deployment)

| Run | Build | Suite | Accuracy | Pure Cap | Grade | Notes |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| 2026-11-05 03:34 UTC | `dev780` (pre-sync) | 30-stage | 29/30 (98.3%) | 97.9% | A+ | Medical 6/10 (harness budget bug) |
| 2026-11-05 12:01 UTC | `dev801` (27-commit sync) | 31-stage | 29/30 (98.3%) | 97.9% | A+ | Medical fixed 10/10; security false-positive |
| **2026-11-05 13:43 UTC** | **`dev848` (46-commit sync)** | **31-stage** | **30/30 (100%)** | **100%** | **A+** | **Perfect run; security clean again** |

The 46-commit sync (incl. the QSA checkpoint FP8-KV-scale fix #65812) shows no throughput regression:
peak prefill 2,616 vs 2,625 tok/s, single-stream probes 62.9–63.9 tok/s @ 96.9–99.2% MTP acceptance
(unchanged within run-to-run noise).

---

## 📎 Artifacts

- **Machine-readable report:** [`enterprise_eval_127.0.0.1_8080_20261105_141137.json`](./enterprise_eval_127.0.0.1_8080_20261105_141137.json)
- **Deployment recipe:** [`DGX-Spark-Recipes/vllm`](https://github.com/cnsiva/DGX-Spark-Recipes) — profile `qwen3.8-flash-next-fp4.conf`
- **Reproduce:**
  ```bash
  docker run --rm --network host \
    -v $(pwd)/results:/app/results -v $(pwd)/logs:/app/logs \
    llm-eval-suite:latest -e http://127.0.0.1:8080/v1 \
    -m Qwen3.8-Flash-Next-FP4 -s all
  ```
