#!/usr/bin/env python3
"""
Benchmark suite comparing LLM inference backends (e.g. ExLlamaV3 vs llama.cpp, vLLM vs SGLang).
Supports:
1. Native Arena Mode (--live): Live concurrent comparison between two endpoints side-by-side in real time.
2. Single-Endpoint Benchmark: Run standardized tasks against a single backend.
3. Offline Report Comparison: Compare two or more saved JSON benchmark reports head-to-head.

Measures:
1. TTFT (Time To First Token) / Prefill Latency
2. Decode throughput (tok/s)
3. Total generation time & token count
4. Coding accuracy / syntax validation
5. Concurrency bug diagnosis
6. High-throughput system architecture quality
7. Long-context hidden constraint retrieval
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error
import argparse
import threading
import re

# ANSI Color & Formatting Constants
NO_COLOR = bool(os.getenv("NO_COLOR")) or not sys.stdout.isatty()
CYAN = "" if NO_COLOR else "\033[96m"
GREEN = "" if NO_COLOR else "\033[92m"
YELLOW = "" if NO_COLOR else "\033[93m"
RED = "" if NO_COLOR else "\033[91m"
MAGENTA = "" if NO_COLOR else "\033[95m"
BOLD = "" if NO_COLOR else "\033[1m"
DIM = "" if NO_COLOR else "\033[2m"
RESET = "" if NO_COLOR else "\033[0m"


BENCHMARK_TASKS = [
    {
        "id": "task1_avl_tree",
        "category": "Coding & Algorithms",
        "name": "AVL Tree with Rotations & In-Order Iterator",
        "system": "You are an expert systems software engineer. Write clean, production-grade, bug-free Python code.",
        "prompt": (
            "Write a complete, production-grade self-balancing AVL Tree in Python with complete type annotations.\n"
            "Requirements:\n"
            "1. Implement `AVLNode[T]` and `AVLTree[T]` with support for `insert(val: T) -> None`, `delete(val: T) -> bool`, and `search(val: T) -> bool`.\n"
            "2. Implement left rotation, right rotation, and double rotations correctly with node height recalculations.\n"
            "3. Implement an iterator `__iter__` that performs in-order traversal yielding values in sorted order.\n"
            "4. Include a self-contained unit test function `test_avl()` that inserts [50, 25, 75, 10, 30, 60, 80, 5, 15, 27, 55], asserts sorted traversal, deletes several nodes (including root and leaf), and verifies tree invariants."
        ),
        "max_tokens": 1500,
        "temperature": 0.2,
    },
    {
        "id": "task2_concurrency_debug",
        "category": "Code Debugging & Reasoning",
        "name": "Concurrent Bounded Buffer Bug Diagnosis",
        "system": "You are a senior concurrency and distributed systems engineer. Be precise, identify exact root causes, and provide corrected code.",
        "prompt": (
            "Analyze the following Python multi-threaded bounded buffer implementation. Identify all concurrency bugs, race conditions, and edge-case failures, explain why they occur, and provide the fully corrected, thread-safe implementation:\n\n"
            "```python\n"
            "import threading\n"
            "import time\n\n"
            "class BuggyBoundedBuffer:\n"
            "    def __init__(self, capacity):\n"
            "        self.capacity = capacity\n"
            "        self.buffer = []\n"
            "        self.lock = threading.Lock()\n"
            "        self.not_full = threading.Condition(self.lock)\n"
            "        self.not_empty = threading.Condition(self.lock)\n"
            "        self.processed_items = set()\n\n"
            "    def put(self, item):\n"
            "        self.lock.acquire()\n"
            "        if len(self.buffer) >= self.capacity:\n"
            "            self.not_full.wait()\n"
            "        self.buffer.append(item)\n"
            "        self.processed_items.add(item)\n"
            "        self.not_empty.signal()\n"
            "        self.lock.release()\n\n"
            "    def get(self):\n"
            "        self.lock.acquire()\n"
            "        if len(self.buffer) == 0:\n"
            "            self.not_empty.wait()\n"
            "        item = self.buffer.pop(0)\n"
            "        self.not_full.signal()\n"
            "        self.lock.release()\n"
            "        return item\n"
            "```\n"
        ),
        "max_tokens": 1200,
        "temperature": 0.2,
    },
    {
        "id": "task3_system_design",
        "category": "System Architecture & Design",
        "name": "High-Throughput Distributed Rate Limiter",
        "system": "You are a Principal Infrastructure Architect at a global scale technology company.",
        "prompt": (
            "Write a detailed technical architecture and design document for a Global Distributed Rate Limiter & Abuse Prevention Service.\n"
            "Scale & Requirements:\n"
            "- 500,000 requests/second globally across 4 regions.\n"
            "- P99 latency overhead < 3ms on the critical request path.\n"
            "- Consistent policy enforcement across multi-tenant API clients.\n\n"
            "Structure your document with:\n"
            "1. Architectural Overview & Component Topology (ASCII/Mermaid diagram).\n"
            "2. Algorithm Comparison & Selection (Token Bucket vs Leaky Bucket vs Sliding Window Counter with Redis/Aerospike/local memory).\n"
            "3. Synchronization & Global State Strategy (dealing with regional replication lag, eventual consistency vs strict limits).\n"
            "4. Resiliency & Failure Modes (how the system behaves during network partition between regions or datastore degradation).\n"
            "5. Concrete Redis Lua script or Go/Rust data structure for atomic rate evaluation."
        ),
        "max_tokens": 1800,
        "temperature": 0.5,
    },
    {
        "id": "task4_long_context_constraints",
        "category": "Long Context & Retrieval",
        "name": "Hidden Architectural Constraints Adherence",
        "system": "You are a senior backend engineer implementing client libraries strictly according to specifications.",
        "prompt": (
            "You are implementing an enterprise payment gateway client. Below is a background document describing company standards.\n\n"
            + ("# Section: Enterprise Architectural Standards and Policies\n"
               "This corporate standard defines logging, metrics, error wrapping, and connection pooling rules.\n"
               "All microservices communicating over gRPC or HTTPS must adhere to the standard definitions.\n"
               "The platform team continuously monitors adherence to these standards via telemetry probes.\n\n" * 25)
            + "\n--- CRITICAL HIDDEN SPECIFICATION 1 ---\n"
            + "MANDATORY DATABASE AUDIT TAG: Whenever initializing database or downstream RPC sessions, the connection header `x-sec-audit-tag` MUST be set strictly to the static constant `SEC_AUDIT_PROD_9981`. Any other value will trigger security revocation.\n\n"
            + ("# Section: Data Pipeline Standards\n"
               "Standard ETL patterns require idempotency keys and partitioned Kafka topics.\n"
               "Message deduplication windows are fixed at 3600 seconds.\n\n" * 25)
            + "\n--- CRITICAL HIDDEN SPECIFICATION 2 ---\n"
            + "MANDATORY RETRY BACKOFF FORMULA: Retries must NEVER use pure exponential backoff. The delay `d` in seconds must strictly follow the formula: `d = min(15.0, 0.25 * (2 ** attempt) + random.uniform(-0.05, 0.05))`.\n\n"
            + ("# Section: Observability and Error Propagation\n"
               "Distributed tracing spans must be passed across HTTP headers.\n"
               "Tracing identifiers follow OpenTelemetry standard baggage.\n\n" * 25)
            + "\n--- CRITICAL HIDDEN SPECIFICATION 3 ---\n"
            + "MANDATORY ERROR ENVELOPE: All exception responses must return a dictionary with exact keys `error_code`, `message`, and `x_custom_trace_id` where `x_custom_trace_id` is generated via `uuid.uuid5(uuid.NAMESPACE_DNS, f'error.{error_code}')`.\n\n"
            + "TASK: Now, write a Python class `PaymentGatewayClient` that implements an `execute_payment(user_id: str, amount_cents: int) -> dict` method honoring all 3 critical hidden specifications above (Audit Tag, Retry Formula, and Error Envelope). Highlight where each constraint is implemented."
        ),
        "max_tokens": 1200,
        "temperature": 0.2,
    }
]


def discover_model(endpoint: str, api_key: str = "", specified_model: str = None) -> str:
    """Discovers available model ID from endpoint /models, or falls back to specified or default."""
    if specified_model and specified_model != "default":
        return specified_model
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        models_url = f"{base}/models"
    else:
        models_url = f"{base}/v1/models" if "/v1" not in base else f"{base}/models"
    try:
        req = urllib.request.Request(models_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data if isinstance(data, list) else (data.get("data") or data.get("models") or [])
            if models and isinstance(models, list):
                m0 = models[0]
                m_id = m0.get("id") or m0.get("name")
                if m_id:
                    return m_id
    except Exception:
        pass
    return specified_model or "default"


def evaluate_arena_task(task_id: str, full_output: str) -> dict:
    """Evaluates task execution correctness, constraints, and syntax adherence."""
    details = {}
    if task_id == "task1_avl_tree":
        has_avl = "class AVLTree" in full_output or "class AvlTree" in full_output
        has_rot = "rotate" in full_output.lower()
        has_t = "def test_avl" in full_output or "test_avl()" in full_output
        details = {
            "has_avl_class": has_avl,
            "has_rotations": has_rot,
            "has_unit_test": has_t,
        }
        if has_avl and has_rot and has_t:
            status, score = "PASS", 1.0
        elif has_avl or has_rot:
            status, score = "PARTIAL", 0.5
        else:
            status, score = "FAIL", 0.0

    elif task_id == "task2_concurrency_debug":
        spurious = "while" in full_output and ("wait" in full_output or "spurious" in full_output.lower())
        lock_ctx = "with self.lock" in full_output or "try:" in full_output or "acquire" in full_output
        mem_leak = "processed_items" in full_output
        details = {
            "found_spurious_wakeup": spurious,
            "found_lock_context": lock_ctx,
            "found_memory_leak": mem_leak,
        }
        if spurious and (lock_ctx or mem_leak):
            status, score = "PASS", 1.0
        elif spurious or lock_ctx:
            status, score = "PARTIAL", 0.5
        else:
            status, score = "FAIL", 0.0

    elif task_id == "task3_system_design":
        lower = full_output.lower()
        algo = any(x in lower for x in ["token bucket", "leaky bucket", "sliding window", "rate limit"])
        storage = any(x in lower for x in ["redis", "lua", "aerospike", "in-memory", "atomic"])
        resilience = any(x in lower for x in ["partition", "resilience", "fallback", "degradation", "consistency", "replication"])
        length_ok = len(full_output.split()) >= 200
        details = {
            "algorithm_analysis": algo,
            "storage_strategy": storage,
            "resiliency_strategy": resilience,
            "sufficient_depth": length_ok,
        }
        passed_cnt = sum([algo, storage, resilience, length_ok])
        if passed_cnt >= 3:
            status, score = "PASS", 1.0
        elif passed_cnt >= 2:
            status, score = "PARTIAL", 0.5
        else:
            status, score = "FAIL", 0.0

    elif task_id == "task4_long_context_constraints":
        spec1 = "SEC_AUDIT_PROD_9981" in full_output
        spec2 = ("0.25 * (2 ** attempt)" in full_output or "0.25 *(2**attempt)" in full_output
                 or "random.uniform(-0.05, 0.05)" in full_output or ("2 ** attempt" in full_output and "0.25" in full_output))
        spec3 = "uuid.uuid5" in full_output and "x_custom_trace_id" in full_output
        details = {
            "spec1_audit_tag": spec1,
            "spec2_retry_formula": spec2,
            "spec3_uuid5_envelope": spec3,
        }
        if spec1 and spec2 and spec3:
            status, score = "PASS", 1.0
        elif sum([spec1, spec2, spec3]) >= 2:
            status, score = "PARTIAL", 0.5
        else:
            status, score = "FAIL", 0.0
    else:
        status, score = ("PASS", 1.0) if len(full_output) > 100 else ("FAIL", 0.0)

    return {"status": status, "score": score, "details": details}


def compute_task_advantage(res1: dict, res2: dict, name1: str, name2: str) -> str:
    """Calculates relative advantage string between two task runs."""
    st1 = res1.get("status", "FAIL")
    st2 = res2.get("status", "FAIL")
    spd1 = res1.get("tok_per_sec", 0.0)
    spd2 = res2.get("tok_per_sec", 0.0)
    ttft1 = res1.get("ttft_ms", 0.0)
    ttft2 = res2.get("ttft_ms", 0.0)
    t1 = res1.get("tokens", 0)
    t2 = res2.get("tokens", 0)

    # 1. Correctness precedence
    if "PASS" in st1 and "PASS" not in st2:
        return f"{name1} Victory (Correctness)"
    if "PASS" in st2 and "PASS" not in st1:
        return f"{name2} Victory (Correctness)"
    if "PARTIAL" in st1 and "FAIL" in st2:
        return f"{name1} Victory (Partial vs Fail)"
    if "PARTIAL" in st2 and "FAIL" in st1:
        return f"{name2} Victory (Partial vs Fail)"

    # Both passed or both failed: throughput and latency comparison
    advantages = []
    if spd1 > 0 and spd2 > 0:
        if spd1 > spd2 * 1.05:
            diff = ((spd1 - spd2) / spd2) * 100
            advantages.append(f"{name1} (+{diff:.1f}% tok/s)")
        elif spd2 > spd1 * 1.05:
            diff = ((spd2 - spd1) / spd1) * 100
            advantages.append(f"{name2} (+{diff:.1f}% tok/s)")

    if ttft1 > 0 and ttft2 > 0:
        if ttft1 < ttft2 * 0.90:
            ratio = ttft2 / ttft1
            advantages.append(f"{name1} ({ratio:.1f}x lower TTFT)")
        elif ttft2 < ttft1 * 0.90:
            ratio = ttft1 / ttft2
            advantages.append(f"{name2} ({ratio:.1f}x lower TTFT)")

    if not advantages:
        if t1 > 0 and t2 > 0 and abs(t1 - t2) / max(t1, t2) > 0.15:
            if t1 < t2:
                advantages.append(f"{name1} ({t2 - t1} fewer tokens)")
            else:
                advantages.append(f"{name2} ({t1 - t2} fewer tokens)")
        else:
            return "Tied"

    return ", ".join(advantages)


def _stream_worker(endpoint: str, model: str, api_key: str, task: dict, state: dict):
    """Worker thread that executes a single streaming completion request and updates state."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": task["prompt"]},
        ],
        "max_tokens": task["max_tokens"],
        "temperature": task["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)

    t_start = time.perf_counter()
    state["t_start"] = t_start
    state["status"] = "CONNECTING"

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            state["status"] = "STREAMING"
            for line in response:
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str.startswith("data:"):
                    continue
                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue

                if "error" in chunk:
                    state["error"] = str(chunk["error"])
                    state["status"] = "ERROR"
                    break

                if "usage" in chunk and chunk["usage"]:
                    state["completion_tokens"] = chunk["usage"].get("completion_tokens", state["completion_tokens"])

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                if choices[0].get("finish_reason") == "error":
                    err_msg = choices[0].get("message", {}).get("content", "Server error during generation")
                    state["error"] = err_msg
                    state["status"] = "ERROR"
                    break

                delta = choices[0].get("delta", {})
                text_piece = delta.get("content") or delta.get("reasoning_content") or ""
                if text_piece:
                    now = time.perf_counter()
                    if state["t_first"] is None:
                        state["t_first"] = now
                        state["ttft_ms"] = (now - t_start) * 1000.0
                    state["chunks"].append(text_piece)
                    state["tokens"] += 1
                    elapsed = now - state["t_first"]
                    if elapsed > 0 and state["tokens"] > 1:
                        state["tok_s"] = (state["tokens"] - 1) / elapsed
                    elif elapsed > 0:
                        state["tok_s"] = state["tokens"] / elapsed

        t_end = time.perf_counter()
        state["t_end"] = t_end
        total_time = t_end - t_start
        state["total_time_s"] = total_time
        final_tokens = state["completion_tokens"] if state["completion_tokens"] > 0 else state["tokens"]
        gen_time = (t_end - state["t_first"]) if state["t_first"] else total_time
        final_speed = ((final_tokens - 1) / gen_time) if gen_time > 0 and final_tokens > 1 else (final_tokens / gen_time if gen_time > 0 else 0.0)
        state["tokens"] = final_tokens
        state["tok_s"] = round(final_speed, 2)
        if state["ttft_ms"] is None:
            state["ttft_ms"] = round(total_time * 1000.0, 1)
        if state["status"] != "ERROR":
            state["status"] = "DONE"

    except Exception as e:
        t_end = time.perf_counter()
        state["t_end"] = t_end
        state["total_time_s"] = t_end - t_start
        state["error"] = str(e)
        state["status"] = "ERROR"
        if state["ttft_ms"] is None:
            state["ttft_ms"] = 0.0


def run_live_arena(
    endpoint1: str,
    model1: str = None,
    endpoint2: str = None,
    model2: str = None,
    api_key1: str = "",
    api_key2: str = "",
    output_file: str = None,
    tasks: list = None,
):
    """
    Executes Native Arena Mode:
    1. Concurrently connects to both endpoints in parallel threads.
    2. Sends each test prompt to both servers simultaneously.
    3. Streams live token speeds and TTFT from both servers side-by-side in real time.
    4. Automatically calculates the relative advantage and saves the Markdown comparison report.
    """
    ep1 = endpoint1.rstrip("/")
    ep2 = endpoint2.rstrip("/") if endpoint2 else ""
    if not ep2:
        print(f"{RED}Error: Native Arena Mode requires two endpoints (--endpoint1 and --endpoint2){RESET}")
        sys.exit(1)

    # 1. Discover or confirm models
    m1 = discover_model(ep1, api_key1, model1)
    m2 = discover_model(ep2, api_key2, model2)

    active_tasks = tasks or BENCHMARK_TASKS

    banner_w = 98
    print("\n" + "=" * banner_w)
    print(f" {BOLD}{CYAN}LLM ARENA MODE: CONCURRENT REAL-TIME HEAD-TO-HEAD BENCHMARK{RESET} ".center(banner_w + (len(BOLD+CYAN+RESET) if not NO_COLOR else 0)))
    print("=" * banner_w)
    print(f"  • Endpoint 1 : {ep1}  (Model: {BOLD}{m1}{RESET})")
    print(f"  • Endpoint 2 : {ep2}  (Model: {BOLD}{m2}{RESET})")
    print(f"  • Mode       : {GREEN}Live Parallel Streaming Ticker (Simultaneous Dispatch){RESET}")
    print(f"  • Total Tasks: {len(active_tasks)} Architectural & Reasoning Tasks")
    print("=" * banner_w)

    arena_results = []
    t_arena_start = time.perf_counter()

    for idx, task in enumerate(active_tasks, 1):
        print(f"\n[{idx}/{len(active_tasks)}] {BOLD}{task['name']}{RESET} ({DIM}{task['category']}{RESET})")
        print(f"  {CYAN}⚡ Dispatching prompt simultaneously to both endpoints...{RESET}")

        state1 = {
            "endpoint": ep1,
            "model": m1,
            "tokens": 0,
            "tok_s": 0.0,
            "ttft_ms": None,
            "status": "CONNECTING",
            "chunks": [],
            "error": None,
            "t_start": None,
            "t_first": None,
            "t_end": None,
            "completion_tokens": 0,
            "total_time_s": 0.0,
        }
        state2 = {
            "endpoint": ep2,
            "model": m2,
            "tokens": 0,
            "tok_s": 0.0,
            "ttft_ms": None,
            "status": "CONNECTING",
            "chunks": [],
            "error": None,
            "t_start": None,
            "t_first": None,
            "t_end": None,
            "completion_tokens": 0,
            "total_time_s": 0.0,
        }

        t1 = threading.Thread(target=_stream_worker, args=(ep1, m1, api_key1, task, state1), daemon=True)
        t2 = threading.Thread(target=_stream_worker, args=(ep2, m2, api_key2, task, state2), daemon=True)

        t1.start()
        t2.start()

        # Live side-by-side streaming monitor ticker
        last_non_tty_print = 0.0
        while t1.is_alive() or t2.is_alive():
            time.sleep(0.08)
            now = time.perf_counter()

            # Format Endpoint 1 status
            if state1["status"] == "CONNECTING":
                s1_str = f"M1: Connecting..."
            elif state1["status"] == "STREAMING":
                ttft_part = f"TTFT {state1['ttft_ms']:.0f}ms" if state1['ttft_ms'] is not None else "TTFT ..."
                s1_str = f"M1: {state1['tokens']:>4}t │ {state1['tok_s']:>5.1f}t/s │ {ttft_part}"
            elif state1["status"] == "DONE":
                s1_str = f"M1: DONE ({state1['tokens']}t │ {state1['tok_s']:.1f}t/s)"
            else:
                s1_str = f"M1: ERROR"

            # Format Endpoint 2 status
            if state2["status"] == "CONNECTING":
                s2_str = f"M2: Connecting..."
            elif state2["status"] == "STREAMING":
                ttft_part = f"TTFT {state2['ttft_ms']:.0f}ms" if state2['ttft_ms'] is not None else "TTFT ..."
                s2_str = f"M2: {state2['tokens']:>4}t │ {state2['tok_s']:>5.1f}t/s │ {ttft_part}"
            elif state2["status"] == "DONE":
                s2_str = f"M2: DONE ({state2['tokens']}t │ {state2['tok_s']:.1f}t/s)"
            else:
                s2_str = f"M2: ERROR"

            if sys.stdout.isatty():
                sys.stdout.write(f"\r  {CYAN}{s1_str:<42}{RESET} ⚔️   {MAGENTA}{s2_str:<42}{RESET}\033[K")
                sys.stdout.flush()
            else:
                if now - last_non_tty_print >= 2.5:
                    print(f"  [Arena Live] {s1_str}  vs  {s2_str}")
                    last_non_tty_print = now

        t1.join()
        t2.join()

        if sys.stdout.isatty():
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        # Evaluate outputs
        out1 = "".join(state1["chunks"])
        out2 = "".join(state2["chunks"])

        eval1 = evaluate_arena_task(task["id"], out1) if not state1["error"] else {"status": "FAIL", "score": 0.0, "details": {"error": state1["error"]}}
        eval2 = evaluate_arena_task(task["id"], out2) if not state2["error"] else {"status": "FAIL", "score": 0.0, "details": {"error": state2["error"]}}

        res1_task = {
            "model": m1,
            "endpoint": ep1,
            "tokens": state1["tokens"],
            "total_time_s": round(state1["total_time_s"], 3),
            "ttft_ms": round(state1["ttft_ms"] or 0.0, 1),
            "tok_per_sec": round(state1["tok_s"], 2),
            "status": eval1["status"],
            "score": eval1["score"],
            "eval_details": eval1["details"],
            "error": state1["error"],
            "output_sample": out1[:350] + "..." if len(out1) > 350 else out1,
            "full_output": out1,
        }
        res2_task = {
            "model": m2,
            "endpoint": ep2,
            "tokens": state2["tokens"],
            "total_time_s": round(state2["total_time_s"], 3),
            "ttft_ms": round(state2["ttft_ms"] or 0.0, 1),
            "tok_per_sec": round(state2["tok_s"], 2),
            "status": eval2["status"],
            "score": eval2["score"],
            "eval_details": eval2["details"],
            "error": state2["error"],
            "output_sample": out2[:350] + "..." if len(out2) > 350 else out2,
            "full_output": out2,
        }

        adv_str = compute_task_advantage(res1_task, res2_task, f"Model 1 ({m1[:12]})", f"Model 2 ({m2[:12]})")

        arena_task_record = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "model1": res1_task,
            "model2": res2_task,
            "advantage": adv_str,
        }
        arena_results.append(arena_task_record)

        # Print formatted task completion card
        box_w = 98
        print(f"  ┌{'─' * (box_w - 4)}┐")
        st1_color = GREEN if "PASS" in eval1["status"] else (YELLOW if "PARTIAL" in eval1["status"] else RED)
        st2_color = GREEN if "PASS" in eval2["status"] else (YELLOW if "PARTIAL" in eval2["status"] else RED)
        line1 = f"  │ Model 1 ({m1[:16]}): {state1['tokens']:>5,} toks │ {state1['total_time_s']:>5.2f}s │ TTFT: {state1['ttft_ms']:>6.1f} ms │ {state1['tok_s']:>6.2f} tok/s │ {st1_color}{eval1['status']:<7}{RESET} │"
        line2 = f"  │ Model 2 ({m2[:16]}): {state2['tokens']:>5,} toks │ {state2['total_time_s']:>5.2f}s │ TTFT: {state2['ttft_ms']:>6.1f} ms │ {state2['tok_s']:>6.2f} tok/s │ {st2_color}{eval2['status']:<7}{RESET} │"
        line_adv = f"  │ Advantage: {BOLD}{adv_str}{RESET}"
        pad_adv = box_w - len(f"  │ Advantage: {adv_str}") - 2
        line_adv_full = f"{line_adv}{' ' * max(0, pad_adv)}│"

        print(line1)
        print(line2)
        print(f"  ├{'─' * (box_w - 4)}┤")
        print(line_adv_full)
        print(f"  └{'─' * (box_w - 4)}┘")

    total_arena_time = time.perf_counter() - t_arena_start

    # Executive Summaries
    def compute_model_summary(key: str, model_name: str, ep: str):
        total_eval = len(arena_results)
        passed = sum(1 for r in arena_results if r[key]["status"] == "PASS")
        partial = sum(1 for r in arena_results if r[key]["status"] == "PARTIAL")
        failed = sum(1 for r in arena_results if r[key]["status"] not in ("PASS", "PARTIAL"))
        eff_pct = ((passed + 0.5 * partial) / total_eval * 100.0) if total_eval > 0 else 0.0

        tot_tokens = sum(r[key]["tokens"] for r in arena_results)
        passed_toks = sum(r[key]["tokens"] for r in arena_results if r[key]["status"] == "PASS") + \
                      sum(int(r[key]["tokens"] * 0.5) for r in arena_results if r[key]["status"] == "PARTIAL")
        tok_economy = (passed_toks / (passed + 0.5 * partial)) if (passed + partial) > 0 else 0.0

        speeds = [r[key]["tok_per_sec"] for r in arena_results if r[key]["tok_per_sec"] > 0]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        ttfts = [r[key]["ttft_ms"] for r in arena_results if r[key]["ttft_ms"] > 0]
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

        wall_time = sum(r[key]["total_time_s"] for r in arena_results)

        token_conciseness_factor = min(2.0, max(0.2, 1000.0 / (tok_economy if tok_economy > 0 else 1000.0)))
        eff_index = round((eff_pct * 0.6) + (min(100.0, avg_speed * 1.5) * 0.25) + (token_conciseness_factor * 15.0), 1)

        return {
            "model": model_name,
            "endpoint": ep,
            "total_evaluated": total_eval,
            "passed": passed,
            "partial": partial,
            "failed": failed,
            "effectiveness_rate_pct": round(eff_pct, 1),
            "total_tokens_emitted": tot_tokens,
            "token_economy_tokens_per_passed_task": round(tok_economy, 1),
            "avg_decode_tok_s": round(avg_speed, 2),
            "avg_ttft_ms": round(avg_ttft, 1),
            "total_wall_time_s": round(wall_time, 2),
            "efficiency_index": eff_index,
        }

    sum1 = compute_model_summary("model1", m1, ep1)
    sum2 = compute_model_summary("model2", m2, ep2)

    # Relative advantage calculations for Executive Summary
    from eval import compute_relative_advantages
    adv = compute_relative_advantages(sum1, sum2)
    verdict = adv["verdict"]

    eff_a, eff_b = sum1["effectiveness_rate_pct"], sum2["effectiveness_rate_pct"]
    te_a, te_b = sum1["token_economy_tokens_per_passed_task"], sum2["token_economy_tokens_per_passed_task"]
    tot_a, tot_b = sum1["total_tokens_emitted"], sum2["total_tokens_emitted"]
    spd_a, spd_b = sum1["avg_decode_tok_s"], sum2["avg_decode_tok_s"]
    ttft_a, ttft_b = sum1["avg_ttft_ms"], sum2["avg_ttft_ms"]
    wall_a, wall_b = sum1["total_wall_time_s"], sum2["total_wall_time_s"]
    ei_a, ei_b = sum1["efficiency_index"], sum2["efficiency_index"]

    # Print Executive CLI Report
    print("\n" + "=" * 100)
    print(f" {BOLD}{CYAN}CROSS-MODEL HEAD-TO-HEAD ARENA EVALUATION REPORT{RESET} ".center(100 + (len(BOLD+CYAN+RESET) if not NO_COLOR else 0)))
    print("=" * 100)
    print(f"  Model A : {m1} ({ep1})")
    print(f"  Model B : {m2} ({ep2})\n")

    exec_hdr = f"{'Core Performance Metric':<35} | {'Model A (' + str(m1)[:16] + ')':<26} | {'Model B (' + str(m2)[:16] + ')':<26} | {'Advantage'}"
    print(f"{BOLD}{exec_hdr}{RESET}")
    pass_a_str = f"{sum1['passed']}/{sum1['total_evaluated']} ({eff_a:.1f}%)"
    pass_b_str = f"{sum2['passed']}/{sum2['total_evaluated']} ({eff_b:.1f}%)"
    te_a_str = f"{te_a:.1f} tokens/task"
    te_b_str = f"{te_b:.1f} tokens/task"
    tot_a_str = f"{tot_a:,} tokens"
    tot_b_str = f"{tot_b:,} tokens"
    spd_a_str = f"{spd_a:.2f} tok/s"
    spd_b_str = f"{spd_b:.2f} tok/s"
    ttft_a_str = f"{ttft_a:.1f} ms"
    ttft_b_str = f"{ttft_b:.1f} ms"
    wall_a_str = f"{wall_a:.2f} s"
    wall_b_str = f"{wall_b:.2f} s"
    ei_a_str = f"{ei_a:.1f} / 100"
    ei_b_str = f"{ei_b:.1f} / 100"

    print(f"{'Effectiveness (Pass Rate)':<35} | {pass_a_str:<26} | {pass_b_str:<26} | {adv['adv_eff']}")
    print(f"{'Token Economy (Tokens/Victory)':<35} | {te_a_str:<26} | {te_b_str:<26} | {adv['adv_te']}")
    print(f"{'Total Solution Tokens Consumed':<35} | {tot_a_str:<26} | {tot_b_str:<26} | {adv['adv_tot']}")
    print(f"{'Mean Decode Speed (Throughput)':<35} | {spd_a_str:<26} | {spd_b_str:<26} | {adv['adv_spd']}")
    print(f"{'Mean Time-To-First-Token (TTFT)':<35} | {ttft_a_str:<26} | {ttft_b_str:<26} | {adv['adv_ttft']}")
    print(f"{'Total Generation Wall Time':<35} | {wall_a_str:<26} | {wall_b_str:<26} | {adv['adv_wall']}")
    print(f"{'Composite Efficiency Index':<35} | {ei_a_str:<26} | {ei_b_str:<26} | {adv['adv_ei']}")
    print("=" * 100)

    # Domain matrix CLI
    print(f"\n{BOLD}{CYAN}ARENA TASK BREAKDOWN MATRIX{RESET}")
    print("-" * 100)
    dom_hdr = f"{'Task':<35} | {'Model A (' + str(m1)[:14] + ')':<26} | {'Model B (' + str(m2)[:14] + ')':<26} | {'Outcome / Advantage'}"
    print(f"{BOLD}{dom_hdr}{RESET}")
    print("-" * 100)
    for idx, r in enumerate(arena_results, 1):
        lbl = f"{idx}. {r['name'][:30]}"
        m1_info = f"{r['model1']['status']} ({r['model1']['tokens']}t | {r['model1']['tok_per_sec']}t/s)"
        m2_info = f"{r['model2']['status']} ({r['model2']['tokens']}t | {r['model2']['tok_per_sec']}t/s)"
        print(f"{lbl:<35} | {m1_info:<26} | {m2_info:<26} | {r['advantage']}")
    print("=" * 100)

    print(f"\n{BOLD}{GREEN}EXECUTIVE VERDICT:{RESET} {BOLD}{verdict}{RESET}\n")

    # Generate Output Paths
    safe_a = re.sub(r"[^\w\-]", "_", str(m1))
    safe_b = re.sub(r"[^\w\-]", "_", str(m2))
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    if output_file:
        md_file = output_file if output_file.endswith(".md") else f"{output_file}.md"
        json_file = output_file if output_file.endswith(".json") else (output_file[:-3] + ".json" if output_file.endswith(".md") else f"{output_file}.json")
    else:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        md_file = os.path.join(results_dir, f"live_comparison_{safe_a}_vs_{safe_b}_{timestamp_str}.md")
        json_file = os.path.join(results_dir, f"live_comparison_{safe_a}_vs_{safe_b}_{timestamp_str}.json")

    # Ensure output dir exists
    out_dir = os.path.dirname(md_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. Write Markdown Report
    md_lines = [
        "# Native Arena Mode: Cross-Model Head-to-Head Benchmark Report",
        "",
        f"- **Date Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Execution Mode**: Native Arena Mode (Live Concurrent Streaming)",
        f"- **Model A**: `{m1}` (`{ep1}`)",
        f"- **Model B**: `{m2}` (`{ep2}`)",
        f"- **Total Suite Wall Time**: `{total_arena_time:.2f} s`",
        "",
        "## Executive Summary & Efficiency Index",
        "",
        f"| Metric / Dimension | Model A (`{m1}`) | Model B (`{m2}`) | Relative Advantage |",
        "|:---|:---|:---|:---|",
        f"| **Effectiveness (Accuracy)** | `{sum1['passed']}/{sum1['total_evaluated']} ({eff_a:.1f}%)` | `{sum2['passed']}/{sum2['total_evaluated']} ({eff_b:.1f}%)` | **{adv['adv_eff']}** |",
        f"| **Token Economy (Solution Conciseness)** | `{te_a:.1f} tokens/task` | `{te_b:.1f} tokens/task` | **{adv['adv_te']}** |",
        f"| **Total Solution Tokens Consumed** | `{tot_a:,} tokens` | `{tot_b:,} tokens` | **{adv['adv_tot']}** |",
        f"| **Mean Decode Throughput** | `{spd_a:.2f} tok/s` | `{spd_b:.2f} tok/s` | **{adv['adv_spd']}** |",
        f"| **Mean Time-to-First-Token (TTFT)** | `{ttft_a:.1f} ms` | `{ttft_b:.1f} ms` | **{adv['adv_ttft']}** |",
        f"| **Total Generation Wall Time** | `{wall_a:.2f} s` | `{wall_b:.2f} s` | **{adv['adv_wall']}** |",
        f"| **Composite Efficiency Index** | `{ei_a:.1f} / 100` | `{ei_b:.1f} / 100` | **{adv['adv_ei']}** |",
        "",
        "## Arena Task Comparison Matrix",
        "",
        f"| # | Benchmark Task | Category | Model A (`{m1}`) | Model B (`{m2}`) | Outcome / Advantage |",
        "|:---|:---|:---|:---|:---|:---|",
    ]

    for idx, r in enumerate(arena_results, 1):
        m1_desc = f"{r['model1']['status']} ({r['model1']['tokens']}t, {r['model1']['tok_per_sec']}t/s, {r['model1']['ttft_ms']}ms)"
        m2_desc = f"{r['model2']['status']} ({r['model2']['tokens']}t, {r['model2']['tok_per_sec']}t/s, {r['model2']['ttft_ms']}ms)"
        md_lines.append(f"| {idx} | {r['name']} | {r['category']} | {m1_desc} | {m2_desc} | **{r['advantage']}** |")

    md_lines.extend([
        "",
        "## Strategic Verdict & Deployment Recommendation",
        "",
        f"> **Executive Verdict**: {verdict}",
        "",
        "## Detailed Task Verification Checks",
        ""
    ])

    for idx, r in enumerate(arena_results, 1):
        md_lines.extend([
            f"### Task {idx}: {r['name']}",
            f"- **Category**: {r['category']}",
            f"- **Model A Checks**: `{json.dumps(r['model1']['eval_details'])}`",
            f"- **Model B Checks**: `{json.dumps(r['model2']['eval_details'])}`",
            f"- **Relative Task Advantage**: {r['advantage']}",
            ""
        ])

    with open(md_file, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # 2. Write Structured JSON Report
    json_summary = {
        "benchmark_mode": "native_arena_live",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_suite_wall_time_s": round(total_arena_time, 2),
        "endpoint1": ep1,
        "model1": m1,
        "endpoint2": ep2,
        "model2": m2,
        "executive_summary": {
            "model1": sum1,
            "model2": sum2,
            "relative_advantages": {
                "effectiveness": adv["adv_eff"],
                "token_economy": adv["adv_te"],
                "total_tokens": adv["adv_tot"],
                "decode_speed": adv["adv_spd"],
                "ttft": adv["adv_ttft"],
                "efficiency_index": adv["adv_ei"],
            },
            "verdict": verdict,
        },
        "tasks": arena_results,
    }

    with open(json_file, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"{GREEN}✓ Live Arena Benchmark Completed!{RESET}")
    print(f"  • Markdown Comparison Report : {BOLD}{md_file}{RESET}")
    print(f"  • Structured JSON Report     : {BOLD}{json_file}{RESET}\n")

    return json_summary


def run_benchmark(base_url: str, output_file: str, model: str = "qwen3.8-27b-exl3-3.0bpw", api_key: str = ""):
    """Legacy single-endpoint benchmark runner."""
    print(f"=== Starting Benchmark Suite against {base_url} (Model: {model}) ===")
    results = []

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Check models endpoint
    try:
        req = urllib.request.Request(f"{base_url}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            models_data = json.loads(resp.read().decode())
            print(f"Connected to endpoint. Models: {json.dumps(models_data)}")
    except Exception as e:
        print(f"Warning: Could not fetch models endpoint: {e}")

    for idx, task in enumerate(BENCHMARK_TASKS, 1):
        print(f"\n[{idx}/{len(BENCHMARK_TASKS)}] Running {task['id']} ({task['name']})...")
        state = {
            "tokens": 0,
            "tok_s": 0.0,
            "ttft_ms": None,
            "status": "CONNECTING",
            "chunks": [],
            "error": None,
            "t_start": None,
            "t_first": None,
            "t_end": None,
            "completion_tokens": 0,
            "total_time_s": 0.0,
        }
        _stream_worker(task, base_url, model, headers, state)

        if state["error"] and not state["chunks"]:
            print(f"  -> Error executing {task['id']}: {state['error']}")
            results.append({"task_id": task["id"], "error": state["error"], "status": "FAIL"})
            continue

        full_output = "".join(state["chunks"])
        final_tokens = state["tokens"]
        total_time = state["total_time_s"]
        ttft_ms = state["ttft_ms"] if state["ttft_ms"] is not None else 0.0
        tok_s = state["tok_s"]

        print(f"  -> Generated {final_tokens} tokens in {total_time:.2f}s")
        print(f"  -> TTFT (prefill latency): {ttft_ms:.1f} ms")
        print(f"  -> Decode Speed: {tok_s:.2f} tok/s")

        eval_res = evaluate_arena_task(task["id"], full_output)

        task_res = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "tokens": final_tokens,
            "total_time_s": round(total_time, 3),
            "ttft_ms": round(ttft_ms, 1),
            "tok_per_sec": round(tok_s, 2),
            "status": eval_res["status"],
            "score": eval_res["score"],
            "eval_details": eval_res["details"],
            "output_sample": full_output[:400] + "..." if len(full_output) > 400 else full_output,
            "full_output": full_output,
        }
        if state["error"]:
            task_res["error"] = state["error"]

        results.append(task_res)

    # Summary
    avg_speed = sum(r.get("tok_per_sec", 0) for r in results if "tok_per_sec" in r) / max(1, len([r for r in results if "tok_per_sec" in r]))
    summary = {
        "base_url": base_url,
        "model": model,
        "avg_decode_tok_s": round(avg_speed, 2),
        "results": results,
    }

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Benchmark completed! Saved to {output_file} (Avg: {avg_speed:.2f} tok/s) ===")
    return summary


def main():
    default_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1").rstrip("/")
    parser = argparse.ArgumentParser(description="LLM Inference Arena & Comparison Harness (Native Arena Mode or Offline Report Comparison)")

    # Native Arena Mode Flags
    parser.add_argument("--live", "--live-compare", action="store_true",
                        help="Enable Native Arena Mode: live concurrent comparison between two endpoints side-by-side in real time")
    parser.add_argument("--endpoint1", "--url1", default=None,
                        help="First LLM server base endpoint (e.g. http://127.0.0.1:8888/v1)")
    parser.add_argument("--model1", default=None, help="First model name or ID")
    parser.add_argument("--api-key1", default="", help="API key for endpoint 1 (optional)")
    parser.add_argument("--endpoint2", "--url2", default=None,
                        help="Second LLM server base endpoint (e.g. http://172.16.16.29:8000/v1)")
    parser.add_argument("--model2", default=None, help="Second model name or ID")
    parser.add_argument("--api-key2", default="", help="API key for endpoint 2 (optional)")

    # Single-endpoint benchmark flags
    parser.add_argument("--url", "--endpoint", default=default_url, help="Base API URL for single-endpoint benchmark")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "default", help="Model name or ID")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key for single-endpoint benchmark")

    # Output & Reports
    parser.add_argument("--out", "-o", default=None, help="Output path for Markdown comparison (.md) or JSON (.json)")
    parser.add_argument("--reports", nargs="+", help="Compare two or more JSON reports offline head-to-head")
    parser.add_argument("report_paths", nargs="*", help="JSON reports to compare offline head-to-head")

    args = parser.parse_args()
    reports = args.reports or args.report_paths

    # Route 1: Offline Report Comparison
    if reports:
        from eval import compare_benchmark_reports
        compare_benchmark_reports(reports, output_markdown=args.out if args.out and args.out.endswith(".md") else None)

    # Route 2: Native Arena Mode (Live Concurrent Comparison)
    elif args.live or (args.endpoint1 and args.endpoint2):
        ep1 = args.endpoint1 or args.url
        ep2 = args.endpoint2
        if not ep1 or not ep2:
            print(f"{RED}Error: Arena Mode requires both --endpoint1 and --endpoint2.{RESET}")
            sys.exit(1)
        run_live_arena(
            endpoint1=ep1,
            model1=args.model1,
            endpoint2=ep2,
            model2=args.model2,
            api_key1=args.api_key1,
            api_key2=args.api_key2,
            output_file=args.out,
        )

    # Route 3: Single-Endpoint Benchmark
    else:
        out_file = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "compare_benchmark.json")
        run_benchmark(args.url, out_file, model=args.model, api_key=args.api_key)


if __name__ == "__main__":
    main()
