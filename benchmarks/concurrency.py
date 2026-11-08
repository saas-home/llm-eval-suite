#!/usr/bin/env python3
"""
Multi-client concurrency, prefill & decode throughput stress test script.
Tests:
- Concurrency scaling across multiple client levels (e.g. 1, 2, 3, 4 clients).
- Prompt prefill and token decode throughput across context horizons (e.g. 2K, 4K, 8K, 16K, 32K tokens).
- Simultaneous synchronized client release using barriers to stress continuous batching.
- Latency (TTFT), prefill throughput (tok/s), decode throughput (tok/s), and aggregate throughput.
"""

import json
import time
import urllib.request
import threading
import argparse
import os
import sys
import uuid
import statistics

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from benchmarks.common import ChunkMetrics, aggregate_decode_rate  # noqa: E402
from client import LLMClient  # noqa: E402

_ep = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_API_URL = _ep if _ep.endswith("/chat/completions") else f"{_ep}/chat/completions"
DEFAULT_HEALTH_URL = f"{_ep[:-3]}/health" if _ep.endswith("/v1") else f"{_ep}/health"

SAMPLE_PROMPTS = [
    "Write a detailed Python function implementing merge sort with clear explanations and docstrings.",
    "Write a detailed Python function implementing quick sort with clear explanations and docstrings.",
    "Write a detailed Python class implementing a Binary Search Tree with insert and search methods.",
    "Write a detailed Python class implementing a Min-Heap priority queue with push and pop methods.",
    "Write a detailed Python function implementing Dijkstra's shortest path algorithm with explanations.",
    "Write a detailed Python function implementing breadth-first search on a directed graph with explanations.",
    "Write a detailed Python function implementing depth-first search with topological sort for a DAG.",
    "Write a detailed Python class implementing an LRU cache using a doubly linked list and hash map."
]

def check_health(health_url, api_key=""):
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(health_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def build_concurrency_prompt(target_tokens: int, client_id: int, total_clients: int) -> str:
    filler_para = (
        "Distributed computing clusters coordinate state using consensus protocols, replicated logs, "
        "and partitioned key-value indices. Efficient vector pipelines serialize operations to minimize "
        "inter-node bus contention and optimize high-throughput parallel execution paths. "
    )
    needed_tokens = max(100, target_tokens - 60)
    repeat_count = max(1, int(needed_tokens / 36))
    
    header = (
        f"[BENCHMARK CONCURRENCY MATRIX | Client: {client_id}/{total_clients} | "
        f"Target Context: {target_tokens} | Nonce: {uuid.uuid4().hex[:12]}]\n\n"
    )
    body = filler_para * repeat_count
    footer = (
        f"\n\n[Client {client_id} Query]: In one concise technical sentence, state the primary mechanism "
        f"for optimizing parallel execution throughput in distributed clusters."
    )
    return header + body + footer

def client_task(client_id, prompt, api_url, model, max_tokens, temperature, results, lock, barrier, api_key=""):
    chunks = ChunkMetrics()
    if barrier:
        barrier.wait()
    t_start = time.perf_counter()
    result = LLMClient(api_url, api_key=api_key, model=model).call(
        [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature,
        stream=True, timeout=300, chunk_callback=chunks,
    )
    t_end = time.perf_counter()
    ttft = result["ttft"] if not result.get("error") else 0.0
    prompt_tokens = result["prompt_tokens"] or int(len(prompt.split()) * 1.33)
    prefill_speed = (prompt_tokens / ttft) if ttft > 0 else 0.0

    with lock:
        results[client_id] = {
            "prompt": prompt[:40] + ("..." if len(prompt) > 40 else ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": result["completion_tokens"],
            "ttft_s": ttft,
            "total_time_s": result["total_time"],
            "decode_tok_s": result["decode_speed"],
            "prefill_tok_s": prefill_speed,
            "t_dispatch": t_start,
            "t_first": chunks.first if chunks.first is not None else (t_start + ttft),
            "t_last": chunks.last if chunks.last is not None else t_end,
            "error": result.get("error"),
            "failure_kind": result.get("failure_kind"),
            "endpoint_unreachable": result.get("endpoint_unreachable", False)
        }

def run_matrix_benchmark(clients, contexts, api_url, model, tokens, temperature, api_key="", server_down=None):
    table_header = (
        f"{'Context':>8} | {'Prompt Tok':>10} | {'Compl Tok':>9} | {'Mean TTFT (s)':>14} | "
        f"{'Avg Prefill (t/s)':>17} | {'Agg Prefill (t/s)':>17} | "
        f"{'Avg Decode (t/s)':>16} | {'Agg Decode (t/s)':>16} | {'Status':>8}"
    )

    has_errors = False
    for c in clients:
        label_suffix = " (Single-Client Baseline)" if c == 1 else f" ({c} Concurrent Parallel Streams)"
        print(f"\n[Client Concurrency Level: {c} Client{'s' if c > 1 else ''}]{label_suffix}")
        print(table_header)
        print("-" * len(table_header))

        for ctx in contexts:
            barrier = threading.Barrier(c)
            results = {}
            lock = threading.Lock()
            threads = []

            for i in range(c):
                p = build_concurrency_prompt(ctx, i + 1, c)
                t = threading.Thread(
                    target=client_task,
                    args=(i + 1, p, api_url, model, tokens, temperature, results, lock, barrier, api_key)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            errors = [r["error"] for r in results.values() if r.get("error")]
            if errors:
                has_errors = True
            valid_results = [r for r in results.values() if not r.get("error")]
            tot_prompt = sum(r["prompt_tokens"] for r in valid_results)
            tot_completion = sum(r["completion_tokens"] for r in valid_results)

            ttfts_s = [r["ttft_s"] for r in valid_results if r["ttft_s"] > 0]
            mean_ttft_s = statistics.mean(ttfts_s) if ttfts_s else 0.0

            # Prefill throughput: aggregate batch throughput and per-client average
            max_ttft_s = max((r["ttft_s"] for r in valid_results), default=0.0)
            agg_prefill = (tot_prompt / max_ttft_s) if max_ttft_s > 0 else 0.0
            prefill_speeds = [r["prefill_tok_s"] for r in valid_results if r["ttft_s"] > 0]
            avg_prefill = statistics.mean(prefill_speeds) if prefill_speeds else 0.0

            # Decode throughput: aggregate across concurrent streams and per-client average
            decode_speeds = [r["decode_tok_s"] for r in valid_results if r["decode_tok_s"] > 0]
            agg_decode = aggregate_decode_rate(valid_results, "t_first", "t_last")
            avg_decode = statistics.mean(decode_speeds) if decode_speeds else 0.0

            completed = len(valid_results) == c and all(r["completion_tokens"] > 0 for r in valid_results)
            status = "ERROR" if errors else ("PASS" if completed else "FAIL")
            if status != "PASS":
                has_errors = True
            row = (
                f"{ctx:>8,d} | {tot_prompt:>10,d} | {tot_completion:>9,d} | {mean_ttft_s:>14.2f} | "
                f"{avg_prefill:>17.1f} | {agg_prefill:>17.1f} | "
                f"{avg_decode:>16.1f} | {agg_decode:>16.1f} | {status:>8}"
            )
            print(row)
            for error in sorted(set(errors)):
                print(f"  Request error: {error}")
            if any(r.get("endpoint_unreachable") for r in results.values()):
                if server_down is not None:
                    server_down.set()
                return False
        print("-" * len(table_header))
    return not has_errors

def main():
    parser = argparse.ArgumentParser(description="Multi-client concurrency, prefill and decode throughput stress test.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API completions endpoint URL")
    parser.add_argument("--health-url", default=None, help="Server health endpoint URL")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "default", help="Model ID")
    parser.add_argument("--clients", "-c", type=int, nargs="+", default=[1, 2, 3, 4], help="Concurrency client levels (default: 1 2 3 4)")
    parser.add_argument("--contexts", type=int, nargs="+", default=[2000, 4000, 8000, 16000, 32000], help="Context token horizons (default: 2000 4000 8000 16000 32000)")
    parser.add_argument("--parallel", "-p", type=int, default=None, help="Legacy single-level parallelism flag")
    parser.add_argument("--tokens", "-t", type=int, default=64, help="Max generation tokens per client (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    args = parser.parse_args()

    health_url = args.health_url
    if not health_url:
        if "/v1/" in args.url:
            health_url = args.url.split("/v1/")[0] + "/health"
        else:
            health_url = DEFAULT_HEALTH_URL

    clients = [args.parallel] if args.parallel is not None else sorted(list(set(args.clients)))
    contexts = sorted(list(set(args.contexts)))

    print("=" * 88)
    print("Multi-Client Concurrency, Prefill & Decode Matrix Benchmark")
    print(f"Target URL   : {args.url}")
    print(f"Health URL   : {health_url}")
    print(f"Model ID     : {args.model}")
    print(f"Clients      : {clients}")
    print(f"Contexts     : {contexts}")
    print(f"Max Tokens   : {args.tokens} tokens/client")
    print(f"Temperature  : {args.temperature}")
    print("=" * 88)

    h_before = check_health(health_url, api_key=args.api_key)
    print(f"Health check before test: {h_before}\n")

    server_down = threading.Event()
    success = run_matrix_benchmark(clients, contexts, args.url, args.model, args.tokens, args.temperature, args.api_key, server_down)

    h_after = {"error": "Server unreachable; health check not sent"} if server_down.is_set() else check_health(health_url, api_key=args.api_key)
    print(f"\nHealth check after test: {h_after}")
    print("=" * 88)

    if not success:
        raise SystemExit(3 if server_down.is_set() else 1)

if __name__ == "__main__":
    main()
