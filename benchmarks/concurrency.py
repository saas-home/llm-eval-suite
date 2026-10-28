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
import urllib.error
import threading
import argparse
import os
import uuid
import statistics

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
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(api_url, data=data, headers=headers)

    t_first = None
    t_last = None
    token_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    error_msg = None

    if barrier:
        barrier.wait()
    t_start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                line_data = line[5:].strip()
                if line_data == "[DONE]":
                    break
                try:
                    chunk = json.loads(line_data)
                except Exception:
                    continue

                if "error" in chunk:
                    error_msg = str(chunk["error"])
                    break

                choices = chunk.get("choices") or []
                if choices:
                    choice = choices[0]
                    if choice.get("finish_reason") == "error":
                        error_msg = "Stream terminated with finish_reason: error"
                        break
                    delta = choice.get("delta", {})
                    content = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("thought")
                    )
                    if content:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        token_count += 1

                if "usage" in chunk and chunk["usage"]:
                    prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    completion_tokens = chunk["usage"].get("completion_tokens", 0)
    except Exception as e:
        error_msg = str(e)

    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first else 0.0
    duration = (t_end - t_start)
    if completion_tokens == 0:
        completion_tokens = token_count
    if prompt_tokens == 0:
        prompt_tokens = int(len(prompt.split()) * 1.33)

    decode_time = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    decode_speed = (completion_tokens - 1) / decode_time if (decode_time > 0 and completion_tokens > 1) else 0.0
    prefill_speed = (prompt_tokens / ttft) if ttft > 0 else 0.0

    with lock:
        results[client_id] = {
            "prompt": prompt[:40] + ("..." if len(prompt) > 40 else ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_s": ttft,
            "total_time_s": duration,
            "decode_tok_s": decode_speed,
            "prefill_tok_s": prefill_speed,
            "t_dispatch": t_start,
            "t_first": t_first if t_first else (t_start + ttft),
            "t_last": t_last if t_last else t_end,
            "error": error_msg
        }

def run_matrix_benchmark(clients, contexts, api_url, model, tokens, temperature, api_key=""):
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
            t_start = time.perf_counter()

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
            wall_time = time.perf_counter() - t_start

            tot_prompt = sum(r["prompt_tokens"] for r in results.values())
            tot_completion = sum(r["completion_tokens"] for r in results.values() if not r.get("error"))
            errors = [r["error"] for r in results.values() if r.get("error")]
            if errors:
                has_errors = True
            valid_results = [r for r in results.values() if not r.get("error")]

            ttfts_s = [r["ttft_s"] for r in valid_results if r["ttft_s"] > 0]
            mean_ttft_s = statistics.mean(ttfts_s) if ttfts_s else 0.0

            # Prefill throughput: aggregate batch throughput and per-client average
            max_ttft_s = max((r["ttft_s"] for r in valid_results), default=0.0)
            agg_prefill = (tot_prompt / max_ttft_s) if max_ttft_s > 0 else 0.0
            avg_prefill = (agg_prefill / c) if c > 0 else 0.0

            # Decode throughput: aggregate across concurrent streams and per-client average
            decode_speeds = [r["decode_tok_s"] for r in valid_results if r["decode_tok_s"] > 0]
            agg_decode = sum(decode_speeds)
            avg_decode = (agg_decode / c) if c > 0 else 0.0

            status = "PASS" if not errors and tot_completion > 0 else "FAIL"
            row = (
                f"{ctx:>8,d} | {tot_prompt:>10,d} | {tot_completion:>9,d} | {mean_ttft_s:>14.2f} | "
                f"{avg_prefill:>17.1f} | {agg_prefill:>17.1f} | "
                f"{avg_decode:>16.1f} | {agg_decode:>16.1f} | {status:>8}"
            )
            print(row)
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

    success = run_matrix_benchmark(clients, contexts, args.url, args.model, args.tokens, args.temperature, args.api_key)

    h_after = check_health(health_url, api_key=args.api_key)
    print(f"\nHealth check after test: {h_after}")
    print("=" * 88)

    if not success:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
