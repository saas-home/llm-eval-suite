#!/usr/bin/env python3
"""
Stress test and throughput benchmark across long context windows (up to 204.8k tokens).
Measures:
- Prompt token count
- TTFT (Time To First Token) & Prefill speed (tokens/sec)
- Completion token count & Decode speed (tokens/sec)
"""

import json
import time
import sys
import os
import uuid
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from benchmarks.common import ChunkMetrics  # noqa: E402
from client import LLMClient  # noqa: E402

_ep = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_API_URL = _ep if _ep.endswith("/chat/completions") else f"{_ep}/chat/completions"

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. In computer science and artificial intelligence, "
    "large language models utilize transformer architectures with multi-head self-attention mechanisms "
    "to process sequential data efficiently. Memory bandwidth and compute capacity determine inference speed. "
)

def build_prompt(target_tokens, salt=True):
    # Base text is ~45 tokens and ~230 chars (~5 chars per token)
    repeat_count = max(1, int((target_tokens - (25 if salt else 0)) / 45))
    prefix = f"[Benchmark Target: {target_tokens} | Epoch: {time.time():.4f} | UUID: {uuid.uuid4()}]\n" if salt else ""
    text = (BASE_TEXT * repeat_count)
    return prefix + text + "\n\nSummarize the key aspects mentioned above in detail."

def run_benchmark_run(api_url, prompt_text, max_tokens=64, model="default", target_tokens=0, api_key=""):
    chunks = ChunkMetrics()
    result = LLMClient(api_url, api_key=api_key, model=model).call(
        [{"role": "user", "content": prompt_text}], max_tokens=max_tokens, temperature=0.0,
        stream=True, timeout=1200, chunk_callback=chunks,
    )
    if result.get("error"):
        print(f"\n[REQUEST ERROR]: {result['error']}", flush=True)
        return {key: result[key] for key in ("error", "failure_kind", "endpoint_unreachable") if key in result}

    prompt_tokens = result["prompt_tokens"] or target_tokens
    cached_tokens = result["cached_tokens"]
    ttft = result["ttft"]
    uncached_tokens = max(0, prompt_tokens - cached_tokens)
    cold_prefill_speed = uncached_tokens / ttft if ttft > 0 else 0.0
    effective_prefill_speed = prompt_tokens / ttft if ttft > 0 else 0.0
    
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_estimated": not bool(result["prompt_tokens"]),
        "completion_tokens": result["completion_tokens"],
        "cached_tokens": cached_tokens,
        "ttft_s": ttft,
        "cold_prefill_tok_s": cold_prefill_speed,
        "effective_prefill_tok_s": effective_prefill_speed,
        "decode_time_s": chunks.duration,
        "decode_tok_s": result["decode_speed"]
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark long context prefill and decode speeds.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="Endpoint URL")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "default", help="Model ID")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--tokens", type=int, nargs="+", default=[200000], help="Context token target(s)")
    parser.add_argument("--gen-tokens", type=int, default=64, help="Max generation tokens")
    parser.add_argument("--no-salt", action="store_true", help="Disable unique salt prepending (allows warm prefix caching)")
    parser.add_argument("--out", default="", help="Optional JSON results path")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    if not url.endswith("/chat/completions"):
        if url.endswith("/v1"):
            url = f"{url}/chat/completions"
        else:
            url = f"{url}/v1/chat/completions"

    print("=" * 96)
    print("Prefill & Decode Speed Benchmark Across Context Levels")
    print(f"Target URL: {url}")
    print(f"Model ID  : {args.model}")
    print(f"Prefix Isolation: {'Disabled (warm caching allowed)' if args.no_salt else 'Enabled (cold prefill isolated)'}")
    print("=" * 96)
    
    # Warmup
    print("Warming up GPU...", end="", flush=True)
    w = run_benchmark_run(url, "Warmup: reply with 5 words.", max_tokens=10, model=args.model, target_tokens=10, api_key=args.api_key)
    if w.get("error"):
        print(f" Warmup failed: {w['error']}")
        sys.exit(3)

    print(f" Ready. (Warmup prompt: {w.get('prompt_tokens', 0)} toks, decode: {w.get('decode_tok_s', 0):.1f} t/s)\n")
    
    header = f"{'Target Ctx':>12} | {'Actual Prompt':>14} | {'Cached':>8} | {'TTFT (s)':>10} | {'Cold (t/s)':>12} | {'Effective':>12} | {'Decode (t/s)':>12} | {'Gen Toks':>9}"
    print(header)
    print("-" * len(header))
    
    all_results = []
    incomplete = False
    for target in args.tokens:
        print(f">> Testing ~{target//1024}k context...", end="", flush=True)
        prompt = build_prompt(target, salt=not args.no_salt)
        res = run_benchmark_run(url, prompt, max_tokens=args.gen_tokens, model=args.model, target_tokens=target, api_key=args.api_key)

        print("\r" + " " * 40 + "\r", end="")
        
        if "error" in res and res["error"]:
            print(f"{target:>12,d} | {'FAILED':>14} | {'-':>8} | {'-':>10} | {str(res['error'])[:24]:>12}")
            all_results.append({"target_tokens": target, **res})
            incomplete = True
            if res.get("endpoint_unreachable"):
                break
            continue
            
        prompt_display = f"{res['prompt_tokens']:,}"
        if res["prompt_tokens_estimated"]:
            prompt_display = "~" + prompt_display
        row = (
            f"{target:>12,d} | "
            f"{prompt_display:>14} | "
            f"{res['cached_tokens']:>8,d} | "
            f"{res['ttft_s']:>10.3f} | "
            f"{res['cold_prefill_tok_s']:>12.1f} | "
            f"{res['effective_prefill_tok_s']:>12.1f} | "
            f"{res['decode_tok_s']:>12.2f} | "
            f"{res['completion_tokens']:>9d}"
        )
        print(row, flush=True)
        res["target_tokens"] = target
        all_results.append(res)
        time.sleep(2)
        
    print("-" * len(header))
    print("Benchmark complete.")

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"url": args.url, "model": args.model, "incomplete": incomplete, "results": all_results}, f, indent=2)
        print(f"Saved results to {args.out}")
    if incomplete:
        sys.exit(3)

if __name__ == "__main__":
    main()
