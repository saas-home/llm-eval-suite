#!/usr/bin/env python3
"""
Simple streaming and latency verification script.
Tests:
- Streaming chunk reception
- First token latency (TTFT)
- Generation throughput
- Server health check
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from benchmarks.common import ChunkMetrics  # noqa: E402
from client import LLMClient  # noqa: E402

_ep = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_API_URL = _ep if _ep.endswith("/chat/completions") else f"{_ep}/chat/completions"

def run_test(api_url, prompt_text, max_tokens=64, model="default", api_key=""):
    chunks = ChunkMetrics(display=True)
    result = LLMClient(api_url, api_key=api_key, model=model).call(
        [{"role": "user", "content": prompt_text}], max_tokens=max_tokens, temperature=0.0,
        stream=True, timeout=300, chunk_callback=chunks,
    )
    print()
    if result.get("error"):
        print(f"Request failed: {result['error']}")
        return None
    ttft = result["ttft"]
    prompt_tokens = result["prompt_tokens"]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": result["completion_tokens"],
        "ttft_s": ttft,
        "prefill_tok_s": prompt_tokens / ttft if ttft > 0 else 0,
        "decode_time_s": chunks.duration,
        "decode_tok_s": result["decode_speed"],
        "text": result["text"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test streaming response and measure token latency.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API URL")
    parser.add_argument("--prompt", default="Why is the sky blue? Answer in 2 sentences.", help="Prompt text")
    parser.add_argument("--tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "default", help="Model ID")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    args = parser.parse_args()

    print(f"Connecting to {args.url} (Model: {args.model})...")
    res = run_test(args.url, args.prompt, max_tokens=args.tokens, model=args.model, api_key=args.api_key)
    if res:
        print("\n--- Summary ---")
        print(f"Prompt Tokens: {res['prompt_tokens']}")
        print(f"Gen Tokens   : {res['completion_tokens']}")
        print(f"TTFT         : {res['ttft_s']*1000:.1f} ms")
        print(f"Prefill Speed: {res['prefill_tok_s']:.1f} tok/s")
        print(f"Decode Speed : {res['decode_tok_s']:.1f} tok/s")
    else:
        sys.exit(1)
