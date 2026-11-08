#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from benchmarks.common import ChunkMetrics  # noqa: E402
from client import LLMClient  # noqa: E402


def _resolve_image_path():
    for rel in [os.path.join("assets", "invoice.png"), os.path.join("assets", "image.png"), "invoice.png", "image.png"]:
        for d in [ROOT_DIR, BASE_DIR]:
            p = os.path.join(d, rel)
            if os.path.exists(p):
                return p
    return os.path.join(ROOT_DIR, "assets", "invoice.png")

IMAGE_PATH = _resolve_image_path()

def main():
    default_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1").rstrip("/")
    if not default_url.endswith("/chat/completions"):
        default_url = f"{default_url}/chat/completions"

    parser = argparse.ArgumentParser(description="Test multimodal vision endpoint.")
    parser.add_argument("--url", default=default_url, help="Endpoint chat completions URL")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "default", help="Model name")
    parser.add_argument("--image", default=IMAGE_PATH, help="Path to test image")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key for authentication")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "results", "vision_result.json"), help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Please summarize this document image in detail, extracting key information such as invoice number, dates, parties involved, line items, rates, hours, and total amount."
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }]

    print(f"Sending multimodal vision request to {args.url} (model: {args.model})...")
    result = LLMClient(args.url, api_key=args.api_key, model=args.model).call(
        messages, max_tokens=1024, temperature=0.0, stream=True, timeout=300,
        chunk_callback=ChunkMetrics(display=True),
    )
    if result.get("error"):
        print(f"\nRequest failed: {result['error']}")
        sys.exit(3 if result.get("endpoint_unreachable") else 1)

    print("\n" + "="*60)
    ttft_ms = result["ttft"] * 1000.0
    final_tokens = result["completion_tokens"]
    decode_speed = result["decode_speed"]

    print(f"Total time: {result['total_time']:.2f}s")
    print(f"TTFT (Prefill + Vision Encoding): {ttft_ms:.2f} ms")
    print(f"Tokens Generated: {final_tokens}")
    print(f"Decode Speed: {decode_speed:.2f} tok/s")

    report = {
        "url": args.url,
        "model": args.model,
        "ttft_ms": round(ttft_ms, 2),
        "tokens": final_tokens,
        "decode_tok_s": round(decode_speed, 2),
        "total_time_s": round(result["total_time"], 2),
        "response": result["text"]
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved vision benchmark result to {args.out}")

if __name__ == "__main__":
    main()
