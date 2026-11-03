#!/usr/bin/env python3
"""
scripts/download_open_benchmarks.py
Zero-dependency utility to download open benchmark datasets directly from official sources:
- OpenAI GSM8K (test.jsonl)
- Google IFEval (input_data.jsonl)
- PubMedQA (test ground truth)
- MedQA / USMLE sample questions

Usage:
    python3 scripts/download_open_benchmarks.py --all
    python3 scripts/download_open_benchmarks.py --dataset gsm8k
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DATA_SOURCES = {
    "gsm8k": {
        "description": "OpenAI Grade School Math 8K (test.jsonl)",
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "dest": "data/downloads/gsm8k_test.jsonl",
    },
    "ifeval": {
        "description": "Google Research Instruction Following Eval (input_data.jsonl)",
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl",
        "dest": "data/downloads/ifeval_input.jsonl",
    },
    "pubmedqa": {
        "description": "PubMedQA labeled test dataset sample",
        "url": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json",
        "dest": "data/downloads/pubmedqa_test.json",
    },
}


def download_file(url: str, dest_path: str, description: str) -> bool:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[*] Downloading {description}...")
    print(f"    Source: {url}")
    print(f"    Dest:   {dest_path}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "llm-eval-suite-benchmark-downloader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            with open(dest_path, "wb") as f:
                f.write(content)
        size_kb = len(content) / 1024.0
        print(f"    [+] Saved successfully ({size_kb:.1f} KB)\n")
        return True
    except urllib.error.URLError as e:
        print(f"    [-] Network/download error: {e}\n", file=sys.stderr)
        return False
    except Exception as e:
        print(f"    [-] Unexpected error: {e}\n", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download open LLM benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATA_SOURCES.keys()) + ["all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    args = parser.parse_args()

    targets = list(DATA_SOURCES.keys()) if args.dataset == "all" else [args.dataset]
    success_count = 0
    for key in targets:
        info = DATA_SOURCES[key]
        if download_file(info["url"], info["dest"], info["description"]):
            success_count += 1

    print(f"Completed: {success_count}/{len(targets)} datasets downloaded successfully.")


if __name__ == "__main__":
    main()
