#!/usr/bin/env python3
"""
Extreme-Precision Long-Context Benchmark Suite:
Calibrated for deep dependency tracking and floating-point ledger reconciliation across 60,000+ tokens.

Tests:
1. 14-Stage Sequential ALU Dependency Chain (arithmetic pipeline across 60k tokens)
2. 25-Transaction Floating-Point Ledger Reconciliation (multi-entity balance audit across 60k tokens)
"""

import sys
import os
import json
import argparse
from decimal import Decimal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from benchmarks.common import benchmark_error, call_nonstream, declared_integer, declared_money  # noqa: E402
from domain_benchmarks import final_content  # noqa: E402

_ep = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_API_URL = _ep if _ep.endswith("/chat/completions") else f"{_ep}/chat/completions"

def call_model(url, messages, max_tokens=3500, temperature=0.0, enable_thinking=True, timeout=600, model="default", api_key=""):
    return call_nonstream(url, messages, max_tokens, temperature, enable_thinking, timeout, model, api_key)

FILLER_BLOCK = (
    "In distributed storage clusters, erasure coding partitions data objects into k data chunks and m coding chunks "
    "using Cauchy or Vandermonde distribution matrices. The storage layer streams chunks across distinct failure domains "
    "to survive concurrent chassis or rack failures without data loss. Rebuilding degraded chunks requires reading k surviving "
    "fragments over low-latency RDMA interconnects and solving linear systems over Galois Field GF(2^8). Write pipelines "
    "utilize non-blocking asynchronous state machines to overlap network serializations with NVMe barrier flushes. "
) # ~75 tokens

def test_14_stage_dependency(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("TEST 1: 14-Stage Sequential ALU Dependency Chain (60k Tokens Context)")
    print("="*80)
    
    raw_ops = [
        ("stage_01", "stage_01 = 5123"),
        ("stage_02", "stage_02 = stage_01 + 1789"),
        ("stage_03", "stage_03 = stage_02 * 2 - 315"),
        ("stage_04", "stage_04 = stage_03 + 8421"),
        ("stage_05", "stage_05 = stage_04 - 3105"),
        ("stage_06", "stage_06 = (stage_05 // 5) + 47"),
        ("stage_07", "stage_07 = stage_06 * 3 - 101"),
        ("stage_08", "stage_08 = stage_07 + 19382"),
        ("stage_09", "stage_09 = stage_08 - 8921"),
        ("stage_10", "stage_10 = stage_09 * 2 + 55"),
        ("stage_11", "stage_11 = stage_10 + 12044"),
        ("stage_12", "stage_12 = stage_11 - 5020"),
        ("stage_13", "stage_13 = stage_12 * 2 - 99"),
        ("stage_14", "stage_14 = stage_13 + 33"),
    ]
    
    # Ground truth:
    val = 5123
    val = val + 1789
    val = val * 2 - 315
    val = val + 8421
    val = val - 3105
    val = (val // 5) + 47
    val = val * 3 - 101
    val = val + 19382
    val = val - 8921
    val = val * 2 + 55
    val = val + 12044
    val = val - 5020
    val = val * 2 - 99
    val = val + 33
    expected_final = val  # 101276
    
    total_blocks = 800
    stride = total_blocks // len(raw_ops)
    
    doc = []
    op_idx = 0
    for b in range(total_blocks):
        doc.append(FILLER_BLOCK)
        if b % stride == 0 and op_idx < len(raw_ops):
            _, op_text = raw_ops[op_idx]
            doc.append(f"\n[CRITICAL_ALU_INSTRUCTION: {op_text}]\n")
            op_idx += 1
            
    doc_text = "".join(doc)
    prompt = (
        f"{doc_text}\n\n"
        "TASK:\n"
        "Carefully locate all 14 sequential ALU instructions (stage_01 through stage_14) in the document.\n"
        "Calculate the exact intermediate value of each stage step-by-step, and provide the exact final integer value of stage_14.\n"
        "Output format: End with 'FINAL_STAGE_14: <integer>'."
    )
    
    print("  Ingesting ~60,000 tokens with 14 chained dependency stages...")
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=3500, temperature=0.0, enable_thinking=True, model=model, api_key=api_key)
    name = "14-Stage Dependency Chain"
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s)")
    
    passed = declared_integer(final_content(res), "FINAL_STAGE_14") == expected_final
    print(f"  - Expected stage_14 = {expected_final}: {'MATCH' if passed else 'FAIL'}")
    print(f"  [Result]: {'PASS' if passed else 'FAIL'}")
    return {"name": name, "passed": passed, "expected": expected_final}

def test_financial_ledger_reconciliation(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("TEST 2: 25-Transaction Floating-Point Ledger Reconciliation (60k Tokens Context)")
    print("="*80)
    
    initial_balance = 10000.00
    transactions = [
        ("TX01", 142.35),
        ("TX02", -89.12),
        ("TX03", 530.40),
        ("TX04", -214.88),
        ("TX05", 98.75),
        ("TX06", -431.10),
        ("TX07", 75.22),
        ("TX08", -1050.50),
        ("TX09", 312.80),
        ("TX10", -64.30),
        ("TX11", 890.15),
        ("TX12", -125.45),
        ("TX13", 44.90),
        ("TX14", -780.00),
        ("TX15", 215.65),
        ("TX16", -33.20),
        ("TX17", 412.50),
        ("TX18", -95.75),
        ("TX19", 1100.00),
        ("TX20", -540.25),
        ("TX21", 82.40),
        ("TX22", -19.85),
        ("TX23", 305.10),
        ("TX24", -450.60),
        ("TX25", 1250.00)
    ]
    
    bal = initial_balance
    for _, amt in transactions:
        bal += amt
    expected_balance = round(bal, 2)  # 11565.22
    
    total_blocks = 800
    stride = total_blocks // (len(transactions) + 1)
    
    doc = [f"[LEDGER_INITIAL_BALANCE: Account #908412 initial balance is ${initial_balance:.2f}]\n"]
    tx_idx = 0
    for b in range(total_blocks):
        doc.append(FILLER_BLOCK)
        if b % stride == 0 and tx_idx < len(transactions):
            tx_id, amt = transactions[tx_idx]
            sign_str = f"+${amt:.2f}" if amt > 0 else f"-${abs(amt):.2f}"
            doc.append(f"\n[LEDGER_AUDIT_POSTING: Transaction {tx_id}: {sign_str}]\n")
            tx_idx += 1
            
    doc_text = "".join(doc)
    prompt = (
        f"{doc_text}\n\n"
        "TASK:\n"
        "Locate the initial balance and all 25 ledger audit postings (TX01 through TX25) in the text.\n"
        "Sum all postings to the initial balance step-by-step and calculate the exact ending balance to two decimal places.\n"
        "State the final ending balance clearly: 'ENDING_BALANCE: $<amount>'."
    )
    
    print("  Ingesting ~60,000 tokens with 25 floating-point financial transactions...")
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=3500, temperature=0.0, enable_thinking=True, model=model, api_key=api_key)
    name = "25-Transaction Ledger Reconciliation"
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s)")
    
    exp_str = f"{expected_balance:.2f}"
    passed = declared_money(final_content(res), "ENDING_BALANCE") == Decimal(exp_str)
    print(f"  - Expected ending balance = ${exp_str}: {'MATCH' if passed else 'FAIL'}")
    print(f"  [Result]: {'PASS' if passed else 'FAIL'}")
    return {"name": name, "passed": passed, "expected": exp_str}

def main():
    parser = argparse.ArgumentParser(description="Extreme Precision Long-Context Stress Test.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API completion endpoint")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "default", help="Model name or ID")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "results", "extreme_precision_results.json"), help="Path to output JSON")
    args = parser.parse_args()
    
    results = []
    checks = [
        ("14-Stage Dependency Chain", test_14_stage_dependency),
        ("25-Transaction Ledger Reconciliation", test_financial_ledger_reconciliation),
    ]
    for index, (_, check) in enumerate(checks):
        result = check(args.url, model=args.model, api_key=args.api_key)
        results.append(result)
        if result.get("endpoint_unreachable"):
            results.extend({"name": name, "passed": False, "status": "SKIPPED",
                            "reason": "Server unreachable"} for name, _ in checks[index + 1:])
            break
    
    print("\n" + "="*80)
    print("EXTREME PRECISION BENCHMARK SUMMARY")
    print("="*80)
    for r in results:
        status = r.get("status", "PASSED" if r["passed"] else "FAILED")
        print(f" • {r['name']:<42} : {status}")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}")
    if any(result.get("status") == "ERROR" for result in results):
        sys.exit(3)

if __name__ == "__main__":
    main()
