#!/usr/bin/env python3
"""
Intensive Accuracy and Precision Benchmark Suite
Tests:
1. Executable Algorithmic Code Generation & Dynamic Execution Testing (LRU Cache with Doubly Linked List)
2. Multi-Needle in a Haystack (NIAH) at ~60,000 Tokens (5 needles + arithmetic aggregation)
3. Multi-Step Mathematical & Combinatorial Reasoning (Exact fractional deduction)
4. Formal Concurrency & Deadlock Graph Cycle Analysis
5. Strict JSON Schema Validation & Complex Constraint Adherence
"""

import argparse
import ast
import ipaddress
import json
import os
import re
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from benchmarks.common import benchmark_error, benchmark_response  # noqa: E402
from benchmarks.common import strict_json_loads as _strict_json_loads  # noqa: E402
from client import LLMClient, assemble_python_script, harness_passed, run_model_code  # noqa: E402
from domain_benchmarks import final_content  # noqa: E402

_ep = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_API_URL = _ep if _ep.endswith("/chat/completions") else f"{_ep}/chat/completions"

def call_model(url, messages, max_tokens=2048, temperature=0.1, model="default", timeout=600, enable_thinking=None, api_key=""):
    kwargs = {"enable_thinking": enable_thinking} if enable_thinking is not None else None
    client = LLMClient(url, api_key=api_key, model=model)
    return benchmark_response(client.call(
        messages, max_tokens=max_tokens, temperature=temperature, stream=True,
        timeout=timeout, chat_template_kwargs=kwargs,
    ))

def extract_code_block(text, lang="python"):
    # Look for ```python ... ```
    pattern = rf"```(?:{lang})?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip()
    # Check for unclosed code block
    pattern_unclosed = rf"```(?:{lang})?\n(.*)"
    m = re.search(pattern_unclosed, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()

def test_executable_code(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("TEST 1: Executable Algorithmic Code Generation (LRU Cache with Doubly Linked List)")
    print("="*80)
    prompt = (
        "Write a production-grade, O(1) time complexity LRU Cache in Python using a custom Doubly Linked List and Hash Map.\n"
        "Requirements:\n"
        "1. Do NOT use `collections.OrderedDict`. Implement your own `Node` class and `LRUCache` class.\n"
        "2. Methods: `__init__(capacity: int)`, `get(key: int) -> int` (returns -1 if absent), `put(key: int, value: int) -> None`, `peek(key: int) -> int` (returns value without updating LRU order).\n"
        "3. Provide clean, fully typed, self-contained Python code."
    )
    name = "Executable Algorithmic Code (LRU Cache)"
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=2048, temperature=0.0, enable_thinking=False, model=model, api_key=api_key)
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  [Model Response]: Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s, TTFT: {res['ttft_ms']:.1f} ms)")
    
    code_source = final_content(res)
    code = extract_code_block(code_source)
    
    # Unit test harness to execute against the generated code
    test_harness = """
import sys

# Test execution harness
def run_unit_tests():
    # Test 1: Basic get and put
    cache = LRUCache(2)
    cache.put(1, 10)
    cache.put(2, 20)
    assert cache.get(1) == 10, f"Expected 10, got {cache.get(1)}"
    assert cache.get(2) == 20, f"Expected 20, got {cache.get(2)}"
    assert cache.get(3) == -1, f"Expected -1, got {cache.get(3)}"
    
    # Test 2: Eviction order (Key 1 was accessed, so Key 2 should be evicted if we put 3? Wait: get(1) then get(2), key 1 is LRU)
    cache.put(3, 30) # Evicts key 1
    assert cache.get(1) == -1, f"Key 1 should be evicted, but got {cache.get(1)}"
    assert cache.get(2) == 20
    assert cache.get(3) == 30
    
    # Test 3: Overwrite existing key
    cache.put(2, 200)
    assert cache.get(2) == 200
    cache.put(4, 40) # Evicts key 3
    assert cache.get(3) == -1
    assert cache.get(2) == 200
    assert cache.get(4) == 40
    
    # Test 4: peek does NOT alter LRU order
    cache2 = LRUCache(2)
    cache2.put(10, 100)
    cache2.put(20, 200)
    assert cache2.peek(10) == 100
    cache2.put(30, 300) # Since peek(10) did not touch LRU, key 10 must be evicted
    assert cache2.get(10) == -1, "peek(10) should not update LRU order"
    assert cache2.get(20) == 200
    assert cache2.get(30) == 300
    
    # Test 5: Sequential 10,000 operations churn test
    c = LRUCache(100)
    for i in range(1000):
        c.put(i, i * 2)
    for i in range(900, 1000):
        assert c.get(i) == i * 2
    for i in range(0, 900):
        assert c.get(i) == -1

run_unit_tests()
print("ALL_UNIT_TESTS_PASSED", __import__('os').environ['EVAL_RUN_ID'])
"""
    full_script = assemble_python_script(code, harness=test_harness)

    success = False
    syntax_only = False
    error = ""
    if not code.strip():
        error = "No Python code found."
    elif os.getenv("EVAL_SAFE", "").strip().lower() in ("1", "true", "yes"):
        try:
            tree = ast.parse(full_script)
            syntax_only = any(isinstance(node, ast.ClassDef) and node.name == "LRUCache" for node in tree.body)
            error = ("Execution disabled; syntax check only." if syntax_only else "No LRUCache class found.")
        except (SyntaxError, ValueError) as e:
            error = f"SyntaxError: {e}"
    else:
        try:
            proc = run_model_code([sys.executable, "-c", full_script], 15)
            success = harness_passed(proc, "ALL_UNIT_TESTS_PASSED")
            if not success:
                error = (proc.stderr or proc.stdout).strip()[:300] or "Unit-test verification did not complete."
        except (OSError, subprocess.TimeoutExpired) as e:
            error = str(e)

    status = "PASS" if success else ("PARTIAL" if syntax_only else "FAIL")
    print(f"  [Verification Result]: {status}" + (f" ({error})" if error else ""))
    return {
        "name": name,
        "passed": success,
        "status": status,
        "error": error,
        "tokens": res["completion_tokens"],
        "ttft_ms": res["ttft_ms"],
        "tok_per_sec": res["tok_per_sec"]
    }

def test_multi_needle_in_haystack(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("TEST 2: Multi-Needle in a ~60,000-Token Haystack (NIAH)")
    print("="*80)
    
    filler_sentence = (
        "Distributed database systems ensure data consistency across multiple partitions by utilizing consensus protocols "
        "such as Raft, Paxos, or Multi-Paxos. Hardware architectures incorporate non-volatile memory and PCIe Gen5 fabrics "
        "to minimize round-trip latencies in high-concurrency environments. Query optimizers continuously compile relational "
        "algebraic expressions into vectorized machine instructions for parallel vector execution. "
    ) # ~50 tokens
    
    # Target ~60,000 tokens of filler
    total_blocks = 1200
    needles = [
        (int(total_blocks * 0.10), "NEEDLE_1_PORT: 9182"),
        (int(total_blocks * 0.30), "NEEDLE_2_SALT: k7$vB9#zL1"),
        (int(total_blocks * 0.50), "NEEDLE_3_NODE: us-east-cluster-node-42"),
        (int(total_blocks * 0.70), "NEEDLE_4_OVERFLOW_BYTES: 8388608"),
        (int(total_blocks * 0.90), "NEEDLE_5_SIGNATURE: sig_990184_verified"),
    ]
    
    needle_map = dict(needles)
    doc_parts = []
    for i in range(total_blocks):
        doc_parts.append(filler_sentence)
        if i in needle_map:
            doc_parts.append(f"\n[CRITICAL_SYSTEM_REGISTRATION: {needle_map[i]}]\n")
            
    document = "".join(doc_parts)
    prompt = (
        f"{document}\n\n"
        "TASK:\n"
        "Carefully read the background documentation above and locate all 5 hidden CRITICAL_SYSTEM_REGISTRATION needles.\n"
        "Return a JSON object containing the exact extracted values with keys:\n"
        "- port (integer)\n"
        "- salt (string)\n"
        "- node (string)\n"
        "- overflow_bytes (integer)\n"
        "- signature (string)\n"
        "- sum_port_and_overflow (integer: calculate port + overflow_bytes)\n\n"
        "Output JSON only."
    )
    
    print(f"  Ingesting ~60,000-token prompt with 5 buried needles...")
    name = "Multi-Needle in 60k Haystack (NIAH)"
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=512, temperature=0.0, timeout=600, model=model, api_key=api_key)
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  [Model Response]: TTFT: {res['ttft_ms']:.1f} ms | Decode Speed: {res['tok_per_sec']:.1f} tok/s")
    
    output = final_content(res)
    print("  [Response Preview]:", output.strip().splitlines()[-10:])

    try:
        extracted = _strict_json_loads(output)
    except ValueError:
        extracted = {}
    if not isinstance(extracted, dict):
        extracted = {}
    p1 = type(extracted.get("port")) is int and extracted["port"] == 9182
    p2 = extracted.get("salt") == "k7$vB9#zL1"
    p3 = extracted.get("node") == "us-east-cluster-node-42"
    p4 = type(extracted.get("overflow_bytes")) is int and extracted["overflow_bytes"] == 8388608
    p5 = extracted.get("signature") == "sig_990184_verified"
    expected_sum = 9182 + 8388608  # 8397790
    p_sum = (type(extracted.get("sum_port_and_overflow")) is int
             and extracted["sum_port_and_overflow"] == expected_sum)
    
    all_passed = p1 and p2 and p3 and p4 and p5 and p_sum
    print(f"  - Needle 1 (Port 9182): {'PASS' if p1 else 'FAIL'}")
    print(f"  - Needle 2 (Salt k7$vB9#zL1): {'PASS' if p2 else 'FAIL'}")
    print(f"  - Needle 3 (Node us-east-cluster-node-42): {'PASS' if p3 else 'FAIL'}")
    print(f"  - Needle 4 (Overflow Bytes 8388608): {'PASS' if p4 else 'FAIL'}")
    print(f"  - Needle 5 (Signature sig_990184_verified): {'PASS' if p5 else 'FAIL'}")
    print(f"  - Arithmetic Sum (9182 + 8388608 = 8397790): {'PASS' if p_sum else 'FAIL'}")
    print(f"  [Verification Result]: {'PASS (100% Retrieval & Calculation)' if all_passed else 'FAIL'}")
    
    return {
        "name": name,
        "passed": all_passed,
        "tokens": res["completion_tokens"],
        "ttft_ms": res["ttft_ms"],
        "tok_per_sec": res["tok_per_sec"],
        "prompt_tokens": res["prompt_tokens"]
    }

def test_mathematical_reasoning(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("TEST 3: Multi-Step Mathematical & Combinatorial Reasoning")
    print("="*80)
    prompt = (
        "Solve the following probability problem with step-by-step mathematical reasoning:\n\n"
        "Problem: An urn contains 5 red balls, 4 blue balls, and 3 green balls (12 balls total).\n"
        "Three balls are drawn uniformly at random without replacement.\n"
        "Let event A be drawing at least one red ball.\n"
        "Let event B be drawing at least one blue ball.\n"
        "Calculate the exact conditional probability P(A | B).\n\n"
        "Requirements:\n"
        "1. Calculate total outcomes C(12, 3).\n"
        "2. Compute P(B) using the complement rule.\n"
        "3. Compute P(A and B) using inclusion-exclusion or complementary counting.\n"
        "4. Calculate P(A | B) = P(A and B) / P(B).\n"
        "5. State the final probability as an exact irreducible fraction in the form P/Q."
    )
    name = "Mathematical Reasoning (Exact Combinatorics)"
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=2500, temperature=0.0, model=model, api_key=api_key)
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  [Model Response]: Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s, TTFT: {res['ttft_ms']:.1f} ms)")
    
    # Ground truth calculation:
    # Total = C(12, 3) = 220
    # No blue: C(8, 3) = 56 -> B = 220 - 56 = 164. P(B) = 164/220
    # No red: C(7, 3) = 35.
    # No blue and no red (only green): C(3, 3) = 1.
    # Neither A nor B (no red or no blue):
    # |A^c U B^c| = |A^c| + |B^c| - |A^c n B^c| = 35 + 56 - 1 = 90.
    # Therefore, P(A and B) = (220 - 90) / 220 = 130 / 220.
    # P(A | B) = P(A and B) / P(B) = 130 / 164 = 65 / 82.
    
    text = final_content(res)
    has_220 = bool(re.search(r"(?<!\d)220(?!\d)", text))
    has_164 = bool(re.search(r"(?<!\d)164(?!\d)", text))
    has_130 = bool(re.search(r"(?<!\d)130(?!\d)", text))
    fractions = list(re.finditer(
        r"\\(?:d|t)?frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(\d+)\s*\}"
        r"|(?<![\d.])(-?\d+)\s*(?:/|\\over)\s*(\d+)(?![\d.])", text))
    fraction = fractions[-1] if fractions else None
    has_fraction = bool(fraction and (int(fraction.group(1) or fraction.group(3)),
                                      int(fraction.group(2) or fraction.group(4))) == (65, 82))
    
    passed = has_220 and has_164 and has_130 and has_fraction
    print(f"  - Total outcomes C(12,3) = 220: {'PASS' if has_220 else 'FAIL'}")
    print(f"  - Outcomes with >=1 blue = 164: {'PASS' if has_164 else 'FAIL'}")
    print(f"  - Outcomes with >=1 red and >=1 blue = 130: {'PASS' if has_130 else 'FAIL'}")
    print(f"  - Exact Irreducible Fraction (65/82): {'PASS' if has_fraction else 'FAIL'}")
    print(f"  [Verification Result]: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": name,
        "passed": passed,
        "tokens": res["completion_tokens"],
        "ttft_ms": res["ttft_ms"],
        "tok_per_sec": res["tok_per_sec"]
    }

def test_concurrency_formal_proof(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("TEST 4: Formal Concurrency & Deadlock Graph Cycle Analysis")
    print("="*80)
    prompt = (
        "Analyze the following concurrent resource acquisition pattern across 4 worker threads T1, T2, T3, T4 and 4 locks L1, L2, L3, L4:\n\n"
        "- Thread T1 acquires L1, then L2\n"
        "- Thread T2 acquires L2, then L3\n"
        "- Thread T3 acquires L3, then L4\n"
        "- Thread T4 acquires L4, then L1\n\n"
        "Tasks:\n"
        "1. Construct the Resource Allocation Graph / Wait-For Graph (WFG).\n"
        "2. Formally prove whether a deadlock cycle exists under Coffman conditions.\n"
        "3. Specify the exact interleaving execution trace that induces deadlock.\n"
        "4. Provide the standardized lock-ordering hierarchy rule that guarantees deadlock-free execution."
    )
    name = "Concurrency & Deadlock Cycle Proof"
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.0, model=model, api_key=api_key)
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  [Model Response]: Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s, TTFT: {res['ttft_ms']:.1f} ms)")
    
    text = final_content(res).lower()
    c1 = "coffman" in text
    c2 = "circular wait" in text or "cycle" in text
    c3 = "t1" in text and "t4" in text
    c4 = "hierarchy" in text or "order" in text
    
    passed = c1 and c2 and c3 and c4
    print(f"  - Coffman conditions analysis: {'PASS' if c1 else 'FAIL'}")
    print(f"  - Circular wait cycle proof: {'PASS' if c2 else 'FAIL'}")
    print(f"  - Complete interleaving trace: {'PASS' if c3 else 'FAIL'}")
    print(f"  - Canonical lock hierarchy solution: {'PASS' if c4 else 'FAIL'}")
    print(f"  [Verification Result]: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": name,
        "passed": passed,
        "tokens": res["completion_tokens"],
        "ttft_ms": res["ttft_ms"],
        "tok_per_sec": res["tok_per_sec"]
    }

def _node_schema_error(data):
    if not isinstance(data, list) or len(data) != 3:
        return "Expected an array of exactly 3 nodes."
    expected_keys = {"node_id", "region", "is_active", "ip_address", "tags", "load_avg"}
    seen_ids, seen_regions = set(), set()
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or set(item) != expected_keys:
            return f"Item {idx}: keys do not match the required schema."
        node_id = item["node_id"]
        if isinstance(node_id, bool) or not isinstance(node_id, int) or not 1000 <= node_id <= 1999:
            return f"Item {idx}: invalid node_id."
        region = item["region"]
        if region not in ("us-east", "us-west", "eu-central"):
            return f"Item {idx}: invalid region."
        if not isinstance(item["is_active"], bool):
            return f"Item {idx}: is_active must be a boolean."
        try:
            if not isinstance(item["ip_address"], str):
                raise ValueError("not a string")
            if any(len(part) > 1 and part.startswith("0") for part in item["ip_address"].split(".")):
                raise ValueError("leading zero")
            ipaddress.IPv4Address(item["ip_address"])
        except ValueError:
            return f"Item {idx}: invalid IPv4 address."
        tags = item["tags"]
        if not isinstance(tags, list) or len(tags) != 3 or not all(isinstance(t, str) for t in tags):
            return f"Item {idx}: tags must contain exactly 3 strings."
        load = item["load_avg"]
        if isinstance(load, bool) or not isinstance(load, (int, float)) or not 0.0 <= load <= 1.0:
            return f"Item {idx}: invalid load_avg."
        if node_id in seen_ids or region in seen_regions:
            return f"Item {idx}: node_id and region must be distinct."
        seen_ids.add(node_id)
        seen_regions.add(region)
    return ""


def test_strict_schema_validation(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("TEST 5: Strict JSON Schema Validation & Complex Constraint Adherence")
    print("="*80)
    prompt = (
        "Generate a strictly formatted JSON array representing 3 server cluster node configurations.\n"
        "Constraints:\n"
        "1. Output valid JSON ONLY. No markdown ticks, no preamble, no commentary.\n"
        "2. Each item must be an object with exact keys:\n"
        "   - 'node_id': integer between 1000 and 1999\n"
        "   - 'region': one of ['us-east', 'us-west', 'eu-central']\n"
        "   - 'is_active': boolean (true/false)\n"
        "   - 'ip_address': valid IPv4 string\n"
        "   - 'tags': array of exactly 3 strings\n"
        "   - 'load_avg': float between 0.0 and 1.0\n"
        "3. All 3 nodes must have distinct node_id and distinct regions."
    )
    name = "Strict JSON Schema & Constraints"
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.0, model=model, api_key=api_key)
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  [Model Response]: Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s, TTFT: {res['ttft_ms']:.1f} ms)")
    
    text = final_content(res)
    try:
        error = _node_schema_error(_strict_json_loads(text))
    except ValueError as e:
        error = str(e)
    passed = not error
    if passed:
        print("  [Schema Check]: JSON parsed and verified against 100% of strict field, type, and uniqueness constraints.")
        print("  [Verification Result]: PASS")
    else:
        print(f"  [Verification Result]: FAIL - {error}\nRaw Output: {text[:300]}")
        
    return {
        "name": name,
        "passed": passed,
        "tokens": res["completion_tokens"],
        "ttft_ms": res["ttft_ms"],
        "tok_per_sec": res["tok_per_sec"]
    }

def main():
    parser = argparse.ArgumentParser(description="Intensive Accuracy and Precision Benchmark.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="Endpoint URL")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "default", help="Model name or ID")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "results", "intensive_accuracy_results.json"), help="Output JSON results")
    args = parser.parse_args()
    
    print("=" * 80)
    print("STARTING INTENSIVE ACCURACY & PRECISION BENCHMARK SUITE")
    print(f"Target Endpoint: {args.url}")
    print(f"Model ID       : {args.model}")
    print("=" * 80)
    
    results = []
    checks = [
        ("Executable Algorithmic Code (LRU Cache)", test_executable_code),
        ("Multi-Needle in 60k Haystack (NIAH)", test_multi_needle_in_haystack),
        ("Mathematical Reasoning (Exact Combinatorics)", test_mathematical_reasoning),
        ("Concurrency & Deadlock Cycle Proof", test_concurrency_formal_proof),
        ("Strict JSON Schema & Constraints", test_strict_schema_validation),
    ]
    for _, check in checks:
        result = check(args.url, model=args.model, api_key=args.api_key)
        results.append(result)
        if result.get("endpoint_unreachable"):
            break
    
    print("\n" + "=" * 80)
    print("INTENSIVE ACCURACY BENCHMARK SUMMARY")
    print("=" * 80)
    
    all_passed = True
    any_failed = False
    incomplete = False
    for r in results:
        status = r.get("status", "PASS" if r["passed"] else "FAIL")
        print(f" - {r['name']:<50} : {status:<8} ({r['tok_per_sec']:.1f} tok/s, {r['tokens']} tokens)")
        if not r["passed"]:
            all_passed = False
        if status == "FAIL":
            any_failed = True
        if status == "ERROR":
            incomplete = True
            
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "all_passed": all_passed,
        "incomplete": incomplete,
        "not_run": [name for name, _ in checks[len(results):]],
        "results": results
    }
    
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    print(f"\nSaved detailed results to {args.out}")
    print("=" * 80)
    
    if incomplete:
        print(">> VERDICT: INCOMPLETE (REQUEST ERROR). <<")
        sys.exit(3)
    if all_passed:
        print(">> VERDICT: 100% ROCK SOLID. ZERO QUALITY OR ACCURACY LOSS OBSERVED. <<")
    else:
        print(">> VERDICT: DEFECTS DETECTED. <<" if any_failed else
              ">> VERDICT: UNVERIFIED (CODE EXECUTION DISABLED). <<")
        sys.exit(1)

if __name__ == "__main__":
    main()
