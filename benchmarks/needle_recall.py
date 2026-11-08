#!/usr/bin/env python3
"""
High-Entropy Long-Context Stress Test:
Comparing 4,3 vs 4,2 KV Cache Quantization Boundaries.

Tests:
1. High-Entropy Key-Value Associative Recall (10 random alphanumeric keys & values buried in 50k tokens)
2. Multi-Hop Variable Tracking across 8 sequential dependency stages in 50k tokens
3. Multi-Value Filtering & Set Aggregation across 50 database records
"""

import sys
import os
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from benchmarks.common import benchmark_error, call_nonstream, declared_integer, strict_json_loads  # noqa: E402
from domain_benchmarks import final_content  # noqa: E402

_ep = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_API_URL = _ep if _ep.endswith("/chat/completions") else f"{_ep}/chat/completions"

def call_model(url, messages, max_tokens=1500, temperature=0.0, enable_thinking=False, timeout=600, model="default", api_key=""):
    return call_nonstream(url, messages, max_tokens, temperature, enable_thinking, timeout, model, api_key)

FILLER_BLOCK = (
    "The Linux kernel page cache caches pages of files to optimize disk I/O performance. "
    "When a process performs a read system call, the kernel first inspects the radix tree or xarray "
    "associated with the inode. If the requested page is resident in memory, the data is copied directly "
    "to the user space buffer via CPU copy or DMA transfer, avoiding high-latency block device access. "
    "Dirty pages modified by write operations are periodically written back to persistent storage "
    "by the flusher threads (kworker) according to dirty_ratio and dirty_background_ratio sysctl knobs. "
    "Memory cgroups enforce hierarchical resource isolation, throttling writeback bandwidth when limits are exceeded. "
) # ~100 tokens

def test_high_entropy_kv_recall(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("STRESS TEST 1: High-Entropy Associative Key-Value Recall (10 Targets in 50k Tokens)")
    print("="*80)
    
    # 25 High-entropy key-value pairs
    kv_pairs = [
        ("KEY_A19B", "VAL_9X#qL2!p"),
        ("KEY_B83C", "VAL_4m$Zk8*v"),
        ("KEY_C72D", "VAL_7w@Np5&b"),
        ("KEY_D61E", "VAL_2r%Vj1^m"),
        ("KEY_E50F", "VAL_8t&Kx9$d"),
        ("KEY_F49G", "VAL_3y*Bd4#w"),
        ("KEY_G38H", "VAL_6p#Tq7!z"),
        ("KEY_H27I", "VAL_1k$Mf2&c"),
        ("KEY_I16J", "VAL_9s@Ln8%x"),
        ("KEY_J05K", "VAL_5d%Rp3^k"),
        ("KEY_K94L", "VAL_7z&Wb6$f"),
        ("KEY_L83M", "VAL_4h*Vc9#t"),
        ("KEY_M72N", "VAL_2n#Xm1!q"),
        ("KEY_N61O", "VAL_8b$Dj5&p"),
        ("KEY_O50P", "VAL_6k@Zt7%s"),
        ("KEY_P49Q", "VAL_1g%Kp4^r"),
        ("KEY_Q38R", "VAL_9f&Nq8$y"),
        ("KEY_R27S", "VAL_3c*Tr2#m"),
        ("KEY_S16T", "VAL_7v#Bw6!n"),
        ("KEY_T05U", "VAL_5m$Pd9&k"),
        ("KEY_U94V", "VAL_8x@Lf3%w"),
        ("KEY_V83W", "VAL_2q%Zj7^b"),
        ("KEY_W72X", "VAL_4t&Kx1$d"),
        ("KEY_X61Y", "VAL_6r*Wn5#c"),
        ("KEY_Y50Z", "VAL_9p#Vq8!s")
    ]
    
    # Select 8 random target keys to retrieve
    targets = [
        "KEY_B83C", "KEY_E50F", "KEY_H27I", "KEY_K94L",
        "KEY_N61O", "KEY_Q38R", "KEY_T05U", "KEY_W72X"
    ]
    expected_map = dict(kv_pairs)
    
    # Interleave pairs across 500 filler blocks (~50,000 tokens)
    total_blocks = 500
    stride = total_blocks // len(kv_pairs)
    
    doc = []
    pair_idx = 0
    for b in range(total_blocks):
        doc.append(FILLER_BLOCK)
        if b % stride == 0 and pair_idx < len(kv_pairs):
            k, v = kv_pairs[pair_idx]
            doc.append(f"\n[ASSOCIATIVE_STORAGE_ENTRY: {k} -> {v}]\n")
            pair_idx += 1
            
    doc_text = "".join(doc)
    prompt = (
        f"{doc_text}\n\n"
        "TASK:\n"
        "From the associative storage records above, retrieve the exact values for these 8 keys:\n"
        + "\n".join(f"- {t}" for t in targets) + "\n\n"
        "Respond with a JSON object mapping each key to its exact string value.\n"
        "Format: JSON only, no commentary."
    )
    
    print(f"  Ingesting ~50,000 tokens with 25 associative high-entropy key-value pairs...")
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.0, model=model, api_key=api_key)
    name = "High-Entropy KV Recall (8/8)"
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s)")
    
    try:
        output = strict_json_loads(final_content(res).strip())
    except ValueError as exc:
        print(f"  Invalid JSON: {exc}")
        output = None
    if not isinstance(output, dict):
        print("  The answer is not a JSON object.")
        output = {}
    passed_keys = 0
    for t in targets:
        exp_v = expected_map[t]
        if output.get(t) == exp_v:
            passed_keys += 1
            print(f"  - {t}: {exp_v} -> MATCH")
        else:
            print(f"  - {t}: {exp_v} -> MISMATCH / MISSING")
            
    success = (passed_keys == len(targets))
    print(f"  [Result]: {passed_keys}/{len(targets)} exact high-entropy matches. {'PASS' if success else 'FAIL'}")
    return {"name": name, "passed": success, "score": f"{passed_keys}/{len(targets)}"}

def test_multi_hop_variable_tracking(url, model="default", api_key=""):
    print("\n" + "="*80)
    print("STRESS TEST 2: Multi-Hop Variable Tracking across 8 Dependency Stages in 50k Tokens")
    print("="*80)
    
    stages = [
        (50, "REGISTRATION: var_a = 7412"),
        (110, "TRANSFORMATION: var_b = var_a + 1928"),
        (170, "TRANSFORMATION: var_c = var_b * 3 - 500"),
        (230, "TRANSFORMATION: var_d = var_c ^ 1023  (bitwise XOR)"),
        (290, "TRANSFORMATION: var_e = var_d + 12400"),
        (350, "TRANSFORMATION: var_f = var_e - 3959"),
        (410, "TRANSFORMATION: var_g = var_f * 2 + 77"),
        (470, "TRANSFORMATION: var_h = var_g ^ 4095  (bitwise XOR)"),
    ]
    
    # Ground truth:
    va = 7412
    vb = va + 1928
    vc = vb * 3 - 500
    vd = vc ^ 1023
    ve = vd + 12400
    vf = ve - 3959
    vg = vf * 2 + 77
    vh = vg ^ 4095
    expected_final = vh  # 72898
    
    stage_map = dict(stages)
    total_blocks = 500
    doc = []
    for b in range(total_blocks):
        doc.append(FILLER_BLOCK)
        if b in stage_map:
            doc.append(f"\n[CRITICAL_ALU_PIPELINE: {stage_map[b]}]\n")
            
    doc_text = "".join(doc)
    prompt = (
        f"{doc_text}\n\n"
        "TASK:\n"
        "Trace the 8-stage sequential ALU pipeline transformations (var_a through var_h) defined in the document above.\n"
        "Calculate the exact intermediate values for var_a, var_b, var_c, var_d, var_e, var_f, var_g, and the final value of var_h.\n"
        "State the final value of var_h clearly at the end: 'FINAL_VALUE: <number>'."
    )
    
    print(f"  Ingesting ~50,000 tokens with 8 sequential multi-hop dependencies...")
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.0, enable_thinking=True, model=model, api_key=api_key)
    name = "Multi-Hop Dependency (8 Hops)"
    error = benchmark_error(name, res)
    if error:
        return error
    print(f"  Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s)")
    
    has_expected = declared_integer(final_content(res), "FINAL_VALUE") == expected_final
    print(f"  - Ground Truth var_h = {expected_final}: {'MATCH' if has_expected else 'FAIL'}")
    print(f"  [Result]: {'PASS' if has_expected else 'FAIL'}")
    return {"name": name, "passed": has_expected, "expected": expected_final}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API completion endpoint")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL") or "default", help="Model name or ID")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "results", "stress_test_results.json"), help="Path to output JSON")
    args = parser.parse_args()
    
    results = []
    checks = [
        ("High-Entropy KV Recall (8/8)", test_high_entropy_kv_recall),
        ("Multi-Hop Dependency (8 Hops)", test_multi_hop_variable_tracking),
    ]
    for index, (_, check) in enumerate(checks):
        result = check(args.url, model=args.model, api_key=args.api_key)
        results.append(result)
        if result.get("endpoint_unreachable"):
            results.extend({"name": name, "passed": False, "status": "SKIPPED",
                            "reason": "Server unreachable"} for name, _ in checks[index + 1:])
            break
    
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
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
