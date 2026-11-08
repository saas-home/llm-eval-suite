#!/usr/bin/env python3
"""
Enterprise Hardened Adversarial, Reasoning & Agentic Benchmark Suite
(tests/scripts/test_adversarial_agent.py)

Specifically calibrated for advanced local LLMs (such as Qwen3.8-27B) running on
quantized KV caches (e.g. EXL3 with 4-bit Key / 3-bit Value compression).

Evaluates 5 Hardening Pillars:
1. Multi-Hop Graph Traversal with Adversarial Distractors & Version Deprecation (~100k tokens)
2. Novel Algorithmic Code Synthesis & Automated 5,000-Operation Property-Based Fuzz Testing
3. Combinatorial Anti-Constraint & Negative Constraint Adherence (IFEval Tier)
4. Counterfactual Axiomatic Symbolic Algebra with Non-Commutative Modular Operators
5. Extreme-Depth Frontier Needle Synthesis (0.5%, 50%, and 99.5% Context Horizons)
"""

import sys
import os
import time
import json
import re
import random
import subprocess
import ast
import argparse

# Shared terminal formatting from the parent package (respects NO_COLOR / non-TTY).
_BENCH_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCH_PARENT not in sys.path:
    sys.path.insert(0, _BENCH_PARENT)
from client import (  # noqa: E402
    CYAN, GREEN, MAGENTA, RED, RESET, YELLOW, LLMClient, assemble_python_script,
    extract_python_code, harness_passed, log, run_model_code,
)
from benchmarks.common import benchmark_error, benchmark_response, declared_integer as _declared_integer  # noqa: E402
from domain_benchmarks import final_content  # noqa: E402
from eval import grade_multihop_answer  # noqa: E402


class BenchmarkClient:
    def __init__(self, endpoint: str, model: str, api_key: str = ""):
        self.client = LLMClient(endpoint, api_key=api_key, model=model)

    def call(self, messages, max_tokens=2048, temperature=0.7, stream=True,
             tools=None, tool_choice=None, timeout=600):
        return benchmark_response(self.client.call(
            messages, max_tokens=max_tokens, temperature=temperature, stream=stream,
            tools=tools, tool_choice=tool_choice, timeout=timeout,
        ))


# ============================================================================
# PILLAR 1: MULTI-HOP GRAPH TRAVERSAL WITH ADVERSARIAL DISTRACTORS (100k TOKENS)
# ============================================================================
def test_multihop_graph_distractors(client: BenchmarkClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 1/5] Multi-Hop Graph Traversal with Versioned Temporal Distractors (~100k Context)", bold=True, color=CYAN)
    log("="*88)

    filler_chunk = (
        "Enterprise cloud transit routing manages BGP autonomous system interconnects across software-defined WAN fabrics. "
        "Packet encapsulation leverages Geneve and VXLAN tunnels terminating on distributed SmartNIC vSwitch datanodes. "
        "Telemetry pipelines process streaming flow logs through Kafka clusters with partitioned RocksDB state stores. "
        "Zero-trust microsegmentation evaluates SPIFFE identity assertions and mutual TLS session keys at every ingress hop. "
    ) # ~60 tokens

    total_blocks = 1600 # ~96,000 tokens
    
    # Place adversarial distractors (older, deprecated routing policies)
    distractor_v1 = (
        "\n[ARCHITECTURAL_ROUTING_SPECIFICATION v1.0 (2024 - DEPRECATED)]\n"
        "Policy Rule 401: In the event of Primary Switch 7 outage, traffic from Client Mobile Edge routes through:\n"
        "Node A -> Gateway Chicago -> Database Shard Gamma (Read-Only Replica 1).\n"
        "Security Note: Route 401 was revoked by RFC-77 due to cross-datacenter latency violations.\n"
    )
    distractor_v2 = (
        "\n[ARCHITECTURAL_ROUTING_SPECIFICATION v2.2 (2025 - REVOKED / ROLLBACK)]\n"
        "Policy Rule 402: Under Switch 7 hardware failure, failover path is:\n"
        "Node A -> Gateway Frankfurt -> Database Shard Gamma (Staging Cluster).\n"
        "Audit Flag: Revoked after disaster recovery drill failure.\n"
    )
    # The active production specification
    active_v3 = (
        "\n[ARCHITECTURAL_ROUTING_SPECIFICATION v3.1 (2026 - ACTIVE PRODUCTION APPROVED)]\n"
        "Policy Rule 403: When Primary Switch 7 fails, the mandatory active failover sequence is:\n"
        "Client Mobile Edge -> Ingress Node Alpha -> Transit Hub Tokyo -> Secure Bridge Omega -> Database Shard Gamma (Active Primary).\n"
        "Mandatory Verification Code: AUTH_ROUTING_KEY_9921_TOK.\n"
        "Failover SLA: Must terminate within 45ms over dedicated dark fiber link.\n"
    )

    doc_parts = []
    for b in range(total_blocks):
        doc_parts.append(filler_chunk)
        if b == int(total_blocks * 0.15):
            doc_parts.append(distractor_v1)
        elif b == int(total_blocks * 0.50):
            doc_parts.append(distractor_v2)
        elif b == int(total_blocks * 0.85):
            doc_parts.append(active_v3)

    document = "".join(doc_parts)
    prompt = (
        f"{document}\n\n"
        "MISSION-CRITICAL ARCHITECTURAL INCIDENT ANALYSIS:\n"
        "A critical hardware outage occurs on Primary Switch 7. Review the ARCHITECTURAL_ROUTING_SPECIFICATION sections above and determine:\n"
        "1. What is the current, active, and approved failover routing path from Client Mobile Edge to Database Shard Gamma?\n"
        "2. State the exact Mandatory Verification Code for the active route.\n"
        "3. Explain why the previous v1.0 (Chicago) and v2.2 (Frankfurt) routes are invalid and must not be used.\n\n"
        "Conclude your response strictly with: 'ACTIVE_FAILOVER_CODE: <verification code>'."
    )

    log(f"  Ingesting ~{total_blocks * 60:,}-token context with 2 temporal distractors and 1 active specification...")
    name = "Multi-Hop Graph & Adversarial Distractors"
    res = client.call([{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.0, stream=True)
    error = benchmark_error(name, res)
    if error:
        return error
    text = final_content(res)

    route = grade_multihop_answer(text)
    has_active_tokyo = route["route_in_order"]
    has_code = route["declared_code_correct"]
    has_rejected_chicago = ("chicago" in text.lower()) and ("deprecated" in text.lower() or "rfc-77" in text.lower() or "revoked" in text.lower() or "invalid" in text.lower())
    has_rejected_frankfurt = ("frankfurt" in text.lower()) and ("revoked" in text.lower() or "rollback" in text.lower() or "invalid" in text.lower())

    passed = has_active_tokyo and has_code and has_rejected_chicago and has_rejected_frankfurt
    log(f"  - Active Route Identification (Tokyo/Omega): {'PASS' if has_active_tokyo else 'FAIL'}")
    log(f"  - Exact Verification Key Recall: {'PASS' if has_code else 'FAIL'}")
    log(f"  - Rejection of Deprecated v1.0 Chicago Distractor: {'PASS' if has_rejected_chicago else 'FAIL'}")
    log(f"  - Rejection of Revoked v2.2 Frankfurt Distractor: {'PASS' if has_rejected_frankfurt else 'FAIL'}")
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | TTFT: {res['ttft_ms']:.1f} ms | Status: {'PASS' if passed else 'FAIL'}")

    return {
        "name": name,
        "passed": passed,
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"],
        "ttft_ms": round(res["ttft_ms"], 1),
        "tok_s": round(res["tok_per_sec"], 2)
    }


# ============================================================================
# PILLAR 2: NOVEL ALGORITHMIC SYNTHESIS WITH 5,000-OP PROPERTY FUZZ TEST
# ============================================================================
def test_novel_algorithmic_fuzz(client: BenchmarkClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 2/5] Novel Algorithmic Synthesis & Automated 5,000-Op Property Fuzz Testing", bold=True, color=CYAN)
    log("="*88)

    prompt = (
        "Implement a custom, non-standard Python class named `ConcurrentMonotonicRingBuffer`.\n"
        "This data structure must fulfill the following exact specifications:\n"
        "1. Constructor: `__init__(self, capacity: int)`: Initializes buffer with positive fixed integer capacity.\n"
        "2. Method: `push(self, value: int) -> int`: Adds value to the ring buffer. If buffer is full, it overwrites "
        "the oldest element. Returns the monotonically increasing 64-bit sequence ID assigned to this item (starting at sequence ID 1).\n"
        "3. Method: `get_by_seq(self, seq_id: int) -> int`: Returns the value associated with `seq_id`. If `seq_id` has already been "
        "overwritten or has not yet been pushed, raises `KeyError`.\n"
        "4. Method: `get_latest(self) -> tuple[int, int]`: Returns `(latest_seq_id, latest_value)`. If buffer is empty, raises `IndexError`.\n"
        "5. Method: `snapshot_in_order(self) -> list[tuple[int, int]]`: Returns list of all current `(seq_id, value)` pairs currently stored, "
        "ordered from oldest to newest.\n"
        "6. Provide self-contained Python code inside a ```python ``` code block. Do NOT use third-party libraries. "
        "Keep comments and docstrings concise to emit complete implementation code."
    )

    log("  Requesting synthesis of non-standard Monotonic Ring Buffer...")
    name = "Novel Algorithmic Synthesis (5k Fuzz)"
    res = client.call([{"role": "user", "content": prompt}], max_tokens=3500, temperature=0.2, stream=True)
    error = benchmark_error(name, res)
    if error:
        return error
    raw = final_content(res)

    # Extract code
    code = extract_python_code(raw, prefer_class="class ConcurrentMonotonicRingBuffer")

    # Property-based fuzz verification harness
    fuzz_harness = """
import sys
import random

def run_fuzz_verification():
    # Phase 1: Basic validation
    buf = ConcurrentMonotonicRingBuffer(3)
    s1 = buf.push(10)
    s2 = buf.push(20)
    s3 = buf.push(30)
    assert (s1, s2, s3) == (1, 2, 3), f"Sequences must be 1, 2, 3, got {(s1, s2, s3)}"
    assert buf.get_by_seq(1) == 10
    assert buf.get_by_seq(2) == 20
    assert buf.get_by_seq(3) == 30
    assert buf.get_latest() == (3, 30)

    # Overwrite oldest (1 is evicted)
    s4 = buf.push(40)
    assert s4 == 4
    try:
        buf.get_by_seq(1)
        assert False, "Seq 1 should have raised KeyError after eviction"
    except KeyError:
        pass
    assert buf.get_by_seq(2) == 20
    assert buf.get_by_seq(4) == 40
    assert buf.snapshot_in_order() == [(2, 20), (3, 30), (4, 40)]

    # Phase 2: 5,000 randomized property-based operations
    cap = 50
    rb = ConcurrentMonotonicRingBuffer(cap)
    ground_truth = {}
    current_seq = 0

    random.seed(42)
    for op in range(5000):
        val = random.randint(1, 1000000)
        current_seq += 1
        seq_ret = rb.push(val)
        assert seq_ret == current_seq, f"Sequence drift at op {op}: expected {current_seq}, got {seq_ret}"
        ground_truth[current_seq] = val

        # Clean oldest in ground truth if capacity exceeded
        oldest_valid_seq = max(1, current_seq - cap + 1)
        if (current_seq - cap) in ground_truth:
            del ground_truth[current_seq - cap]

        # Intermittent random queries
        if op % 10 == 0:
            assert rb.get_latest() == (current_seq, val)
            # Query random valid key
            q_seq = random.randint(oldest_valid_seq, current_seq)
            assert rb.get_by_seq(q_seq) == ground_truth[q_seq]

            # Query evicted key
            if oldest_valid_seq > 1:
                evicted_seq = random.randint(1, oldest_valid_seq - 1)
                try:
                    rb.get_by_seq(evicted_seq)
                    assert False, f"Evicted sequence {evicted_seq} did not raise KeyError"
                except KeyError:
                    pass

        # Intermittent snapshot check
        if op % 250 == 0:
            snap = rb.snapshot_in_order()
            expected_snap = sorted(ground_truth.items(), key=lambda x: x[0])
            assert snap == expected_snap, f"Snapshot mismatch at op {op}: {snap} vs {expected_snap}"

    print("ALL_5000_FUZZ_TESTS_PASSED", __import__('os').environ['EVAL_RUN_ID'])

if __name__ == "__main__":
    run_fuzz_verification()
"""
    import_preamble = "import threading\nfrom collections import deque, OrderedDict\nimport sys\nimport random\nimport time"
    full_script = assemble_python_script(code, preamble=import_preamble, harness=fuzz_harness)
    fuzz_passed = False
    syntax_only = False
    err_msg = ""
    if not code.strip():
        err_msg = "No ConcurrentMonotonicRingBuffer code found."
    elif os.getenv("EVAL_SAFE", "").strip().lower() in ("1", "true", "yes"):
        try:
            tree = ast.parse(full_script)
            syntax_only = any(isinstance(node, ast.ClassDef) and node.name == "ConcurrentMonotonicRingBuffer"
                              for node in tree.body)
            err_msg = ("Execution disabled; syntax check only." if syntax_only
                       else "No ConcurrentMonotonicRingBuffer class found.")
        except (SyntaxError, ValueError) as e:
            err_msg = f"SyntaxError: {e}"
    else:
        try:
            proc = run_model_code([sys.executable, "-c", full_script], 20)
            fuzz_passed = harness_passed(proc, "ALL_5000_FUZZ_TESTS_PASSED")
            if not fuzz_passed:
                err_msg = (proc.stderr or proc.stdout).strip()[:300] or "Fuzz verification did not complete."
        except (OSError, subprocess.TimeoutExpired) as e:
            err_msg = str(e)

    status = "PASS" if fuzz_passed else ("PARTIAL" if syntax_only else "FAIL")
    log(f"  - Automated 5,000 Operations Fuzz Assertions: {'not run' if syntax_only else status}")
    if err_msg:
        log(f"    [{'NOTE' if syntax_only else 'Error'}]: {err_msg}", color=YELLOW if syntax_only else RED)
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | Status: {status}")

    return {
        "name": name,
        "passed": fuzz_passed,
        "status": status,
        "completion_tokens": res["completion_tokens"],
        "tok_s": round(res["tok_per_sec"], 2),
        "error": err_msg
    }


# ============================================================================
# PILLAR 3: COMBINATORIAL ANTI-CONSTRAINT FOLLOWING (IFEVAL TIER)
# ============================================================================
def _markdown_headings(text):
    headings = []
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        elif fence is None and re.match(r"^#{1,6}(?:\s+|$)", stripped):
            headings.append((stripped, offset, offset + len(line)))
        offset += len(line)
    return headings


def _yaml_mapping_verdict(text, expected):
    """Verify simple scalar mappings; return None for YAML we cannot validate."""
    keys = set()
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)[ \t]*:[ \t]*(.*)", line)
        if not match:
            return None
        key, value = match.groups()
        if key in keys:
            return False
        keys.add(key)
        value = value.strip()
        if value.startswith('"'):
            try:
                if not isinstance(json.loads(value), str):
                    return None
            except json.JSONDecodeError:
                return None
        elif value.startswith("'"):
            if not re.fullmatch(r"'(?:[^']|'')*'", value):
                return None
        else:
            value = value.split(" #", 1)[0].strip()
            if value and not re.fullmatch(r"(?:[A-Za-z0-9_./][A-Za-z0-9_./+%-]*|[+-]\d+(?:\.\d+)?)", value):
                return None
    return keys == expected


def test_ifeval_combinatorial_constraints(client: BenchmarkClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 3/5] Combinatorial Anti-Constraint & Negative Instruction Following (IFEval Tier)", bold=True, color=CYAN)
    log("="*88)

    prompt = (
        "Write an architectural briefing on 'Zero-Trust Kernel Isolation'.\n"
        "You must strictly satisfy all 6 of the following constraints simultaneously:\n\n"
        "Constraint 1: Structure your answer into exactly 4 markdown sections headed by:\n"
        "   '### Section I: Foundation'\n"
        "   '### Section II: Isolation Proof'\n"
        "   '### Section III: Hardware Enclaves'\n"
        "   '### Section IV: Infrastructure Configuration'\n\n"
        "Constraint 2 (Lipogram): In '### Section II: Isolation Proof', you must NEVER use the letter 'e' (neither lowercase 'e' nor uppercase 'E'). Every single word in Section II must be completely free of 'e'.\n\n"
        "Constraint 3 (Word Count): '### Section III: Hardware Enclaves' must be strictly between 90 and 130 words in length.\n\n"
        "Constraint 4 (YAML Embedding): '### Section IV: Infrastructure Configuration' must contain a valid ```yaml ``` code block with exactly 4 keys: `enclave_id`, `page_isolation`, `ring_buffer_mb`, `attestation_pki`.\n\n"
        "Constraint 5 (Negative Anti-Tokens): Do NOT use any of the following 4 words anywhere in your entire output: 'constraint', 'rule', 'forbidden', 'negative'.\n\n"
        "Constraint 6 (Termination): Your final output line must be strictly: '[SECURITY_STAMP_2026_VERIFIED]'."
    )

    log("  Dispatching multi-constraint prompt with lipogram, word counts, and forbidden tokens...")
    name = "Combinatorial Anti-Constraints (IFEval)"
    res = client.call([{"role": "user", "content": prompt}], max_tokens=3500, temperature=0.4, stream=True)
    error = benchmark_error(name, res)
    if error:
        return error
    text = final_content(res)

    # Constraint 1: 4 sections
    headings = _markdown_headings(text)
    expected_headings = ["### Section I: Foundation", "### Section II: Isolation Proof",
                         "### Section III: Hardware Enclaves", "### Section IV: Infrastructure Configuration"]
    c1 = [heading[0] for heading in headings] == expected_headings

    # Constraint 2: Lipogram in Section II (No 'e' or 'E')
    sec2_text = text[headings[1][2]:headings[2][1]].strip() if c1 else ""
    c2 = bool(sec2_text) and "e" not in sec2_text.lower()

    # Constraint 3: Word count in Section III (strictly between 90 and 130)
    sec3_text = text[headings[2][2]:headings[3][1]] if c1 else ""
    sec3_words = len(sec3_text.split())
    c3 = 90 < sec3_words < 130

    # Constraint 4: YAML block in Section IV with exact 4 keys
    sec4_text = text[headings[3][2]:] if c1 else ""
    blocks = re.findall(r"(?ims)^```yaml[ \t]*\r?\n(.*?)^```[ \t]*\r?$", sec4_text)
    expected_keys = {"enclave_id", "page_isolation", "ring_buffer_mb", "attestation_pki"}
    yaml_checks = [_yaml_mapping_verdict(block, expected_keys) for block in blocks]
    c4 = True if True in yaml_checks else (None if None in yaml_checks else False)

    # Constraint 5: Forbidden anti-tokens
    forbidden = ["constraint", "rule", "forbidden", "negative"]
    found_forbidden = [f for f in forbidden if re.search(r"\b" + re.escape(f) + r"\b", text, re.IGNORECASE)]
    c5 = len(found_forbidden) == 0

    # Constraint 6: Termination Stamp
    c6 = bool(text) and text.splitlines()[-1].strip() == "[SECURITY_STAMP_2026_VERIFIED]"

    all_passed = c1 and c2 and c3 and c4 is True and c5 and c6
    status = "PASS" if all_passed else ("PARTIAL" if c4 is None and c1 and c2 and c3 and c5 and c6 else "FAIL")
    log(f"  - C1 (Exact 4 Required Markdown Sections): {'PASS' if c1 else 'FAIL'}")
    log(f"  - C2 (Section II Lipogram: ZERO 'e'/'E' in text): {'PASS' if c2 else 'FAIL'} (Chars: {len(sec2_text)})")
    log(f"  - C3 (Section III Word Count between 90 and 130 words): {'PASS' if c3 else 'FAIL'} (Count: {sec3_words})")
    log(f"  - C4 (Section IV YAML with 4 Exact Keys): {'PASS' if c4 is True else 'unverified' if c4 is None else 'FAIL'}")
    log(f"  - C5 (Zero Forbidden Anti-Tokens): {'PASS' if c5 else 'FAIL'} (Found: {found_forbidden})")
    log(f"  - C6 (Exact Security Stamp Termination): {'PASS' if c6 else 'FAIL'}")
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | Status: {status}")

    return {
        "name": name,
        "passed": all_passed,
        "status": status,
        "completion_tokens": res["completion_tokens"],
        "tok_s": round(res["tok_per_sec"], 2),
        "constraint_details": {
            "4_sections": c1,
            "lipogram_no_e": c2,
            "word_count_90_130": c3,
            "yaml_keys": c4,
            "no_forbidden_words": c5,
            "termination_stamp": c6
        }
    }


# ============================================================================
# PILLAR 4: COUNTERFACTUAL AXIOMATIC SYMBOLIC ALGEBRA
# ============================================================================
def test_counterfactual_axiomatic_math(client: BenchmarkClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 4/5] Counterfactual Axiomatic Symbolic Deduction (Non-Commutative Modular Algebra)", bold=True, color=CYAN)
    log("="*88)

    expected_solutions = [8, 17]

    prompt = (
        "Consider a custom finite-field mathematical system over the ring Z_23 (integers modulo 23, values 0 to 22) "
        "defined by two synthetic binary operations:\n\n"
        "1. Operation (+):  a (+) b = (3*a - 2*b + 5) mod 23\n"
        "2. Operation (*):  a (*) b = (a^2 + b + 2) mod 23\n\n"
        "Solve the following equation for all integer solutions X in the range [0, 22]:\n"
        "   ( (X (+) 4) (*) 3 ) (+) 8 = 16   (mod 23)\n\n"
        "Requirements:\n"
        "- Show the step-by-step reduction for each nested operation using modular arithmetic.\n"
        "- Compute the modular multiplicative inverse of 3 mod 23.\n"
        "- Identify all valid integer values of X in {0, ..., 22}.\n"
        "- State at the very end: 'FINAL_SOLUTIONS_FOR_X: [values]'."
    )

    log("  Requesting step-by-step resolution of custom non-commutative modular algebra...")
    name = "Counterfactual Axiomatic Algebra"
    res = client.call([{"role": "user", "content": prompt}], max_tokens=1800, temperature=0.0, stream=True)
    error = benchmark_error(name, res)
    if error:
        return error
    text = final_content(res)
    declarations = re.findall(r"(?im)^[ \t>*_-]*[`*_]*FINAL_SOLUTIONS_FOR_X[`*_]*[ \t]*:[ \t]*(.*)$", text)
    solutions = []
    if declarations:
        declared = declarations[-1].strip().strip("`*").strip()
        if re.fullmatch(r"\[[ \t]*[+-]?\d+(?:[ \t]*,[ \t]*[+-]?\d+)*[ \t]*\]", declared):
            solutions = [int(value.strip()) for value in declared[1:-1].split(",")]

    has_inv = bool(re.search(r"\binverse[^\n]*(?<!\d)8(?!\d)", text, re.IGNORECASE))
    has_8 = 8 in solutions
    has_17 = 17 in solutions
    matched = sorted(solutions) == expected_solutions

    log(f"  - Modular Multiplicative Inverse of 3 mod 23 (=8): {'PASS' if has_inv else 'FAIL'}")
    log(f"  - Solved First Root X=8: {'PASS' if has_8 else 'FAIL'}")
    log(f"  - Solved Second Root X=17: {'PASS' if has_17 else 'FAIL'}")
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | Status: {'PASS' if matched else 'FAIL'}")

    return {
        "name": name,
        "passed": matched,
        "completion_tokens": res["completion_tokens"],
        "tok_s": round(res["tok_per_sec"], 2)
    }


# ============================================================================
# PILLAR 5: FRONTIER DEPTH PRECISION (0.5%, 50%, 99.5% CONTEXT HORIZON)
# ============================================================================
def test_frontier_depth_precision(client: BenchmarkClient, target_context=120000):
    log("\n" + "="*88, bold=True)
    log(f"[TEST 5/5] Extreme Frontier-Depth Multi-Needle Precision ({target_context:,} Context Window)", bold=True, color=CYAN)
    log("="*88)

    filler_segment = (
        "Distributed consensus engines like Raft and Paxos serialize concurrent log entries using multi-version concurrency control. "
        "Storage backends distribute page tables across high-performance non-volatile media with atomic flush primitives. "
        "Analytical engines parse abstract syntax trees into vectorized SIMD instructions over columnar Apache Arrow memory layouts. "
    ) # ~45 tokens

    blocks = max(100, int(target_context / 45))
    val_front = random.randint(1000, 9999)
    val_middle = 7392
    val_tail = 3105

    pos_front = int(blocks * 0.005) # 0.5% depth
    pos_middle = int(blocks * 0.500) # 50.0% depth
    pos_tail = int(blocks * 0.995)   # 99.5% depth (extreme horizon)

    needle_front = f"\n[CRITICAL_FRONT_REGISTER: ALPHA_REG = {val_front}]\n"
    needle_middle = f"\n[CRITICAL_MIDDLE_REGISTER: BETA_REG = {val_middle}]\n"
    needle_tail = f"\n[CRITICAL_TAIL_REGISTER: GAMMA_REG = {val_tail}]\n"

    expected_composite = (val_front + val_tail) - val_middle # (4821 + 3105) - 7392 = 534

    doc = []
    for b in range(blocks):
        doc.append(filler_segment)
        if b == pos_front:
            doc.append(needle_front)
        elif b == pos_middle:
            doc.append(needle_middle)
        elif b == pos_tail:
            doc.append(needle_tail)

    document = "".join(doc)
    prompt = (
        f"{document}\n\n"
        "TASK:\n"
        "Locate all three critical registers buried in the documentation above:\n"
        "1. ALPHA_REG (near the beginning of the context)\n"
        "2. BETA_REG (in the middle of the context)\n"
        "3. GAMMA_REG (at the extreme end of the context)\n\n"
        "Perform the exact calculation: COMPOSITE_CHECKSUM = (ALPHA_REG + GAMMA_REG) - BETA_REG.\n"
        "State the values of all three registers and the final checksum. Conclude with 'COMPOSITE_CHECKSUM = <integer>'."
    )

    log(f"  Ingesting ~{target_context:,} tokens with needles at 0.5%, 50.0%, and 99.5% depth...")
    name = f"Frontier Needle Horizon ({target_context//1000}k)"
    res = client.call([{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.0, stream=True)
    error = benchmark_error(name, res)
    if error:
        return error
    text = final_content(res)

    m1 = _declared_integer(text, "ALPHA_REG") == val_front
    m2 = _declared_integer(text, "BETA_REG") == val_middle
    m3 = _declared_integer(text, "GAMMA_REG") == val_tail
    m_calc = _declared_integer(text, "COMPOSITE_CHECKSUM") == expected_composite
    passed = m1 and m2 and m3 and m_calc

    log(f"  - Extreme Front Needle (0.5% depth, ALPHA={val_front}): {'PASS' if m1 else 'FAIL'}")
    log(f"  - Mid-Context Needle (50.0% depth, BETA={val_middle}): {'PASS' if m2 else 'FAIL'}")
    log(f"  - Extreme Horizon Tail Needle (99.5% depth, GAMMA={val_tail}): {'PASS' if m3 else 'FAIL'}")
    log(f"  - Composite Calculation ({val_front} + {val_tail} - {val_middle} = {expected_composite}): {'PASS' if m_calc else 'FAIL'}")
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | TTFT: {res['ttft_ms']:.1f} ms | Status: {'PASS' if passed else 'FAIL'}")

    return {
        "name": name,
        "passed": passed,
        "completion_tokens": res["completion_tokens"],
        "ttft_ms": round(res["ttft_ms"], 1),
        "tok_s": round(res["tok_per_sec"], 2)
    }


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
def main():
    default_ep = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT") or "http://127.0.0.1:8000/v1"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    default_out = os.path.join(root_dir, "results", "hardened_adversarial_results.json")

    parser = argparse.ArgumentParser(description="Enterprise Hardened Adversarial & Agentic Benchmark.")
    parser.add_argument("--endpoint", "-e", default=default_ep, help="Server endpoint")
    parser.add_argument("--model", "-m", default=os.getenv("OPENAI_MODEL") or "default", help="Model name or ID")
    parser.add_argument("--api-key", "-k", default=os.getenv("OPENAI_API_KEY", ""), help="API Key")
    parser.add_argument("--context-depth", default=32000, type=int, help="Target context depth for frontier test (default: 32000)")
    parser.add_argument("--out", "-o", default=default_out, help="Report output path")
    args = parser.parse_args()

    log("\n" + "="*88, bold=True)
    log("  ENTERPRISE HARDENED ADVERSARIAL & AGENTIC BENCHMARK SUITE", bold=True, color=MAGENTA)
    log(f"  Target Endpoint: {args.endpoint}")
    log(f"  Model ID       : {args.model}")
    log("="*88)

    client = BenchmarkClient(args.endpoint, args.model, args.api_key)

    results = []
    t_start = time.perf_counter()
    checks = [
        ("Multi-Hop Graph & Adversarial Distractors", test_multihop_graph_distractors, {}),
        ("Novel Algorithmic Synthesis (5k Fuzz)", test_novel_algorithmic_fuzz, {}),
        ("Combinatorial Anti-Constraints (IFEval)", test_ifeval_combinatorial_constraints, {}),
        ("Counterfactual Axiomatic Algebra", test_counterfactual_axiomatic_math, {}),
        (f"Frontier Needle Horizon ({args.context_depth//1000}k)", test_frontier_depth_precision,
         {"target_context": args.context_depth}),
    ]
    for _, check, kwargs in checks:
        result = check(client, **kwargs)
        results.append(result)
        if result.get("endpoint_unreachable"):
            break

    total_wall_time = time.perf_counter() - t_start

    log("\n" + "="*88, bold=True)
    log("                   HARDENED BENCHMARK SCORECARD", bold=True)
    log("="*88)
    log(f"{'Evaluation Domain':<50} | {'Status':<10} | {'Speed / Metrics':<20}")
    log("-" * 88)

    all_passed = True
    any_failed = False
    incomplete = False
    for r in results:
        status_text = r.get("status", "PASS" if r["passed"] else "FAIL")
        status_color = GREEN if status_text == "PASS" else (YELLOW if status_text == "PARTIAL" else RED)
        if not r["passed"]:
            all_passed = False
        if status_text == "FAIL":
            any_failed = True
        if status_text == "ERROR":
            incomplete = True
        log(f"{r['name']:<50} | {status_color}{status_text:<10}{RESET} | {r.get('tok_s', 0):.2f} tok/s")

    log("="*88)
    verdict = (f"{YELLOW}INCOMPLETE: REQUEST ERROR{RESET}" if incomplete else
               f"{GREEN}100% HARDENED & QUALIFIED{RESET}" if all_passed else
               f"{RED}DEFECTS / QUALIFICATION FAILURES DETECTED{RESET}" if any_failed else
               f"{YELLOW}UNVERIFIED: SOME CHECKS NOT COMPLETED{RESET}")
    log(f"Final Qualification Verdict: {verdict}")
    log(f"Total Wall Time            : {total_wall_time:.2f} s")

    # Export report
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint": args.endpoint,
        "model": args.model,
        "total_wall_time_s": round(total_wall_time, 2),
        "all_passed": all_passed,
        "incomplete": incomplete,
        "not_run": [name for name, _, _ in checks[len(results):]],
        "results": results
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Detailed Report Exported   : {args.out}\n")
    if incomplete:
        sys.exit(3)


if __name__ == "__main__":
    main()
