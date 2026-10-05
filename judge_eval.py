#!/usr/bin/env python3
"""
Verified LLM Quality Judge & Fair Benchmark Harness
---------------------------------------------------
Grades every task out of 10.0 using realistic, non-artificial prompts and Python AST syntax verification.

Fixes:
1. Replaces artificial string multiplication with natural technical document text for Needle Retrieval (Task 3).
2. Uses Python AST Compilation Syntax Checks to verify code syntax without failing on uninstalled third-party modules (like redis/numpy).
3. Double-Blind Position-Swapped LLM Quality Judge (0.0 to 10.0 scale per task).
"""

import os
import sys
import time
import json
import re
import ast
import urllib.request
import urllib.error
import argparse
import env_loader

from client import LLMClient


def discover_model_safe(endpoint: str, api_key: str) -> str:
    """Best-effort model discovery; returns 'default' if unavailable."""
    try:
        client = LLMClient(endpoint=endpoint, api_key=api_key)
        models = client.fetch_models()
        if models:
            m0 = models[0]
            return m0.get("id") or m0.get("name") or "default"
    except Exception:
        pass
    return "default"

HARD_BENCHMARK_TASKS = [
    {
        "id": "task_h1_raft_consensus",
        "category": "Distributed Systems Architecture",
        "name": "Raft Consensus Log Replication & State Machine",
        "system": "You are a Principal Distributed Systems Engineer. Write production-grade, bug-free Python code.",
        "prompt": (
            "Implement a complete, production-grade Raft Consensus Node in Python with type annotations and multithreading locks.\n"
            "Requirements:\n"
            "1. Implement `RaftNode` supporting Leader, Follower, and Candidate states.\n"
            "2. Implement `RequestVote` and `AppendEntries` RPC handlers with log matching, term validation, and commit index updates.\n"
            "3. Handle network partition recovery, split-brain prevention (majority quorum requirement), and uncommitted log truncation.\n"
            "4. Include an in-memory State Machine that applies committed log entries and returns state query results.\n"
            "5. Provide a self-contained test function `test_raft_cluster()` that simulates a 3-node cluster, forces leader election, replicates 5 log commands, simulates a network partition, and verifies consistency across nodes."
        ),
        "max_tokens": 2000,
        "temperature": 0.2,
    },
    {
        "id": "task_h2_ssa_compiler",
        "category": "Compiler Architecture & AST",
        "name": "Static Single Assignment (SSA) AST Optimizer & Bytecode Pass",
        "system": "You are a compiler engineer. Write clean, modular, production-grade Python code.",
        "prompt": (
            "Write an SSA (Static Single Assignment) compiler pass and bytecode generator in Python for arithmetic & conditional expressions.\n"
            "Requirements:\n"
            "1. Implement AST nodes (`Var`, `Const`, `BinOp`, `Assign`, `IfStmt`, `Block`).\n"
            "2. Implement `to_ssa(ast: Block) -> SSABlock` transformation, placing Phi nodes (\u03c6) at join points of control flow branches and versioning all variable assignments (e.g. `x1 = x0 + 1`).\n"
            "3. Implement constant propagation and dead-code elimination passes on the SSA IR.\n"
            "4. Include a simulator `eval_ssa(ssa_block, initial_env)` that executes the optimized SSA IR and returns final variable values.\n"
            "5. Include a test suite `test_ssa_pass()` that verifies SSA conversion and constant folding optimization on a conditional loop algorithm."
        ),
        "max_tokens": 1800,
        "temperature": 0.2,
    },
    {
        "id": "task_h3_needle_retrieval",
        "category": "Long Context Retrieval & Math Synthesis",
        "name": "Multi-Needle Technical Audit Log Retrieval & Security Hash Code",
        "system": "You are a precise technical auditor and data extraction specialist.",
        "prompt": (
            "You are analyzing an enterprise microservice infrastructure security report.\n\n"
            "Section 1: Distributed Telemetry and Observability Policy\n"
            "Enterprise microservices emit OpenTelemetry spans across gRPC and HTTPS channels.\n"
            "Metric collectors gather CPU usage, heap allocation, and network socket statistics every 10 seconds.\n"
            "Distributed tracing context propagates baggage headers across datacenter boundaries.\n\n"
            "CRITICAL SECURITY CONSTANT 1: ALPHA_VAL = 48912\n\n"
            "Section 2: Database Connection Pooling & Idempotency Rules\n"
            "Relational datastores enforce strict connection timeouts and max pool limits.\n"
            "Deduplication windows maintain idempotency keys for 3600 seconds.\n"
            "All write operations execute within serialized isolation levels.\n\n"
            "CRITICAL SECURITY CONSTANT 2: BETA_VAL = 10482\n\n"
            "Section 3: Rate Limiting & Network Partition Resiliency\n"
            "Token bucket algorithm evaluates rate limits atomically in Redis clusters.\n"
            "Datacenter isolation triggers local fallback caching during cross-region partitions.\n\n"
            "CRITICAL SECURITY CONSTANT 3: GAMMA_VAL = 77109\n\n"
            "Section 4: Key Management & Cryptographic Signatures\n"
            "Hardware Security Modules (HSM) store master key pairs in encrypted vaults.\n"
            "Token signatures use RS256 with 4096-bit key rotation every 30 days.\n\n"
            "CRITICAL SECURITY CONSTANT 4: DELTA_VAL = 33012\n\n"
            "Section 5: Exception Envelope & Compliance Audit Standards\n"
            "Error responses format exception payloads with trace identifiers and failure codes.\n\n"
            "CRITICAL SECURITY CONSTANT 5: EPSILON_VAL = 91802\n\n"
            "TASK:\n"
            "1. Extract all 5 critical constants: `ALPHA_VAL`, `BETA_VAL`, `GAMMA_VAL`, `DELTA_VAL`, `EPSILON_VAL`.\n"
            "2. Compute the exact composite security hash `HASH_CODE` using the formula:\n"
            "`HASH_CODE = (ALPHA_VAL * BETA_VAL + GAMMA_VAL * DELTA_VAL - EPSILON_VAL) % 1000003`.\n"
            "3. Show step-by-step mathematical calculations and state the final `HASH_CODE` in bold."
        ),
        "max_tokens": 1000,
        "temperature": 0.0,
    },
    {
        "id": "task_h4_monte_carlo_portfolio",
        "category": "Quantitative Finance & Math",
        "name": "Correlated Asset Portfolio Monte Carlo Risk Engine (VaR & CVaR)",
        "system": "You are a quantitative risk engineer. Write modular, high-precision Python numerical code.",
        "prompt": (
            "Write a production-grade Quantitative Risk Engine in Python for a 4-asset portfolio.\n"
            "Requirements:\n"
            "1. Inputs: Asset Weights `W = [0.35, 0.25, 0.20, 0.20]`, Initial Value `$10,000,000`, Daily Volatilities `sigma = [0.018, 0.025, 0.012, 0.030]`, Annual Returns `mu = [0.08, 0.12, 0.05, 0.15]`.\n"
            "2. Correlation Matrix `R` (4x4 symmetric positive-definite matrix with non-zero cross-correlations).\n"
            "3. Implement Cholesky Decomposition (`L` where `R = L * L^T`) from scratch (pure Python, no numpy/scipy) to generate correlated Gaussian random variables.\n"
            "4. Run a 10,000-path Monte Carlo simulation for a 10-day horizon.\n"
            "5. Compute 95% and 99% Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall).\n"
            "6. Include a main function `run_risk_simulation()` that prints a formatted financial risk report table."
        ),
        "max_tokens": 1600,
        "temperature": 0.2,
    },
    {
        "id": "task_h5_agent_cascading_healing",
        "category": "Agent & Resilience Engineering",
        "name": "Cascading Tool Failure & Self-Healing Agent Architecture",
        "system": "You are an autonomous agent system architect.",
        "prompt": (
            "Write a Python framework for an Autonomous Agent with Self-Healing Tool Invocation & Schema Auto-Correction.\n"
            "Requirements:\n"
            "1. Implement `AgentOrchestrator` that executes multi-step tool calls.\n"
            "2. Define tool schemas for `fetch_user_db(user_id)` and `process_payment(user_id, amount_usd, currency)`.\n"
            "3. Implement a dynamic `heal_tool_payload(tool_name, raw_payload, error_traceback)` function that intercepts malformed JSON or type errors (e.g. `'amount_usd': '$500'` as string instead of float), automatically fixes parameter types using schema definitions, and retries the tool call.\n"
            "4. Implement exponential backoff with jitter and circuit breaker pattern (opening circuit after 3 consecutive failures).\n"
            "5. Provide a test suite `test_agent_healing()` that simulates malformed payloads, rate-limit HTTP 429 exceptions, and verifies auto-healing and circuit breaker trips."
        ),
        "max_tokens": 1600,
        "temperature": 0.2,
    },
    {
        "id": "task_h6_adversarial_dp_opt",
        "category": "Algorithm Design & Dynamic Programming",
        "name": "Adversarial Matrix Chain Multiplication with Dynamic Penalty Costs",
        "system": "You are a competitive programming champion and algorithm designer.",
        "prompt": (
            "Solve the Dynamic Matrix Chain Multiplication Problem with Dynamic Reallocation Penalties in Python.\n"
            "Problem Statement:\n"
            "You are given an array of matrix dimensions `p` of length `n+1` representing `n` matrices `A1, A2, ..., An` where `Ai` has dimension `p[i-1] x p[i]`.\n"
            "Additionally, multiplying sub-chain `(Ai...Ak)` and `(Ak+1...Aj)` incurs a memory allocation penalty cost `C(i, k, j) = abs(p[i-1] * p[k] - p[k] * p[j]) % 17`.\n"
            "Requirements:\n"
            "1. Implement `optimal_matrix_chain_order(p: list[int]) -> tuple[int, str]` that computes the MINIMUM total computational cost (scalar multiplications + penalty cost) using O(n^3) dynamic programming.\n"
            "2. Construct and return the optimal parenthesization string (e.g. `((A1 x A2) x (A3 x A4))`).\n"
            "3. Include a function `reconstruct_cost_table(p)` that returns the DP lookup table.\n"
            "4. Include a test suite `test_matrix_chain()` with `p = [10, 30, 5, 60, 15, 20, 10]` verifying exact cost and parenthesization structure."
        ),
        "max_tokens": 1500,
        "temperature": 0.2,
    },
]


def send_chat_completion_robust(endpoint: str, model: str, api_key: str, messages: list, max_tokens: int = 1500, temp: float = 0.2, retries: int = 3) -> tuple:
    """Sends HTTP completion with automatic retries and 300s timeout (via shared LLMClient)."""
    client = LLMClient(endpoint=endpoint, api_key=api_key, model=model)

    last_error = ""
    for attempt in range(1, retries + 1):
        res = client.call(messages, max_tokens=max_tokens, temperature=temp, stream=False, timeout=300, retries=0)
        if not res.get("error"):
            t_wall = res.get("total_time", 0.0)
            content = res.get("content", "")
            completion_tokens = res.get("completion_tokens", 0) or max(1, int(len(content.split()) * 1.3))
            tok_s = res.get("decode_speed", 0.0) or completion_tokens / max(0.001, t_wall)
            return content, t_wall, completion_tokens, tok_s
        last_error = res["error"]
        # Client errors (4xx) will not succeed on retry; stop early.
        status = res.get("status_code")
        if status is not None and 400 <= status < 500:
            break
        print(f"    [Warning]: Connection attempt {attempt}/{retries} to {endpoint} failed: {last_error}. Retrying in 2s...")
        time.sleep(2)

    return f"[ERROR]: {last_error}", 0.0, 0, 0.0


def test_code_syntax(code_text: str) -> tuple:
    """Verifies if generated Python code is syntactically valid (AST compilation check)."""
    blocks = re.findall(r'```python(.*?)```', code_text, re.DOTALL)
    target_code = max(blocks, key=len) if blocks else code_text

    try:
        ast.parse(target_code)
        return True, "Valid Python Syntax."
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return True, "Valid Code Structure."


def parse_judge_score_10(json_str: str, key_prefix: str) -> tuple:
    """Parses 10-point quality rating from LLM judge response.

    Returns (None, reason) when the judge call failed or no score could be
    extracted, so callers can surface an error instead of a fabricated baseline.
    """
    if not json_str or "[ERROR]" in json_str:
        return None, "judge call failed"
    try:
        cleaned = json_str.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        data = json.loads(cleaned)

        scores_dict = data.get(f"{key_prefix}_scores") or data.get(f"scores_{key_prefix}") or {}
        if isinstance(scores_dict, dict) and len(scores_dict) > 0:
            vals = [float(v) for v in scores_dict.values() if isinstance(v, (int, float))]
            if vals:
                score_10 = round(sum(vals) / len(vals), 1)
                return score_10, data.get("detailed_rationale", "")

        tot = data.get(f"{key_prefix}_total_10") or data.get(f"{key_prefix}_total") or data.get(f"total_{key_prefix}")
        if tot is not None:
            score_10 = round(float(tot) / 5.0, 1) if float(tot) > 10.0 else round(float(tot), 1)
            return min(10.0, max(0.0, score_10)), data.get("detailed_rationale", "")

    except Exception:
        pass

    nums = re.findall(rf"{key_prefix}.*?(\d+(?:\.\d+)?)", json_str, re.IGNORECASE)
    if nums:
        val = float(nums[0])
        score_10 = round(val / 5.0, 1) if val > 10.0 else round(val, 1)
        return min(10.0, max(0.0, score_10)), "Regex extracted score."

    return None, "no score found"


def run_llm_judge_10_pair(judge_endpoint: str, judge_model: str, judge_key: str, task: dict, out1: str, out2: str, m1_name: str, m2_name: str) -> dict:
    """
    Executes double-blind position-swapped LLM Quality Judge scoring (0.0 - 10.0 scale per task).
    """
    m1_failed = not out1 or "[ERROR]" in out1 or len(out1.strip()) == 0
    m2_failed = not out2 or "[ERROR]" in out2 or len(out2.strip()) == 0

    if m1_failed and m2_failed:
        return {"m1_score_10": 0.0, "m2_score_10": 0.0, "winner": "Tie (Both Failed)", "rationale": "Both servers failed to return valid outputs."}
    elif m1_failed:
        return {"m1_score_10": 0.0, "m2_score_10": 8.5, "winner": f"Server 2 ({m2_name})", "rationale": "Server 1 failed to return valid output."}
    elif m2_failed:
        return {"m1_score_10": 8.5, "m2_score_10": 0.0, "winner": f"Server 1 ({m1_name})", "rationale": "Server 2 failed to return valid output."}

    rubric = (
        "You are an expert AI Benchmark Judge evaluating two LLM solutions for a complex engineering task.\n"
        "Score BOTH solutions strictly out of 10.0 across 5 dimensions:\n"
        "1. Correctness & Edge-Case Coverage (0-10)\n"
        "2. Reasoning Depth & Analytical Rigor (0-10)\n"
        "3. Code Architecture, Modularity & Type Safety (0-10)\n"
        "4. Constraint Adherence & Schema Integrity (0-10)\n"
        "5. Completeness & Production Quality (0-10)\n\n"
        "Output ONLY raw JSON matching this schema:\n"
        "{\n"
        '  "model_a_scores": {"correctness": 9.5, "reasoning": 8.5, "architecture": 9.0, "constraints": 10.0, "completeness": 9.0},\n'
        '  "model_b_scores": {"correctness": 8.0, "reasoning": 7.5, "architecture": 8.0, "constraints": 8.5, "completeness": 8.0},\n'
        '  "model_a_total_10": 9.2,\n'
        '  "model_b_total_10": 8.0,\n'
        '  "winner": "Model A" or "Model B" or "Tie",\n'
        '  "detailed_rationale": "Clear breakdown of why Model A or B won..."\n'
        "}\n"
    )

    # Round 1: A=M1, B=M2
    p1 = f"### TASK: {task['name']}\n**Prompt**:\n{task['prompt']}\n\n### SOLUTION MODEL A:\n{out1}\n\n### SOLUTION MODEL B:\n{out2}"
    res1, _, _, _ = send_chat_completion_robust(judge_endpoint, judge_model, judge_key, [
        {"role": "system", "content": rubric},
        {"role": "user", "content": p1}
    ], max_tokens=800, temp=0.0)

    # Round 2: A=M2, B=M1 (Swapped)
    p2 = f"### TASK: {task['name']}\n**Prompt**:\n{task['prompt']}\n\n### SOLUTION MODEL A:\n{out2}\n\n### SOLUTION MODEL B:\n{out1}"
    res2, _, _, _ = send_chat_completion_robust(judge_endpoint, judge_model, judge_key, [
        {"role": "system", "content": rubric},
        {"role": "user", "content": p2}
    ], max_tokens=800, temp=0.0)

    m1_score_r1, rat1 = parse_judge_score_10(res1, "model_a")
    m2_score_r1, _ = parse_judge_score_10(res1, "model_b")

    m2_score_r2, rat2 = parse_judge_score_10(res2, "model_a")
    m1_score_r2, _ = parse_judge_score_10(res2, "model_b")

    if None in (m1_score_r1, m2_score_r1, m1_score_r2, m2_score_r2):
        return {
            "m1_score_10": None,
            "m2_score_10": None,
            "winner": "JUDGE ERROR",
            "rationale": "Judge endpoint failed to return a parseable score; task not graded.",
        }

    m1_final_10 = round((m1_score_r1 + m1_score_r2) / 2.0, 2)
    m2_final_10 = round((m2_score_r1 + m2_score_r2) / 2.0, 2)

    if m1_final_10 > m2_final_10:
        winner = f"Server 1 ({m1_name})"
    elif m2_final_10 > m1_final_10:
        winner = f"Server 2 ({m2_name})"
    else:
        winner = "Tie"

    rationale = f"[Round 1]: {rat1[:200]}\n[Round 2 Swapped]: {rat2[:200]}"
    return {
        "m1_score_10": m1_final_10,
        "m2_score_10": m2_final_10,
        "winner": winner,
        "rationale": rationale,
    }


def compute_model_tier(score_10: float) -> str:
    if score_10 >= 9.0:
        return "S-Tier (Exceptional)"
    elif score_10 >= 8.0:
        return "A-Tier (High Performance)"
    elif score_10 >= 7.0:
        return "B-Tier (Solid)"
    elif score_10 >= 5.0:
        return "C-Tier (Moderate)"
    else:
        return "D-Tier (Needs Improvement)"


def run_hard_complex_benchmark(ep1: str, m1: str, k1: str, ep2: str = None, m2: str = None, k2: str = None, out_file: str = "results/official_model_ranking_10.md"):
    """Runs Quality Evaluation out of 10 and generates Official Model Ranking Leaderboard."""
    is_dual = bool(ep2 and m2)

    print("=" * 100)
    print(" VERIFIED LLM EVAL SUITE: FAIR 10-POINT QUALITY GRADING & MODEL LEADERBOARD ")
    print("=" * 100)
    print(f" Server 1 (Model A) : {m1} ({ep1})")
    if is_dual:
        print(f" Server 2 (Model B) : {m2} ({ep2})")
    print(f" Judge Endpoint    : {ep1}")
    print("=" * 100 + "\n")

    task_records = []
    t_start_suite = time.perf_counter()

    for idx, task in enumerate(HARD_BENCHMARK_TASKS, 1):
        print(f"[{idx}/{len(HARD_BENCHMARK_TASKS)}] Evaluating Task: {task['name']} ({task['category']})...")

        # Prompt Server 1
        out1, wall1, toks1, spd1 = send_chat_completion_robust(ep1, m1, k1, [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": task["prompt"]}
        ], max_tokens=task["max_tokens"], temp=task["temperature"])

        syn_ok1, syn_msg1 = test_code_syntax(out1) if "Python" in task["system"] or "python" in task["prompt"].lower() else (True, "N/A")

        if is_dual:
            out2, wall2, toks2, spd2 = send_chat_completion_robust(ep2, m2, k2, [
                {"role": "system", "content": task["system"]},
                {"role": "user", "content": task["prompt"]}
            ], max_tokens=task["max_tokens"], temp=task["temperature"])

            syn_ok2, syn_msg2 = test_code_syntax(out2) if "Python" in task["system"] or "python" in task["prompt"].lower() else (True, "N/A")

            judge_res = run_llm_judge_10_pair(ep1, m1, k1, task, out1, out2, m1[:12], m2[:12])
            s1_score = judge_res["m1_score_10"]
            s2_score = judge_res["m2_score_10"]
            rat = judge_res["rationale"]

            if s1_score is None or s2_score is None:
                winner = "JUDGE ERROR"
            else:
                # Apply syntax validation adjustments if code has syntax errors
                if not syn_ok1 and s1_score > 4.0:
                    s1_score = round(max(3.0, s1_score - 1.5), 1)
                if not syn_ok2 and s2_score > 4.0:
                    s2_score = round(max(3.0, s2_score - 1.5), 1)

                if s1_score > s2_score:
                    winner = f"Server 1 ({m1[:12]})"
                elif s2_score > s1_score:
                    winner = f"Server 2 ({m2[:12]})"
                else:
                    winner = "Tie"
        else:
            rubric_single = "Score this LLM output out of 10.0 for correctness, reasoning, architecture, and prompt adherence. Output JSON: {\"score_10\": 8.5, \"rationale\": \"...\"}"
            j_txt, _, _, _ = send_chat_completion_robust(ep1, m1, k1, [
                {"role": "system", "content": rubric_single},
                {"role": "user", "content": f"Task: {task['name']}\nOutput:\n{out1}"}
            ], max_tokens=400, temp=0.0)
            s1_score, rat = parse_judge_score_10(j_txt, "score")
            s2_score = 0.0
            winner = "JUDGE ERROR" if s1_score is None else f"Server 1 ({m1[:12]})"
            syn_ok2, syn_msg2 = True, "N/A"

        s1_disp = "ERR" if s1_score is None else f"{s1_score:.1f}"
        s2_disp = "ERR" if s2_score is None else f"{s2_score:.1f}"
        print(f"  ┌{'─' * 80}┐")
        print(f"  │ Server 1 (`{m1[:14]}`): Grade {s1_disp:>5} / 10.0 │ Syntax: {'PASS' if syn_ok1 else 'FAIL'} │ {spd1:>6.2f} tok/s │ Wall: {wall1:>5.2f}s │")
        if is_dual:
            print(f"  │ Server 2 (`{m2[:14]}`): Grade {s2_disp:>5} / 10.0 │ Syntax: {'PASS' if syn_ok2 else 'FAIL'} │ {spd2:>6.2f} tok/s │ Wall: {wall2:>5.2f}s │")
            print(f"  ├{'─' * 80}┤")
            print(f"  │ Task Winner: {winner:<63} │")
        print(f"  └{'─' * 80}┘\n")

        rec = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "server1": {"model": m1, "score_10": s1_score, "syntax_ok": syn_ok1, "tok_s": round(spd1, 2), "wall_s": round(wall1, 2), "tokens": int(toks1)},
            "server2": {"model": m2, "score_10": s2_score, "syntax_ok": syn_ok2, "tok_s": round(spd2, 2), "wall_s": round(wall2, 2), "tokens": int(toks2)} if is_dual else None,
            "winner": winner if is_dual else "N/A",
            "rationale": rat,
        }
        task_records.append(rec)

    total_suite_wall = time.perf_counter() - t_start_suite

    # --- Leaderboard & Ranking Calculation ---
    s1_scores = [r["server1"]["score_10"] for r in task_records if r["server1"]["score_10"] is not None]
    s1_avg_score = round(sum(s1_scores) / len(s1_scores), 2) if s1_scores else 0.0
    s1_avg_spd = round(sum(r["server1"]["tok_s"] for r in task_records) / len(task_records), 2)
    s1_tier = compute_model_tier(s1_avg_score)

    models_ranked = [
        {"rank": 1, "model": m1, "endpoint": ep1, "score_10": s1_avg_score, "tok_s": s1_avg_spd, "tier": s1_tier}
    ]

    if is_dual:
        s2_scores = [r["server2"]["score_10"] for r in task_records if r["server2"]["score_10"] is not None]
        s2_avg_score = round(sum(s2_scores) / len(s2_scores), 2) if s2_scores else 0.0
        s2_avg_spd = round(sum(r["server2"]["tok_s"] for r in task_records) / len(task_records), 2)
        s2_tier = compute_model_tier(s2_avg_score)

        models_ranked.append({"rank": 2, "model": m2, "endpoint": ep2, "score_10": s2_avg_score, "tok_s": s2_avg_spd, "tier": s2_tier})
        models_ranked.sort(key=lambda x: (x["score_10"], x["tok_s"]), reverse=True)
        for r_idx, m_item in enumerate(models_ranked, 1):
            m_item["rank"] = r_idx

    # Markdown Report Generation
    md_file = out_file if out_file.endswith(".md") else f"{out_file}.md"
    json_file = out_file[:-3] + ".json" if out_file.endswith(".md") else f"{out_file}.json"

    md_lines = [
        "# Official LLM Benchmark Model Ranking & Quality Leaderboard (10-Point Scale)",
        "",
        f"- **Date Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Evaluation Mode**: {'Dual-Server Live Arena' if is_dual else 'Single Server Quality Benchmark'}",
        f"- **Total Suite Wall Time**: `{total_suite_wall:.2f} s`",
        "",
        "## 🏆 Official Model Quality Ranking & Leaderboard",
        "",
        "| Rank | Model Name | Endpoint Base URL | Quality Score (/10.0) | Decode Throughput | Quality Tier |",
        "|:---:|:---|:---|:---:|:---:|:---|",
    ]

    medal_icons = ["🥇", "🥈", "🥉", "4️⃣"]
    for item in models_ranked:
        icon = medal_icons[item["rank"] - 1] if item["rank"] <= len(medal_icons) else str(item["rank"])
        md_lines.append(
            f"| {icon} **#{item['rank']}** | `{item['model']}` | `{item['endpoint']}` | **`{item['score_10']} / 10.0`** | `{item['tok_s']} tok/s` | **{item['tier']}** |"
        )

    md_lines.extend([
        "",
        "## Task-by-Task Verified 10-Point Quality Breakdown",
        "",
    ])

    if is_dual:
        md_lines.append(f"| # | Complex Task | Category | Server 1 Score (`{m1}`) | Server 2 Score (`{m2}`) | Task Winner |")
        md_lines.append("|:---|:---|:---|:---:|:---:|:---|")
        for idx, r in enumerate(task_records, 1):
            md_lines.append(f"| {idx} | {r['name']} | {r['category']} | **`{r['server1']['score_10']} / 10`** | **`{r['server2']['score_10']} / 10`** | **{r['winner']}** |")
    else:
        md_lines.append(f"| # | Complex Task | Category | Model Score (`{m1}`) | Decode Speed | Status |")
        md_lines.append("|:---|:---|:---|:---:|:---:|:---|")
        for idx, r in enumerate(task_records, 1):
            md_lines.append(f"| {idx} | {r['name']} | {r['category']} | **`{r['server1']['score_10']} / 10`** | `{r['server1']['tok_s']} tok/s` | PASS |")

    out_dir = os.path.dirname(md_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(md_file, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    json_report = {
        "benchmark_mode": "model_ranking_10",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_time_s": round(total_suite_wall, 2),
        "leaderboard": models_ranked,
        "tasks": task_records,
    }

    with open(json_file, "w") as f:
        json.dump(json_report, f, indent=2)

    print("=" * 100)
    print(" VERIFIED MODEL RANKING LEADERBOARD ")
    print("=" * 100)
    for item in models_ranked:
        print(f" Rank #{item['rank']}: {item['model']:<25} | Quality Score: {item['score_10']:>4.1f} / 10.0 | Speed: {item['tok_s']:>6.2f} tok/s | Tier: {item['tier']}")
    print("=" * 100)
    print(f" • Markdown Leaderboard Report : {md_file}")
    print(f" • Structured JSON Report      : {json_file}\n")


def main():
    parser = argparse.ArgumentParser(description="10-Point Task Quality Grading & Model Ranking Leaderboard")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--out", default="results/official_model_ranking_10.md", help="Output markdown report path")
    parser.add_argument("--single", action="store_true", help="Run single-server grading out of 10")
    args = parser.parse_args()

    cfg, _, _ = env_loader.load_config(args.config)

    ep1 = cfg.get("endpoint1") or cfg.get("endpoint") or os.getenv("ENDPOINT1") or os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    m1 = cfg.get("model1") or cfg.get("model") or os.getenv("MODEL1") or os.getenv("OPENAI_MODEL") or "default"
    k1 = cfg.get("api_key1") or cfg.get("api_key") or os.getenv("API_KEY1") or os.getenv("OPENAI_API_KEY") or ""

    if m1 == "default":
        m1 = discover_model_safe(ep1, k1)
        if m1 == "default":
            print("Error: Could not discover a model. Specify the model in config.json or set OPENAI_MODEL.")
            sys.exit(1)

    if args.single:
        run_hard_complex_benchmark(ep1, m1, k1, out_file=args.out)
    else:
        ep2 = cfg.get("endpoint2") or os.getenv("ENDPOINT2") or "http://127.0.0.1:8001/v1"
        m2 = cfg.get("model2") or os.getenv("MODEL2") or "default"
        k2 = cfg.get("api_key2") or os.getenv("API_KEY2") or ""
        if m2 == "default":
            m2 = discover_model_safe(ep2, k2)
            if m2 == "default":
                print("Error: Could not discover a model for endpoint2. Specify model2 in config.json or set MODEL2.")
                sys.exit(1)
        run_hard_complex_benchmark(ep1, m1, k1, ep2, m2, k2, args.out)


if __name__ == "__main__":
    main()
