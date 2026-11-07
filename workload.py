#!/usr/bin/env python3
"""
workload.py -- Test 46: the deployment's own workload.

The default is the target self-hosting scenario: 4 users, each sending a new
200K-token document set (no prefix-cache reuse), thinking on, about a
2,000-token answer, users arriving 3-10 s apart. Each user asks about one
contract in the set: quote a clause the lawyers labelled (CUAD), then write
a review of that contract. The quote is graded like Test 47; the review sets
the answer length.

Phases:
  1. One user alone: single-user prefill, time to first token and decode.
  2. All users, staggered: what each user waits for and how fast the answer
     streams. A user meets the workable gate when the first token, the whole
     answer and the per-user decode speed are all within the configured
     limits. Goodput = users who met it.

Every setting is configurable (config.json "workload"). Zero external
dependencies (stdlib + this repo's modules).
"""

import math
import random
import statistics
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import domain_benchmarks as db
import suites
from client import GREEN, YELLOW, RED, CYAN, log, mark_infra_failure, INFRA_FAILURE_KINDS

DEFAULT_WORKLOAD: Dict[str, Any] = {
    "prompt_tokens": 200000,
    "users": 4,
    "answer_tokens": 2000,
    "max_output_tokens": 8192,      # thinking + answer
    "stagger_s": [3, 10],           # random gap between users' requests
    "seed": 20261106,
    "gates": {"first_token_s": 90, "complete_s": 300, "per_user_tok_s": 15},
}
PROMPT_MARGIN_TOKENS = 1000         # chat template and sizing error
CALIBRATION_CHARS = 40000           # one request measures the tokenizer's chars per token
REQUEST_TIMEOUT_S = 1800            # 4 x 200K prompts can queue for minutes before the first token
FALLBACK_TOKENS_PER_CHAR = 0.25


def resolve_workload(block: Optional[Dict[str, Any]], max_context: Optional[int]) -> Dict[str, Any]:
    """The workload settings: config values over the defaults. The prompt is
    reduced to fit the server's context with room for the output; the
    result says so."""
    w = dict(DEFAULT_WORKLOAD)
    w["gates"] = dict(DEFAULT_WORKLOAD["gates"])
    for key, value in (block or {}).items():
        if key == "gates" and isinstance(value, dict):
            w["gates"].update(value)
        elif key in DEFAULT_WORKLOAD:
            w[key] = value
    w["users"] = max(1, int(w["users"]))
    lo, hi = (list(w["stagger_s"]) + [0, 0])[:2]
    w["stagger_s"] = [float(min(lo, hi)), float(max(lo, hi))]
    if max_context:
        fit = int(max_context) - int(w["max_output_tokens"]) - PROMPT_MARGIN_TOKENS
        if fit < int(w["prompt_tokens"]):
            w["requested_prompt_tokens"] = int(w["prompt_tokens"])
            w["prompt_tokens"] = max(1000, fit)
    return w


def stagger_offsets(users: int, lo: float, hi: float, seed: int) -> List[float]:
    """Start time of each user's request: the first at 0, each next one a
    random lo-hi seconds later (fixed seed, so runs are comparable)."""
    rnd = random.Random(seed)
    offsets, t = [0.0], 0.0
    for _ in range(users - 1):
        t += rnd.uniform(lo, hi)
        offsets.append(round(t, 3))
    return offsets


def review_words(w: Dict[str, Any]) -> int:
    return max(100, int(int(w["answer_tokens"]) * 0.75))


def build_prompt(contracts: List[Dict[str, Any]], target_chars: int, seed: int, focus: Dict[str, Any],
                 question: str, words: int) -> str:
    """A document set of target_chars characters of contract text that
    contains the focus contract, in a seeded order, opening with a unique
    session tag so no request shares a cached prefix with another. The last
    filler contract is cut to hit the size (the focus contract is never cut)."""
    rnd = random.Random(seed)
    others = [c for c in contracts if c["id"] != focus["id"]]
    rnd.shuffle(others)
    chosen, size = [], len(focus["text"])
    for c in others:
        room = target_chars - size
        if room <= 0:
            break
        if len(c["text"]) > room:
            c = dict(c, text=c["text"][:room].rsplit(" ", 1)[0] + "\n[... document continues ...]")
        chosen.append(c)
        size += len(c["text"])
    # The focus contract sits between 30% and 90% of the way through the set.
    pos = min(len(chosen), max(0, int(len(chosen) * rnd.uniform(0.3, 0.9))))
    chosen.insert(pos, focus)
    parts = [f"[Session {uuid.uuid4()}]\nYou are a legal assistant. Below is a set of {len(chosen)} contracts."]
    for i, c in enumerate(chosen, 1):
        parts.append(f"=== DOCUMENT {i}: {c['title']} ===\n{c['text']}")
    k = pos + 1
    parts.append("=== END OF DOCUMENTS ===")
    parts.append(
        f"Question about DOCUMENT {k} ({focus['title']}): {' '.join(question.split())}\n\n"
        "Reply in two parts.\n"
        f"Part 1: if DOCUMENT {k} contains such a clause, write 'QUOTE:' followed by the clause text copied "
        "exactly from the document; if it does not, write exactly 'NOT FOUND'.\n"
        f"Part 2: write 'REVIEW:' followed by a review of DOCUMENT {k} for the lawyer: its key obligations, "
        f"risks and unusual terms, about {words} words.")
    return "\n\n".join(parts)


PROMPT_FIT_TOLERANCE = 0.01   # prompts land within 1% of the configured size


def client_count(client, prompt: str) -> Optional[int]:
    counter = getattr(client, "count_prompt_tokens", None)
    return counter([{"role": "user", "content": prompt}]) if counter else None


def fit_prompt(client, contracts: List[Dict[str, Any]], target_chars: int, target_tokens: int, seed: int,
               focus: Dict[str, Any], question: str, words: int) -> str:
    """build_prompt, then, when the server can count tokens (/tokenize),
    rescale the text until the prompt is within 1% of target_tokens. Before,
    whole contracts were added past the target: prompts ran 4-11% long."""
    prompt = build_prompt(contracts, target_chars, seed, focus, question, words)
    for _ in range(4):
        n = client_count(client, prompt)
        if not n or abs(n - target_tokens) <= target_tokens * PROMPT_FIT_TOLERANCE:
            break
        target_chars = int(target_chars * target_tokens / n)
        prompt = build_prompt(contracts, target_chars, seed, focus, question, words)
    return prompt


def pick_questions(contracts: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """n (contract, clause) questions whose clause the contract has: one per
    contract first, then further clauses of the same contracts when n is
    larger than the contract set (each prompt is still a new document: its
    session tag and filler differ)."""
    pairs = [(c, cat) for c in contracts for cat, spans in sorted(c["clauses"].items()) if spans]
    random.Random(seed).shuffle(pairs)
    first: Dict[str, Any] = {}
    for c, cat in pairs:
        first.setdefault(c["id"], (c, cat))
    leads = list(first.values())
    ordered = leads + [p for p in pairs if p not in leads]
    if not ordered:
        return []
    return [{"contract": c, "category": cat, "answers": c["clauses"][cat],
             "question": c.get("questions", {}).get(cat, cat)}
            for c, cat in (ordered[i % len(ordered)] for i in range(n))]


def tokens_per_char(client, contracts: List[Dict[str, Any]]) -> float:
    """The server's tokens per character on contract text, from one short
    request (its prompt_tokens over the characters sent)."""
    sample = "\n\n".join(c["text"] for c in contracts)[:CALIBRATION_CHARS]
    res = client.call([{"role": "user", "content": sample}], max_tokens=1, temperature=0.0, stream=False,
                      timeout=300, retries=1)
    if res.get("error") or not res.get("prompt_tokens"):
        return FALLBACK_TOKENS_PER_CHAR
    return res["prompt_tokens"] / len(sample)


def split_answer(text: str) -> str:
    """Part 1 of the answer (the quote or NOT FOUND), without the review."""
    return (text or "").split("REVIEW:", 1)[0]


def run_user(client, prompt: str, question: Dict[str, Any], w: Dict[str, Any], start_offset: float,
             t_zero: float) -> Dict[str, Any]:
    """Sends one user's request at t_zero + start_offset and measures it."""
    delay = t_zero + start_offset - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    t_sent = time.perf_counter()
    res = client.call([{"role": "user", "content": prompt}], max_tokens=int(w["max_output_tokens"]),
                      sampling=suites.sampling_for(3), stream=True, timeout=REQUEST_TIMEOUT_S, retries=0)
    t_done = time.perf_counter()
    record: Dict[str, Any] = {"start_offset_s": round(t_sent - t_zero, 2)}
    if res.get("error"):
        record.update(error=str(res["error"])[:300], failure_kind=res.get("failure_kind"), meets_gates=False,
                      score=0.0, outcome="error", unreachable=bool(res.get("endpoint_unreachable")))
        return record
    text = db.final_content(res)
    score, detail = suites.grade_clause_answer(question["answers"], split_answer(text))
    answer_ttft = res.get("answer_ttft")
    gates = w["gates"]
    first, complete, decode = res.get("ttft") or 0.0, t_done - t_sent, res.get("decode_speed") or 0.0
    record.update({
        "prompt_tokens": res.get("prompt_tokens", 0),
        "cached_tokens": res.get("cached_tokens", 0),
        "first_token_s": round(first, 2),
        "answer_start_s": round(answer_ttft, 2) if answer_ttft is not None else None,
        "complete_s": round(complete, 2),
        "completion_tokens": res.get("completion_tokens", 0),
        "reasoning_chunks": res.get("reasoning_chunks", 0),
        "answer_chunks": res.get("content_chunks", 0),
        "decode_tok_s": round(decode, 1),
        "prefill_tok_s": round(res["prompt_tokens"] / res["ttft"], 1) if res.get("ttft") else 0.0,
        "finish_reason": res.get("finish_reason"),
        "score": score,
        "outcome": detail["outcome"],
        "category": question["category"],
        "contract": question["contract"]["title"],
        "response_tail": text[-600:],
    })
    # Gates are checked on the unrounded times. A cut-off answer (token limit)
    # is not a complete answer.
    record["gate_checks"] = {
        "first_token": first <= gates["first_token_s"],
        "complete": complete <= gates["complete_s"] and res.get("finish_reason") != "length",
        "decode": decode >= gates["per_user_tok_s"],
    }
    record["meets_gates"] = all(record["gate_checks"].values())
    return record


def _log_users(rows: List[Dict[str, Any]]) -> None:
    log(f"  {'User':>4} | {'Start s':>7} | {'Prompt tok':>10} | {'Cached':>6} | {'First tok s':>11} | "
        f"{'Answer s':>8} | {'Done s':>7} | {'Decode t/s':>10} | {'Quote':>7} | Gates")
    for i, r in enumerate(rows, 1):
        if r.get("error"):
            log(f"  {i:>4} | {r['start_offset_s']:>7} | error: {r['error'][:90]}", color=RED)
            continue
        ans = "-" if r["answer_start_s"] is None else f"{r['answer_start_s']:.1f}"
        log(f"  {i:>4} | {r['start_offset_s']:>7} | {r['prompt_tokens']:>10,} | {r['cached_tokens']:>6,} | "
            f"{r['first_token_s']:>11.1f} | {ans:>8} | {r['complete_s']:>7.1f} | {r['decode_tok_s']:>10.1f} | "
            f"{'right' if r['score'] else 'wrong':>7} | {'met' if r['meets_gates'] else 'missed'}",
            color=None if r["meets_gates"] else YELLOW)


def aggregate_rates(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Combined prefill and decode across concurrent users: tokens over the
    wall time from the first request to the last first token (prefill) and
    from the first first token to the last finish (decode)."""
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return {"aggregate_prefill_tok_s": 0.0, "aggregate_decode_tok_s": 0.0}
    t_start = min(r["start_offset_s"] for r in ok)
    t_last_first = max(r["start_offset_s"] + r["first_token_s"] for r in ok)
    t_first_first = min(r["start_offset_s"] + r["first_token_s"] for r in ok)
    t_end = max(r["start_offset_s"] + r["complete_s"] for r in ok)
    prefill_span, decode_span = t_last_first - t_start, t_end - t_first_first
    return {
        "aggregate_prefill_tok_s": round(sum(r["prompt_tokens"] for r in ok) / prefill_span, 1) if prefill_span > 0 else 0.0,
        "aggregate_decode_tok_s": round(sum(r["completion_tokens"] for r in ok) / decode_span, 1) if decode_span > 0 else 0.0,
    }


def sizing(result: Dict[str, Any]) -> Dict[str, Any]:
    """Users one server carries within the gates, and servers for the
    configured users. Concurrent goodput when any user met the gates;
    otherwise 1 when the user alone did (before, that case reported no
    estimate although one user per server works); otherwise 0."""
    users = result.get("users") or []
    n, good = len(users), result.get("goodput_users", 0)
    if n and good == n:
        per, basis = n, f"all {n} users met the gates at once; more were not tested"
    elif good:
        per, basis = good, f"{good} of {n} users met the gates at once"
    elif (result.get("solo") or {}).get("meets_gates"):
        per = 1
        basis = f"1 user alone met the gates; {n} at once did not" + (
            f"; 2-{n - 1} were not tested" if n > 2 else "")
    else:
        per, basis = 0, "not even 1 user alone met the gates"
    return {"users_per_server": per, "servers_for_users": servers_needed(n, per), "basis": basis}


def servers_needed(users: int, goodput: int) -> Optional[int]:
    """Servers like this one for `users` concurrent users at this workload,
    assuming identical servers behind a load balancer; None when no user met
    the gates on one server."""
    return math.ceil(users / goodput) if goodput > 0 else None


def run_workload(client, block: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    w = resolve_workload(block, getattr(client, "max_context", None))
    log("\n" + "=" * 88, bold=True)
    log(f"[TEST 46] Workload: {w['users']} users x {w['prompt_tokens']:,}-token new documents, "
        f"~{w['answer_tokens']:,}-token answers, thinking on", bold=True, color=CYAN)
    log(f"Gates per user: first token <= {w['gates']['first_token_s']} s, complete answer <= "
        f"{w['gates']['complete_s']} s, decode >= {w['gates']['per_user_tok_s']} tok/s", color=CYAN)
    log("=" * 88)
    contracts_by_id, meta = suites.load_contracts()
    contracts = list(contracts_by_id.values())
    if not contracts:
        return {"status": "SKIPPED", "error": "dataset missing (run scripts/fetch_benchmark_data.py contracts)"}
    if w.get("requested_prompt_tokens"):
        log(f"  Prompt reduced to {w['prompt_tokens']:,} tokens to fit the server's context "
            f"({w['requested_prompt_tokens']:,} requested)", color=YELLOW)

    ratio = tokens_per_char(client, contracts)
    target_chars = int(w["prompt_tokens"] / ratio)
    available = sum(len(c["text"]) for c in contracts)
    if target_chars > available:
        log(f"  The contract set holds about {int(available * ratio):,} tokens; prompts are capped there.", color=YELLOW)
    questions = pick_questions(contracts, w["users"] + 1, int(w["seed"]))
    words = review_words(w)
    prompts = [fit_prompt(client, contracts, target_chars, int(w["prompt_tokens"]), int(w["seed"]) + i,
                          q["contract"], q["question"], words)
               for i, q in enumerate(questions)]
    sizes = [client_count(client, p) for p in prompts]
    if all(sizes):
        log(f"  Prompt sizes (server tokenizer): {', '.join(f'{n:,}' for n in sizes)} tokens")

    log("\n[A] One user alone", bold=True, color=CYAN)
    solo = run_user(client, prompts[0], questions[0], w, 0.0, time.perf_counter())
    _log_users([solo])

    offsets = stagger_offsets(w["users"], w["stagger_s"][0], w["stagger_s"][1], int(w["seed"]))
    log(f"\n[B] {w['users']} users, arriving at {offsets} s", bold=True, color=CYAN)
    rows: List[Optional[Dict[str, Any]]] = [None] * w["users"]
    t_zero = time.perf_counter()

    def worker(i):
        try:
            rows[i] = run_user(client, prompts[i + 1], questions[i + 1], w, offsets[i], t_zero)
        except Exception as e:  # a harness error: the user still counts, as missing the gates
            log(f"  user {i + 1}: harness error: {e}", color=YELLOW)
            rows[i] = {"start_offset_s": offsets[i], "error": f"harness error: {e}"[:300],
                       "failure_kind": "harness", "meets_gates": False, "score": 0.0}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(w["users"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    users = [r for r in rows if r is not None]
    _log_users(users)

    goodput = sum(1 for r in users if r.get("meets_gates"))
    correct = sum(1 for r in users if r.get("score"))
    kv = getattr(client, "fetch_kv_capacity", lambda: None)()
    kv_tokens = kv.get("kv_cache_tokens") if isinstance(kv, dict) else None
    per_session = int(w["prompt_tokens"]) + int(w["max_output_tokens"])
    # PASS: every user within the gates and every clause found. Goodput (the
    # sizing number) counts speed only; accuracy also feeds the long-document
    # use case through `details`.
    if goodput == 0:
        status = "FAIL"
    elif goodput == len(users) and correct == len(users):
        status = "PASS"
    else:
        status = "PARTIAL"
    result: Dict[str, Any] = {
        "status": status,
        "workload": w,
        "tokens_per_char": round(ratio, 4),
        "solo": solo,
        "users": users,
        "goodput_users": goodput,
        "correct_users": correct,
        "kv_sessions_at_workload": (kv_tokens // per_session) if kv_tokens else None,
        "servers_for_users": None,  # set from sizing() below
        **aggregate_rates(users),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in [solo] + users),
        "source": meta.get("source"),
        "license": meta.get("license"),
        # Every answer, for the long-document use case (one attempt each).
        "details": [{"id": f"workload_user_{i}", "score": r.get("score", 0.0),
                     "attempt_scores": [r.get("score", 0.0)],
                     "attempts": [{"expected": "a quote", "outcome": r.get("outcome"),
                                   "response_tail": r.get("response_tail", "")}]}
                    # An errored user (timeout, 5xx, harness error) gave no answer to
                    # grade: it counts in goodput and the gates, not in accuracy.
                    for i, r in enumerate([solo] + users) if not r.get("error")],
    }
    result["sizing"] = sizing(result)
    result["servers_for_users"] = result["sizing"]["servers_for_users"]
    times = [r["first_token_s"] for r in users if not r.get("error")]
    log(f"\n  -> {goodput} of {len(users)} users within the gates | clause found: {correct}/{len(users)} | "
        f"first token {min(times, default=0):.0f}-{max(times, default=0):.0f} s | "
        f"KV holds {result['kv_sessions_at_workload'] or '?'} sessions of this size -> {status}",
        color=GREEN if status == "PASS" else YELLOW)
    if times:
        result["median_first_token_s"] = round(statistics.median(times), 1)
    # Infra: the server could not be reached, or it failed the one user on an
    # idle server. A timeout or 5xx for a concurrent user is what this test
    # measures under load: that user missed the gates (counted above).
    infra = [r["error"] for r in [solo] + users
             if r.get("error") and r.get("failure_kind") in INFRA_FAILURE_KINDS
             and (r is solo or r.get("unreachable"))]
    return mark_infra_failure(result, infra)
