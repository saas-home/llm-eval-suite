#!/usr/bin/env python3
"""
Fetches seeded samples of official public benchmarks into data/suites/.

Each dataset is sampled once with a fixed seed (POOL items, default 100) so
every run of the suite sees the same questions; the runtime --samples flag
takes the first N of the pool. Every output file records its source,
license, revision and sample seed, and data/suites/SOURCES.md summarizes
them. Standard library only (Hugging Face datasets-server JSON API + raw
GitHub files); no `datasets` / `pyarrow` needed.

    python3 scripts/fetch_benchmark_data.py            # all suites
    python3 scripts/fetch_benchmark_data.py medical    # one suite

Datasets whose authors ask that items not be published (GPQA) are NOT
fetched here.
"""

import ast
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "suites")
POOL = 100
SEED = 20261106
API = "https://datasets-server.huggingface.co"


def _get(url, retries=8):
    """GET with backoff; honors Retry-After on HTTP 429 (rate limiting)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llm-eval-suite-data-fetch"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if attempt == retries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 15 * (attempt + 1)
            print(f"    HTTP {e.code}; retrying in {wait}s", flush=True)
            time.sleep(wait)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def _json(url):
    return json.loads(_get(url).decode("utf-8"))


def hf_num_rows(dataset, config, split):
    info = _json(f"{API}/size?dataset={urllib.parse.quote(dataset)}")
    for s in info["size"]["splits"]:
        if s["config"] == config and s["split"] == split:
            return s["num_rows"]
    raise KeyError(f"{dataset}/{config}/{split} not found")


def hf_rows(dataset, config, split, offset, length):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    return [r["row"] for r in _json(f"{API}/rows?{q}")["rows"]]


def hf_sample(dataset, config, split, n=POOL, seed=SEED, keep=None):
    """A seeded sample of n rows from the whole split (all rows if fewer).
    The split is read in 100-row pages (few requests, rate-limit friendly),
    filtered by `keep(row)`, then shuffled with a fixed seed."""
    total = hf_num_rows(dataset, config, split)
    rows = []
    for off in range(0, total, 100):
        rows.extend(hf_rows(dataset, config, split, off, 100))
        time.sleep(0.5)
    rows = [r for r in rows if keep is None or keep(r)]
    random.Random(seed).shuffle(rows)
    return rows[:n], total


def save(suite, name, items, source, license_, note=""):
    os.makedirs(os.path.join(OUT, suite), exist_ok=True)
    body = json.dumps({"source": source, "license": license_, "seed": SEED, "note": note, "items": items},
                      indent=1, ensure_ascii=False)
    path = os.path.join(OUT, suite, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body + "\n")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    print(f"  {suite}/{name}.json: {len(items)} items  [{license_}]  sha256:{digest}", flush=True)
    return {"suite": suite, "file": f"{name}.json", "items": len(items), "source": source, "license": license_,
            "sha256": digest}


LETTERS = "ABCDEFGHIJ"


# ----------------------------------------------------------------------------
# Suites
# ----------------------------------------------------------------------------
def fetch_medical():
    out = []
    rows, _ = hf_sample("GBaker/MedQA-USMLE-4-options", "default", "test")
    out.append(save("medical", "medqa_usmle", [
        {"id": f"medqa_{i}", "question": r["question"], "options": r["options"], "answer": r["answer_idx"]}
        for i, r in enumerate(rows)], "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options (test)", "CC-BY-4.0"))

    rows, _ = hf_sample("openlifescienceai/medmcqa", "default", "validation",
                        keep=lambda r: r.get("choice_type") == "single" and r.get("cop") in (0, 1, 2, 3))
    out.append(save("medical", "medmcqa", [
        {"id": f"medmcqa_{r['id']}", "question": r["question"],
         "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]},
         "answer": LETTERS[r["cop"]], "subject": r.get("subject_name")}
        for r in rows], "https://huggingface.co/datasets/openlifescienceai/medmcqa (validation; test has no keys)",
        "Apache-2.0"))

    rows, _ = hf_sample("qiaojin/PubMedQA", "pqa_labeled", "train")
    out.append(save("medical", "pubmedqa", [
        {"id": f"pubmedqa_{r['pubid']}", "question": r["question"],
         "context": "\n".join(r["context"]["contexts"]), "answer": r["final_decision"]}
        for r in rows], "https://huggingface.co/datasets/qiaojin/PubMedQA (pqa_labeled)", "MIT"))

    items = []
    subjects = ["anatomy", "clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine",
                "college_biology"]
    per = POOL // len(subjects) + 1
    for subj in subjects:
        rows, _ = hf_sample("cais/mmlu", subj, "test", n=per)
        items += [{"id": f"mmlu_{subj}_{i}", "subject": subj, "question": r["question"],
                   "options": dict(zip(LETTERS, r["choices"])), "answer": LETTERS[r["answer"]]}
                  for i, r in enumerate(rows)]
    random.Random(SEED).shuffle(items)
    out.append(save("medical", "mmlu_medical", items[:POOL], "https://huggingface.co/datasets/cais/mmlu (6 medical subjects, test)", "MIT"))
    return out


def fetch_reasoning():
    out = []
    rows, _ = hf_sample("TIGER-Lab/MMLU-Pro", "default", "test")
    out.append(save("reasoning", "mmlu_pro", [
        {"id": f"mmlupro_{r['question_id']}", "category": r["category"], "question": r["question"],
         "options": dict(zip(LETTERS, r["options"])), "answer": r["answer"]}
        for r in rows], "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro (test)", "MIT"))

    rows, _ = hf_sample("HuggingFaceH4/MATH-500", "default", "test")
    out.append(save("reasoning", "math500", [
        {"id": r["unique_id"], "subject": r["subject"], "level": r["level"], "problem": r["problem"], "answer": r["answer"]}
        for r in rows], "https://huggingface.co/datasets/HuggingFaceH4/MATH-500 (from Hendrycks MATH / PRM800K)", "MIT"))

    base = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh"
    tasks = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies",
             "logical_deduction_five_objects", "navigate", "object_counting", "temporal_sequences",
             "tracking_shuffled_objects_five_objects", "web_of_lies", "word_sorting"]
    items = []
    for t in tasks:
        examples = json.loads(_get(f"{base}/{t}.json").decode("utf-8"))["examples"]
        random.Random(SEED).shuffle(examples)
        items += [{"id": f"bbh_{t}_{i}", "task": t, "input": e["input"], "answer": e["target"]}
                  for i, e in enumerate(examples[:POOL // len(tasks) + 1])]
    random.Random(SEED).shuffle(items)
    out.append(save("reasoning", "bbh", items[:POOL], "https://github.com/suzgunmirac/BIG-Bench-Hard (12 tasks)", "MIT"))
    return out


def fetch_coding():
    out = []
    # Python: the original HumanEval (MultiPL-E translates it to the others).
    rows, _ = hf_sample("openai/openai_humaneval", "openai_humaneval", "test")
    out.append(save("coding", "multipl_e_python", [
        {"id": r["task_id"], "language": "python", "prompt": r["prompt"],
         "tests": r["test"] + f"\n\ncheck({r['entry_point']})\n", "entry_point": r["entry_point"]}
        for r in rows], "https://huggingface.co/datasets/openai/openai_humaneval (test)", "MIT"))
    langs = {"typescript": "humaneval-ts", "go": "humaneval-go", "java": "humaneval-java",
             "rust": "humaneval-rs", "csharp": "humaneval-cs"}
    for lang, cfg in langs.items():
        rows, _ = hf_sample("nuprl/MultiPL-E", cfg, "test")
        out.append(save("coding", f"multipl_e_{lang}", [
            {"id": r["name"], "language": lang, "prompt": r["prompt"], "tests": r["tests"],
             "stop_tokens": r.get("stop_tokens", [])}
            for r in rows], f"https://huggingface.co/datasets/nuprl/MultiPL-E ({cfg}; HumanEval MIT)", "MIT"))
    return out


def fetch_writing():
    rows, _ = hf_sample("google/IFEval", "default", "train", n=1000)
    return [save("writing", "ifeval_full", [
        {"id": f"ifeval_{r['key']}", "prompt": r["prompt"], "instruction_ids": r["instruction_id_list"],
         "kwargs": r["kwargs"]}
        for r in rows], "https://huggingface.co/datasets/google/IFEval (all prompts)", "Apache-2.0")]


def fetch_personal():
    out = []
    raw = _get("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    random.Random(SEED).shuffle(rows)
    out.append(save("personal", "simpleqa", [
        {"id": f"simpleqa_{i}", "question": r["problem"], "answer": r["answer"],
         "topic": ast.literal_eval(r["metadata"]).get("topic") if r.get("metadata") else None}
        for i, r in enumerate(rows[:POOL])], "https://github.com/openai/simple-evals (SimpleQA test set)", "MIT"))

    rows, _ = hf_sample("natolambert/xstest-v2-copy", "default", "gpt4", n=1000)
    seen, items = set(), []
    for r in rows:
        if r["prompt"] in seen:
            continue
        seen.add(r["prompt"])
        items.append({"id": f"xstest_{r['id']}", "prompt": r["prompt"], "type": r["type"],
                      "should_refuse": r["type"].startswith("contrast_")})
    random.Random(SEED).shuffle(items)
    out.append(save("personal", "xstest", items[:POOL], "https://huggingface.co/datasets/natolambert/xstest-v2-copy (XSTest v2)", "CC-BY-4.0"))
    return out


# CC-BY-4.0 LegalBench tasks with exact-match answers (licenses verified per
# task at https://github.com/HazyResearch/legalbench/tree/main/tasks).
LEGAL_TASKS = ["hearsay", "personal_jurisdiction", "ucc_v_common_law", "insurance_policy_interpretation",
               "contract_qa", "contract_nli_confidentiality_of_agreement", "cuad_governing_law",
               "cuad_non-compete", "cuad_termination_for_convenience", "cuad_cap_on_liability",
               "cuad_audit_rights", "cuad_change_of_control"]


def fetch_legal():
    items = []
    per = POOL // len(LEGAL_TASKS) + 1
    for task in LEGAL_TASKS:
        license_line = _get(f"https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks/{task}/README.md").decode("utf-8")
        if "by 4.0" not in license_line.lower() or "-nc" in license_line.lower() or "by-sa" in license_line.lower():
            print(f"  skipping {task}: license is not CC-BY-4.0")
            continue
        prompt_tpl = _get(f"https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks/{task}/base_prompt.txt").decode("utf-8")
        rows, _ = hf_sample("nguha/legalbench", task, "test", n=per)
        for i, r in enumerate(rows):
            fields = {k: v for k, v in r.items() if k not in ("answer", "index")}
            items.append({"id": f"legal_{task}_{i}", "task": task, "prompt_template": prompt_tpl,
                          "fields": fields, "answer": str(r["answer"]).strip()})
    random.Random(SEED).shuffle(items)
    return [save("legal", "legalbench", items[:POOL], "https://huggingface.co/datasets/nguha/legalbench (CC-BY-4.0 tasks only)",
                 "CC-BY-4.0", note="tasks: " + ", ".join(LEGAL_TASKS))]


# CUAD clause categories with clear, quotable answers (Test 47 and the
# workload test). Names match the CUAD question text.
CUAD_CATEGORIES = ["Governing Law", "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
                   "Termination For Convenience", "Non-Compete", "Exclusivity", "Change Of Control",
                   "Anti-Assignment", "Cap On Liability", "Liquidated Damages", "Audit Rights", "Insurance",
                   "Most Favored Nation", "Minimum Commitment", "Warranty Duration", "Ip Ownership Assignment",
                   "Non-Disparagement", "Covenant Not To Sue", "Post-Termination Services"]
CUAD_URL = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"
CUAD_CONTRACTS = 45          # about 500K tokens in all: enough for a 200K-token workload prompt per user
CUAD_REVIEW_ITEMS = 30       # 20 with the clause present, 10 without
CUAD_MIN_CHARS, CUAD_MAX_CHARS = 15000, 90000   # about 4K-22K tokens per contract


def _cuad_question(q):
    """(category, details) from a CUAD question: 'Highlight the parts (if any)
    of this contract related to "X" ... Details: Y'."""
    m = re.search(r'related to "([^"]+)"', q)
    details = q.split("Details:", 1)[1].strip() if "Details:" in q else ""
    return (m.group(1) if m else None), details


def fetch_contracts():
    """CUAD v1 (CC BY 4.0): full commercial contracts with lawyer-labelled
    clauses. Saves the contracts once (legal/cuad_contracts.json) and the
    review questions that point into them (legal/cuad_review.json)."""
    data = json.loads(_get(CUAD_URL).decode("utf-8"))
    contracts = []
    for doc in data["data"]:
        para = doc["paragraphs"][0]
        text = para["context"]
        if not CUAD_MIN_CHARS <= len(text) <= CUAD_MAX_CHARS:
            continue
        clauses, details = {}, {}
        for qa in para["qas"]:
            cat, det = _cuad_question(qa["question"])
            if cat in CUAD_CATEGORIES:
                clauses[cat] = sorted({a["text"].strip() for a in qa["answers"] if a["text"].strip()})
                details[cat] = det
        contracts.append({"title": doc["title"], "text": text, "clauses": clauses, "details": details})
    random.Random(SEED).shuffle(contracts)
    contracts = contracts[:CUAD_CONTRACTS]
    for i, c in enumerate(contracts):
        c["id"] = f"cuad_{i}"

    rnd = random.Random(SEED + 1)
    items = []
    want_present, want_absent = CUAD_REVIEW_ITEMS * 2 // 3, CUAD_REVIEW_ITEMS - CUAD_REVIEW_ITEMS * 2 // 3
    for c in contracts:
        if len(items) >= CUAD_REVIEW_ITEMS:
            break
        present = [k for k, v in c["clauses"].items() if v]
        absent = [k for k, v in c["clauses"].items() if not v]
        n_present = sum(1 for it in items if it["answers"])
        if n_present < want_present and present:
            pool = present
        elif len(items) - n_present < want_absent and absent:
            pool = absent
        else:
            continue
        cat = rnd.choice(sorted(pool))
        items.append({"id": f"cuad_review_{len(items)}", "contract_id": c["id"], "category": cat,
                      "question": f"{cat}: {c['details'][cat]}", "answers": c["clauses"][cat]})
    for c in contracts:
        c["questions"] = {cat: f"{cat}: {det}" for cat, det in c.pop("details").items()}
    src = "https://huggingface.co/datasets/theatticusproject/cuad (CUAD v1)"
    return [save("legal", "cuad_contracts", contracts, src, "CC-BY-4.0",
                 note=f"{len(contracts)} contracts of {CUAD_MIN_CHARS}-{CUAD_MAX_CHARS} characters; clauses for: "
                      + ", ".join(CUAD_CATEGORIES)),
            save("legal", "cuad_review", items, src, "CC-BY-4.0",
                 note="one clause question per contract; answers [] means the clause is absent")]


def _bill_points(summary):
    """The numbered points '(1) ... (2) ...' of a BillSum summary."""
    parts = re.split(r"\(\d+\)\s*", summary)
    return [re.sub(r"\s+", " ", p).strip(" ;.") for p in parts[1:] if len(p.strip()) > 20]


def fetch_summaries():
    """BillSum (CC0): US Congressional bills with human summaries. Keeps
    bills of 8K-30K characters whose summary has 3-8 numbered points (the
    judge's coverage checklist)."""
    def keep(r):
        return 8000 <= len(r["text"]) <= 30000 and 3 <= len(_bill_points(r["summary"])) <= 8
    rows, total = hf_sample("FiscalNote/billsum", "default", "test", n=30, keep=keep)
    items = [{"id": f"billsum_{i}", "title": r["title"], "text": r["text"], "summary": r["summary"],
              "points": _bill_points(r["summary"])[:6]} for i, r in enumerate(rows)]
    return [save("writing", "billsum", items, "https://huggingface.co/datasets/FiscalNote/billsum (test)", "CC0-1.0",
                 note=f"sampled from {total} test bills; points = numbered items of the reference summary (max 6)")]


LLMBAR_URL = "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar/"


def fetch_judge_validation():
    """Public expert-rated data to check the LLM judge (judge_validation.py):
    SummEval (MIT): summaries rated by experts for consistency with the
    source (fully consistent = all experts 5/5; inconsistent = average 3 or
    less). LLMBar (MIT): instruction-following answer pairs where experts
    marked the better one (natural and adversarial sets)."""
    rows, _ = hf_sample("mteb/summeval", "default", "test", n=100)
    good, bad = [], []
    for r in rows:
        for summary, cons in zip(r["machine_summaries"], r["consistency"]):
            if cons >= 5.0:
                good.append((r, summary, cons))
            elif cons <= 3.0:
                bad.append((r, summary, cons))
    rnd = random.Random(SEED)
    rnd.shuffle(good)
    rnd.shuffle(bad)

    def pick(pool, n):
        out, seen = [], set()
        for r, summary, cons in pool:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append((r, summary, cons))
            if len(out) == n:
                break
        return out

    items = [{"id": f"summeval_{i}", "source": r["text"], "summary": summary, "expert_consistency": round(cons, 2),
              "consistent": cons >= 5.0}
             for i, (r, summary, cons) in enumerate(pick(good, 20) + pick(bad, 20))]
    out = [save("judge_validation", "summeval", items, "https://huggingface.co/datasets/mteb/summeval (test)", "MIT",
                note="20 summaries all experts rated 5/5 for consistency, 20 rated 3 or less on average")]
    pairs = []
    for subset, n in (("Natural", 20), ("Adversarial/Neighbor", 20)):
        data = json.loads(_get(LLMBAR_URL + subset + "/dataset.json").decode("utf-8"))
        random.Random(SEED).shuffle(data)
        pairs += [{"id": f"llmbar_{subset.split('/')[-1].lower()}_{i}", "subset": subset, "instruction": d["input"],
                   "output_1": d["output_1"], "output_2": d["output_2"], "better": d["label"]}
                  for i, d in enumerate(data[:n])]
    out.append(save("judge_validation", "llmbar", pairs, "https://github.com/princeton-nlp/LLMBar (Natural, Adversarial/Neighbor)",
                    "MIT", note="experts marked the output that follows the instruction better"))
    return out


SUITES = {"medical": fetch_medical, "reasoning": fetch_reasoning, "coding": fetch_coding,
          "writing": fetch_writing, "personal": fetch_personal, "legal": fetch_legal,
          "contracts": fetch_contracts, "summaries": fetch_summaries, "judge_validation": fetch_judge_validation}


def write_sources_md(_entries=None):
    """Rebuilds SOURCES.md from the data files on disk (so a partial run can
    never leave it out of sync)."""
    rows = []
    for suite in sorted(os.listdir(OUT)):
        folder = os.path.join(OUT, suite)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
            digest = hashlib.sha256(raw.rstrip("\n").encode("utf-8")).hexdigest()[:16]
            rows.append(f"| {suite} | {name} | {len(data['items'])} | {data.get('license')} | {data.get('source')} | {digest} |\n")
    with open(os.path.join(OUT, "SOURCES.md"), "w", encoding="utf-8") as f:
        f.write("# Benchmark data sources\n\nSeeded samples (seed %d) of official public benchmarks, fetched by "
                "`scripts/fetch_benchmark_data.py`, plus original tasks written for this repository. Each official "
                "item keeps its original answer key and is redistributed under the license listed; attribution belongs "
                "to the original dataset authors. GPQA is deliberately not included (its authors ask that items not be "
                "published).\n\n" % SEED)
        f.write("| Suite | File | Items | License | Source | sha256 (16) |\n|---|---|---|---|---|---|\n")
        f.writelines(rows)


def main(argv):
    wanted = argv or list(SUITES)
    entries = []
    try:
        for suite in wanted:
            print(f"[{suite}]", flush=True)
            entries += SUITES[suite]()
    finally:
        write_sources_md()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
