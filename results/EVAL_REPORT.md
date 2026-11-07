# Evaluation Report: Qwen3.8-Flash-Next-FP4

This report shows a full run of [llm-eval-suite](../README.md) against one model server. It covers the serving tests (parallel capacity, prefill, decode, full context) and the model-quality tests (coding in six languages, reasoning, medical, writing). The [README](../README.md#what-gets-tested) describes each test.

> [!NOTE]
> Single runs on this server vary. The same temperature-0 prompt gave 5 different texts in 5 runs. See [Phased and sequential runs compared](#phased-and-sequential-runs-compared).

## Contents

- [Run details](#run-details)
- [Result](#result)
- [Schedule](#schedule)
- [Test 3: serving capacity](#test-3-serving-capacity)
- [Test 12: full context](#test-12-full-context)
- [Results by test](#results-by-test)
- [Tests 20-25 by language](#tests-20-25-by-language)
- [Phased and sequential runs compared](#phased-and-sequential-runs-compared)
- [How to repeat this run](#how-to-repeat-this-run)

## Run details

| Item | Value |
|---|---|
| Model | `Qwen3.8-Flash-Next-FP4` |
| Endpoint | `http://<model-server>:8080/v1` (vLLM, configured context 524,288 tokens) |
| Judge | `qwen3.8-27b` at `http://127.0.0.1:8888/v1` |
| Suite | `llm-eval-suite` at `16a0528` plus uncommitted changes (phased scheduler, new Tests 3 and 12) |
| Date | 2026-11-06 |
| Docker image | Ubuntu 24.04 toolchains (Go 1.22, Rust 1.75, Node.js 22, OpenJDK 21, TypeScript 5.7). The image was upgraded after this run; see the [CHANGELOG](../CHANGELOG.md). |
| Main run | 14:56:40 to 15:18:28, **1,307 s (21 min 47 s)**, phased schedule |
| Tests in the main run | 1-11, 14, 16-31. Tests 20-25 in all six languages. |
| Not in the main run | Test 12 (it ran alone earlier, see [Test 12](#test-12-full-context)). Tests 13 and 15. |

## Result

| Metric | Value |
|---|---|
| Tests scored | 27 of 28 (Test 31 SKIPPED: the endpoint serves one model) |
| PASS / PARTIAL / FAIL | **25 / 2 / 0** |
| Effectiveness | 96.3% |
| Capability score | 95.2% (19 of 21 model-quality tests), grade A+ |
| Completion rate | 100% (no infrastructure failures) |
| Decode speed, isolated tests | 50.2 tok/s |
| Time to first token, isolated tests | 387 ms |
| Tokens generated | 118,536 (4,150 per passed test) |
| Efficiency index | 80.2 / 100 |

The two PARTIAL results are Test 20 (Go and Rust diffs failed) and Test 23 (Go unit tests failed).

## Schedule

| Phase | Wall time | Content |
|---|---|---|
| Isolated | 299 s | Tests 3, 1, 7, 8, 9, 31, one at a time |
| Parallel | 1,008 s | 52 jobs. At most 4 requests to the model (the count Test 3 verified). At most 2 requests to the judge (from its `/health`). |

The speed and time-to-first-token values of parallel-phase tests are measured under load. The summary averages use the isolated tests only.

## Test 3: serving capacity

Prefill, one uncached request at a time:

| Target tokens | Prompt tokens | Time to first token | Prefill speed |
|---|---|---|---|
| 2,000 | 2,143 | 1.02 s | 2,101 tok/s |
| 4,000 | 4,324 | 1.91 s | 2,259 tok/s |
| 8,000 | 8,655 | 3.43 s | 2,521 tok/s |
| 16,000 | 17,315 | 6.48 s | 2,673 tok/s |
| 32,000 | 34,669 | 12.95 s | **2,676 tok/s** |

Concurrency sweep, 512 tokens per stream:

| Clients | Total decode | Decode per stream | Mean wait for first token | Max wait | Queued | Server running / waiting |
|---|---|---|---|---|---|---|
| 1 | 34.3 tok/s | 34.3 tok/s | 541 ms | 541 ms | 0 | 1 / 0 |
| 2 | 51.4 tok/s | 25.8 tok/s | 863 ms | 932 ms | 0 | 2 / 0 |
| 4 | **87.6 tok/s** | 22.0 tok/s | 1,271 ms | 1,299 ms | 0 | 4 / 0 |
| 8 | 77.3 tok/s | 20.3 tok/s | 14,477 ms | 29,769 ms | 4 | 4 / 4 |
| 16 | 77.6 tok/s | 20.3 tok/s | 40,811 ms | 83,330 ms | 12 | 4 / 12 |

Verified maximum: **4 parallel requests**. The server does not report its slot count. Its queue counters show 4 running at most, with the other requests waiting. Status: PASS.

## Test 12: full context

Test 12 ran alone on 2026-11-06 at 14:02, together with Test 3. That run took 1,865 s. Each step hides a value at 85% depth, and the final answer must repeat it.

| Target tokens | Prompt tokens | Time to first token | Prefill speed | Decode speed | Outcome |
|---|---|---|---|---|---|
| 4,000 | 4,021 | 1.6 s | 2,586 tok/s | 46.6 tok/s | recalled |
| 8,000 | 7,942 | 3.3 s | 2,415 tok/s | 45.9 tok/s | recalled |
| 16,000 | 15,953 | 6.2 s | 2,560 tok/s | 46.2 tok/s | recalled |
| 32,000 | 31,930 | 12.3 s | 2,597 tok/s | 47.7 tok/s | recalled |
| 64,000 | 63,924 | 24.4 s | 2,619 tok/s | 48.4 tok/s | recalled |
| 128,000 | 127,963 | 49.4 s | 2,593 tok/s | 50.9 tok/s | recalled |
| 256,000 | 255,945 | 102.7 s | 2,492 tok/s | 50.0 tok/s | recalled |
| 513,800 (98%) | 513,745 | 220.2 s | 2,333 tok/s | 51.0 tok/s | recalled |

Status: PASS. The model recalled the value at every step up to 98% of the configured context.

Capacity check (report only, it does not change the status):

| Requests at once | Tokens each | Server max running / waiting | Preemptions | Recalled | Longest wait for first token | Wall time | Run together |
|---|---|---|---|---|---|---|---|
| 4 | 513,800 | 4 / 2 | 2 | 4 of 4 | 945.6 s | 950.9 s | no |
| 4 | 128,400 | 4 / 1 | 0 | 4 of 4 | 206.5 s | 211.6 s | yes |

The vLLM KV cache holds 1,119,425 tokens, enough for 2.14 requests at the maximum length. Four full-context requests do not fit at the same time. Four requests of 128,400 tokens do.

## Results by test

| # | Test | Status | Result |
|---|---|---|---|
| 1 | Streaming | PASS | First token in 196 ms, 34.6 tok/s |
| 2 | Vision (invoice) | PASS | 12 of 12 fields |
| 3 | Serving capacity | PASS | 4 parallel, 87.6 tok/s total, prefill 2,676 tok/s |
| 4 | Four-task suite | PASS | 4 of 4 tasks |
| 5 | Tool calling | PASS | Recovered from the injected error |
| 6 | JSON schema | PASS | Output matched the schema |
| 7 | Prefix cache | PASS (ACTIVE) | 2.1× faster: 5.2 s cold, 2.4 s warm, 6,592 tokens cached |
| 8 | Client abort | PASS | Next request served in 947 ms |
| 9 | Stop words, temperature 0 | PASS | Stopped at the stop word, same output on 3 runs |
| 10 | Precision ledger | PASS | Exact match |
| 11 | API errors | PASS | HTTP 400/422 on invalid requests |
| 14 | Algorithm fuzz | PASS | 5,000 operations without error |
| 16 | CRUXEval | PASS | 2 of 2 |
| 17 | SWE-bench patch | PASS | Regression tests pass |
| 18 | AIME | PASS | 3 of 3 |
| 19 | GPQA Diamond | PASS | 5 of 5 |
| 20 | Surgical diff | PARTIAL | 4 of 6 languages |
| 21 | Multi-file | PASS | 6 of 6 languages |
| 22 | Security review | PASS | 6 of 6 languages |
| 23 | Unit-test writing | PARTIAL | 5 of 6 languages |
| 24 | Strict typing | PASS | 6 of 6 languages |
| 25 | Safe refactor | PASS | 6 of 6 languages |
| 26 | Prompt injection | PASS | Invariants protected |
| 27 | IFEval | PASS | 5 of 5 |
| 28 | Medical | PASS | 10 of 10 |
| 29 | GSM8K | PASS | 5 of 5 |
| 30 | MMLU History | PASS | 5 of 5 |
| 31 | Multi-adapter | SKIPPED | The endpoint serves one model |

## Tests 20-25 by language

| Language | Diff (20) | Multi-file (21) | Security (22) | Unit tests (23) | Typing (24) | Refactor (25) | Passed |
|---|---|---|---|---|---|---|---|
| Python | PASS | PASS | PASS | PASS | PASS | PASS | 6 of 6 |
| TypeScript | PASS | PASS | PASS | PASS | PASS | PASS | 6 of 6 |
| Go | PARTIAL | PASS | PASS | PARTIAL | PASS | PASS | 4 of 6 |
| Java | PASS | PASS | PASS | PASS | PASS | PASS | 6 of 6 |
| Rust | PARTIAL | PASS | PASS | PASS | PASS | PASS | 5 of 6 |
| C# | PASS | PASS | PASS | PASS | PASS | PASS | 6 of 6 |

## Phased and sequential runs compared

The same tests ran again at 15:18:28 with `--sequential`.

| | Phased | Sequential |
|---|---|---|
| Wall time | **1,307 s** | 2,421 s |
| PASS / PARTIAL / FAIL | 25 / 2 / 0 | 25 / 2 / 0 |
| PARTIAL tests | 20, 23 | 20, 23 |
| Effectiveness | 96.3% | 95.6% |
| Decode speed in the summary | 50.2 tok/s (isolated tests) | 48.6 tok/s (all tests) |
| Time to first token in the summary | 387 ms (isolated tests) | 2,723 ms (all tests) |

The phased run was 1.85 times faster. The test results were the same. Three language results differed:

| Language and test | Phased | Sequential |
|---|---|---|
| C# diff (20) | PASS | PARTIAL |
| C# unit tests (23) | PASS | PARTIAL |
| Go unit tests (23) | PARTIAL | PASS |

The differences go in both directions. This server gave 5 different texts for the same temperature-0 prompt in 5 runs, with and without other load. One run cannot separate a model difference from this variation. Use the qualification suites (`--repeats`) to compare models.

## How to repeat this run

Build the image as the [README](../README.md#quick-start-docker) shows, then run:

```bash
# Tests 3 and 12 alone
docker run --rm --network host \
  -v "$PWD/results:/app/results" -v "$PWD/logs:/app/logs" \
  llm-eval-suite:latest -e http://<model-server>:8080/v1 --test 3,12

# Main run: core tests without 12, 13 and 15, six languages, separate judge
docker run --rm --network host \
  -v "$PWD/results:/app/results" -v "$PWD/logs:/app/logs" \
  llm-eval-suite:latest -e http://<model-server>:8080/v1 \
  --exclude 12,13,15 --language polyglot \
  --judge-endpoint http://127.0.0.1:8888/v1

# The same tests, one at a time
docker run --rm --network host \
  -v "$PWD/results:/app/results" -v "$PWD/logs:/app/logs" \
  llm-eval-suite:latest -e http://<model-server>:8080/v1 \
  --exclude 12,13,15 --language polyglot \
  --judge-endpoint http://127.0.0.1:8888/v1 --sequential
```

Change the endpoints to your own servers.

## Files

These files are local. git ignores them because they contain internal IP addresses.

- Main run: `results/stage4_phased.json`, `logs/stage4_phased.log`
- Sequential run: `results/stage4_sequential.json`, `logs/stage4_sequential.log`
- Tests 3 and 12 alone: `results/stage3_test3_12.json`, `logs/stage3.log`
