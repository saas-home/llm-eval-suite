#!/usr/bin/env python3
"""
Shared LLM HTTP client & terminal formatting utilities for llm-eval-suite.

Zero external dependencies (Python stdlib only). Used by eval.py, compare.py,
judge_eval.py, and reporting.py to avoid duplicated protocol code.
"""

import os
import sys
import time
import json
import re
import random
import atexit
import urllib.request
import urllib.error
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_image_path():
    for rel in [os.path.join("assets", "invoice.png"), os.path.join("assets", "image.png"), "invoice.png", "image.png"]:
        p = os.path.join(BASE_DIR, rel)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, "assets", "invoice.png")


IMAGE_PATH = _resolve_image_path()
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
os.makedirs(DEFAULT_LOGS_DIR, exist_ok=True)

# ANSI terminal colors (respect NO_COLOR env and non-TTY pipes)
NO_COLOR = bool(os.getenv("NO_COLOR")) or not sys.stdout.isatty()
BOLD = "" if NO_COLOR else "\033[1m"
DIM = "" if NO_COLOR else "\033[2m"
GREEN = "" if NO_COLOR else "\033[32m"
YELLOW = "" if NO_COLOR else "\033[33m"
RED = "" if NO_COLOR else "\033[31m"
CYAN = "" if NO_COLOR else "\033[36m"
MAGENTA = "" if NO_COLOR else "\033[35m"
RESET = "" if NO_COLOR else "\033[0m"
CLEAR_LINE = "" if NO_COLOR else "\033[K"

_LOG_FILE_HANDLE = None


def init_file_logger(log_file_path: str = None) -> str:
    """Initializes diagnostic file logger in results/logs directory for debugging and diagnosis."""
    global _LOG_FILE_HANDLE
    if not log_file_path:
        log_file_path = os.path.join(DEFAULT_LOGS_DIR, f"eval_diag_{time.strftime('%Y%m%d_%H%M%S')}.log")
    try:
        close_file_logger()
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        _LOG_FILE_HANDLE = open(log_file_path, "a", encoding="utf-8")
        return log_file_path
    except Exception:
        return ""


def close_file_logger():
    """Flushes and cleanly closes the active diagnostic log file handle."""
    global _LOG_FILE_HANDLE
    if _LOG_FILE_HANDLE:
        try:
            _LOG_FILE_HANDLE.flush()
            _LOG_FILE_HANDLE.close()
        except Exception:
            pass
        _LOG_FILE_HANDLE = None


atexit.register(close_file_logger)


def log(msg, bold=False, color=""):
    prefix = bold and BOLD or ""
    c = color or ""
    suffix = (bold or color) and RESET or ""
    print(f"{prefix}{c}{msg}{suffix}", flush=True)
    if _LOG_FILE_HANDLE:
        try:
            clean_msg = re.sub(r"\033\[[0-9;]*m", "", str(msg))
            _LOG_FILE_HANDLE.write(clean_msg + "\n")
            _LOG_FILE_HANDLE.flush()
        except Exception:
            pass


# --- Retry policy (backoff + rate-limit handling) --------------------------
# Transient server errors (5xx) and connection/timeout errors get a short,
# jittered exponential backoff. HTTP 429 (rate limit) gets a separate, more
# patient budget and honors the server's Retry-After hint, because a per-minute
# quota outlasts the generic backoff window.
_RETRY_BASE_SECONDS = 0.5
_RETRY_CAP_SECONDS = 8.0
_RATE_LIMIT_BASE_SECONDS = 2.0
_RATE_LIMIT_CAP_SECONDS = 60.0
RATE_LIMIT_EXTRA_RETRIES = 4


def _parse_retry_after(http_error):
    """Parse a Retry-After header (seconds form) from an HTTPError, else None."""
    try:
        raw = http_error.headers.get("Retry-After")
    except Exception:
        raw = None
    if not raw:
        return None
    try:
        val = float(raw.strip())
    except ValueError:
        return None  # HTTP-date form; fall back to computed backoff
    if 0 <= val <= _RATE_LIMIT_CAP_SECONDS:
        return val
    return None


def _backoff_delay(attempt, status_code=None, retry_after=None):
    """Exponential backoff with jitter; honors Retry-After for rate limits."""
    if status_code == 429:
        if retry_after is not None:
            return retry_after
        window = min(_RATE_LIMIT_BASE_SECONDS * (2 ** attempt), _RATE_LIMIT_CAP_SECONDS)
        return window / 2 + random.uniform(0.0, window / 2)  # half jitter
    window = min(_RETRY_BASE_SECONDS * (2 ** attempt), _RETRY_CAP_SECONDS)
    return random.uniform(0.0, window)  # full jitter


def normalize_endpoint(endpoint: str) -> str:
    """Normalizes an endpoint URL (strips trailing slashes, /chat/completions, and appends /v1 to bare hosts)."""
    if not endpoint:
        return ""
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/chat/completions"):
        endpoint = endpoint[:-len("/chat/completions")].rstrip("/")
    try:
        parsed = urlparse(endpoint)
        if not parsed.path:
            endpoint = f"{endpoint}/v1"
    except Exception:
        pass
    return endpoint


class LLMClient:
    """OpenAI-compatible chat completions client (non-streaming + SSE streaming)."""

    def __init__(self, endpoint: str, api_key: str = None, model: str = None):
        self.base_url = normalize_endpoint(endpoint)
        self.completions_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        if self.base_url.endswith("/v1"):
            self.health_url = f"{self.base_url[:-3]}/health"
            self.metrics_url = f"{self.base_url[:-3]}/metrics"
        else:
            self.health_url = f"{self.base_url}/health"
            self.metrics_url = f"{self.base_url}/metrics"
        self.api_key = api_key
        self.model = model

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def fetch_models(self):
        req = urllib.request.Request(self.models_url, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list):
                    return data
                models = data.get("data", [])
                if not models and "models" in data:
                    models = data.get("models", [])
                return models if isinstance(models, list) else []
        except Exception as e:
            log(f"Warning: Could not fetch models from {self.models_url}: {e}", color=YELLOW)
            return []

    def discover_model(self, fallback="default"):
        """Returns the first discovered model ID/name, or fallback."""
        models = self.fetch_models()
        if models and isinstance(models[0], dict):
            return models[0].get("id") or models[0].get("name") or fallback
        return fallback

    def fetch_health(self):
        req = urllib.request.Request(self.health_url, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def fetch_server_metrics(self):
        """
        Scrapes and parses Prometheus metrics from /metrics (e.g. vLLM, TGI, SGLang).
        Returns a dictionary of key metrics or None if not supported/available.
        """
        req = urllib.request.Request(self.metrics_url, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8")
        except Exception:
            return None

        metrics = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            metric_key = parts[0]
            val_str = parts[1]
            try:
                val = float(val_str)
            except ValueError:
                continue

            if metric_key.startswith("vllm:num_requests_running"):
                metrics["requests_running"] = int(val)
            elif metric_key.startswith("vllm:num_requests_waiting_by_reason"):
                if 'reason="deferred"' in metric_key:
                    metrics["requests_deferred"] = int(val)
                elif 'reason="capacity"' in metric_key:
                    metrics["requests_capacity"] = int(val)
            elif metric_key.startswith("vllm:num_requests_waiting"):
                metrics["requests_waiting"] = int(val)
            elif metric_key.startswith("vllm:kv_cache_usage_perc"):
                metrics["kv_cache_usage_pct"] = round(val * 100.0, 2)
            elif metric_key.startswith("vllm:num_preemptions_total"):
                metrics["preemptions_total"] = int(val)
            elif metric_key.startswith("process_resident_memory_bytes"):
                metrics["process_rss_mb"] = round(val / (1024 * 1024), 2)
        return metrics if metrics else None

    def call(self, messages, max_tokens=1024, temperature=0.0, stream=False,
             tools=None, tool_choice=None, response_format=None, stop=None,
             seed=None, timeout=600, chat_template_kwargs=None, reasoning_effort=None,
             retries=2, chunk_callback=None, model=None):
        """
        Sends a chat completion request. Retries transient failures (5xx,
        timeouts, connection errors) up to `retries` additional attempts.
        Client errors (4xx) are returned immediately without retry.
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if stream:
            payload["stream_options"] = {"include_usage": True}

        retry_index = 0
        rate_limit_retries = 0
        while True:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(self.completions_url, data=data, headers=self._get_headers())
            t0 = time.perf_counter()
            try:
                if not stream:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                    total_time = time.perf_counter() - t0
                    if isinstance(res_json, dict) and res_json.get("error"):
                        err = res_json["error"]
                        err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        return {
                            "error": f"API Error: {err_msg}",
                            "failure_kind": "client" if any(w in err_msg.lower() for w in ["invalid", "not found", "bad request"]) else "infra",
                            "text": "",
                            "content": "",
                            "reasoning": "",
                            "tool_calls": None,
                            "total_time": total_time,
                            "ttft": total_time,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "cached_tokens": 0,
                            "decode_speed": 0.0,
                            "raw": res_json
                        }
                    choices = res_json.get("choices") or [{}]
                    choice = choices[0] if choices else {}
                    if choice.get("finish_reason") == "error":
                        err_msg = choice.get("message", {}).get("content", "Server error during generation")
                        return {
                            "error": f"Server finish_reason error: {err_msg}",
                            "failure_kind": "infra",
                            "text": "",
                            "content": "",
                            "reasoning": "",
                            "tool_calls": None,
                            "total_time": total_time,
                            "ttft": total_time,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "cached_tokens": 0,
                            "decode_speed": 0.0,
                            "raw": res_json
                        }
                    msg = choice.get("message", {})
                    content = msg.get("content") or ""
                    reasoning = (
                        msg.get("reasoning_content")
                        or msg.get("reasoning")
                        or msg.get("thought")
                        or ""
                    )
                    full_text = f"{reasoning}\n{content}".strip() if reasoning else (content or "")
                    tool_calls = msg.get("tool_calls")
                    usage = res_json.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    speed = completion_tokens / total_time if total_time > 0 else 0.0
                    return {
                        "content": content if (content or not reasoning) else "",
                        "reasoning": reasoning,
                        "text": full_text,
                        "tool_calls": tool_calls,
                        "total_time": total_time,
                        "ttft": total_time,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cached_tokens": cached_tokens,
                        "decode_speed": speed,
                        "raw": res_json
                    }
                else:
                    t_first = None
                    t_last = None
                    chunks = []
                    reasoning_chunks = []
                    content_chunks = []
                    token_count = 0
                    prompt_tokens = 0
                    completion_tokens = 0
                    cached_tokens = 0
                    tool_calls = None
                    tool_calls_acc = {}  # Accumulate streamed tool call deltas by index
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        for line in resp:
                            l = line.decode("utf-8", errors="replace").strip()
                            if not l.startswith("data:"):
                                continue
                            d_str = l[5:].strip()
                            if d_str == "[DONE]":
                                break
                            try:
                                c = json.loads(d_str)
                            except Exception:
                                continue
                            if isinstance(c, dict) and c.get("error"):
                                err = c["error"]
                                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                                return {
                                    "error": f"Stream error: {err_msg}",
                                    "failure_kind": "infra",
                                    "text": "".join(chunks),
                                    "content": "".join(content_chunks),
                                    "reasoning": "".join(reasoning_chunks),
                                    "tool_calls": None,
                                    "total_time": time.perf_counter() - t0,
                                    "ttft": (t_first - t0) if t_first else (time.perf_counter() - t0),
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens or token_count,
                                    "cached_tokens": cached_tokens,
                                    "decode_speed": 0.0
                                }
                            if "usage" in c and c["usage"]:
                                prompt_tokens = c["usage"].get("prompt_tokens", prompt_tokens)
                                completion_tokens = c["usage"].get("completion_tokens", completion_tokens)
                                cached_tokens = (c["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                            choices = c.get("choices", [])
                            if choices:
                                if choices[0].get("finish_reason") == "error":
                                    err_msg = choices[0].get("message", {}).get("content", "Server error during streaming")
                                    return {
                                        "error": f"Stream finish_reason error: {err_msg}",
                                        "failure_kind": "infra",
                                        "text": "".join(chunks),
                                        "content": "".join(content_chunks),
                                        "reasoning": "".join(reasoning_chunks),
                                        "tool_calls": None,
                                        "total_time": time.perf_counter() - t0,
                                        "ttft": (t_first - t0) if t_first else (time.perf_counter() - t0),
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens or token_count,
                                        "cached_tokens": cached_tokens,
                                        "decode_speed": 0.0
                                    }
                                delta = choices[0].get("delta", {})
                                if "tool_calls" in delta and delta["tool_calls"]:
                                    for tc_delta in delta["tool_calls"]:
                                        tc_idx = tc_delta.get("index", 0)
                                        if tc_idx not in tool_calls_acc:
                                            tool_calls_acc[tc_idx] = {
                                                "id": tc_delta.get("id", ""),
                                                "type": tc_delta.get("type", "function"),
                                                "function": {"name": "", "arguments": ""}
                                            }
                                        if tc_delta.get("id"):
                                            tool_calls_acc[tc_idx]["id"] = tc_delta.get("id")
                                        fn_delta = tc_delta.get("function", {})
                                        if fn_delta.get("name"):
                                            tool_calls_acc[tc_idx]["function"]["name"] = fn_delta.get("name")
                                        if fn_delta.get("arguments"):
                                            tool_calls_acc[tc_idx]["function"]["arguments"] += fn_delta.get("arguments")
                                reasoning_part = (
                                    delta.get("reasoning_content")
                                    or delta.get("reasoning")
                                    or delta.get("thought")
                                    or ""
                                )
                                content_part = delta.get("content") or ""
                                is_tool_delta = bool("tool_calls" in delta and delta["tool_calls"])
                                chunk_text = reasoning_part + content_part
                                if chunk_text or is_tool_delta:
                                    now = time.perf_counter()
                                    if t_first is None:
                                        t_first = now
                                    t_last = now
                                    if chunk_text:
                                        token_count += 1
                                        chunks.append(chunk_text)
                                        if reasoning_part:
                                            reasoning_chunks.append(reasoning_part)
                                        if content_part:
                                            content_chunks.append(content_part)
                                        if chunk_callback:
                                            try:
                                                chunk_callback(chunk_text, t_first, now, token_count)
                                            except Exception:
                                                pass
                                    elif is_tool_delta:
                                        token_count += 1
                    t_end = time.perf_counter()
                    total_time = t_end - t0
                    ttft = (t_first - t0) if t_first else total_time
                    gen_time = (t_end - t_first) if t_first else total_time
                    comp_tok = completion_tokens if completion_tokens > 0 else token_count
                    speed = ((comp_tok - 1) / gen_time) if gen_time > 0 and comp_tok > 1 else (comp_tok / gen_time if gen_time > 0 else 0.0)
                    content_str = "".join(content_chunks)
                    reasoning_str = "".join(reasoning_chunks)
                    full_text = "".join(chunks)
                    # Finalize accumulated streaming tool calls
                    if tool_calls_acc:
                        tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
                    return {
                        "content": content_str if (content_str or not reasoning_str) else "",
                        "reasoning": reasoning_str,
                        "text": full_text,
                        "tool_calls": tool_calls,
                        "total_time": total_time,
                        "ttft": ttft,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": comp_tok,
                        "cached_tokens": cached_tokens,
                        "decode_speed": speed
                    }
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                total_time = time.perf_counter() - t0
                if (e.code == 400 or e.code == 422) and "reasoning_effort" in payload:
                    err_lower = err_body.lower()
                    if any(term in err_lower for term in ["reasoning_effort", "extra fields", "unrecognized", "unexpected"]):
                        payload.pop("reasoning_effort", None)
                        continue
                # Rate limits (429) draw on a separate, more patient budget and
                # honor Retry-After; other transient server errors (5xx) use the
                # short jittered backoff. Client errors (4xx) are final.
                if e.code == 429:
                    if rate_limit_retries < retries + RATE_LIMIT_EXTRA_RETRIES:
                        rate_limit_retries += 1
                        time.sleep(_backoff_delay(rate_limit_retries - 1, 429, _parse_retry_after(e)))
                        continue
                elif 500 <= e.code < 600:
                    if retry_index < retries:
                        retry_index += 1
                        time.sleep(_backoff_delay(retry_index - 1, e.code))
                        continue
                return {
                    "error": f"HTTPError {e.code}: {e.reason} - {err_body}".strip(),
                    "status_code": e.code,
                    "failure_kind": "rate_limit" if e.code == 429 else ("infra" if 500 <= e.code < 600 else "client"),
                    "text": "",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": None,
                    "total_time": total_time,
                    "ttft": total_time,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "decode_speed": 0.0
                }
            except Exception as e:
                total_time = time.perf_counter() - t0
                # Transient connection/timeout errors use the short backoff.
                if retry_index < retries:
                    retry_index += 1
                    time.sleep(_backoff_delay(retry_index - 1))
                    continue
                return {
                    "error": str(e),
                    "failure_kind": "infra",
                    "text": "",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": None,
                    "total_time": total_time,
                    "ttft": total_time,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "decode_speed": 0.0
                }


def detect_max_context(model_obj, warn=False, default=32768):
    """Detects the context window from a /models entry; falls back to default (32768)."""
    if not isinstance(model_obj, dict):
        if warn:
            if default:
                log(f"Warning: Could not detect max context from model metadata; assuming {default:,} tokens.", color=YELLOW)
            else:
                log("Warning: Could not detect max context from model metadata.", color=YELLOW)
        return default
    candidates = [
        model_obj.get("max_model_len"),
        model_obj.get("context_length"),
        model_obj.get("max_context_length"),
        model_obj.get("context_window"),
        model_obj.get("top_provider", {}).get("context_length") if isinstance(model_obj.get("top_provider"), dict) else None,
        model_obj.get("max_position_embeddings"),
        model_obj.get("max_tokens"),
    ]
    for c in candidates:
        if isinstance(c, (int, float)) and c > 0:
            return int(c)
        if isinstance(c, str) and c.strip().isdigit() and int(c.strip()) > 0:
            return int(c.strip())
    if warn:
        if default:
            log(f"Warning: Server did not report a context limit; assuming {default:,} tokens (override with --max-context).", color=YELLOW)
        else:
            log("Warning: Server did not report a context limit (override with --max-context).", color=YELLOW)
    return default


def build_context_milestones(max_context: int, max_ratio: float = 0.80):
    standard_targets = [4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1000000]
    effective_cap = int(max_context * max_ratio)
    # Reserve a safety buffer for completion tokens (64) + prompt formatting/salt (~200)
    headroom = min(1000, max(256, int(effective_cap * 0.005)))
    safe_ceiling = max(1000, effective_cap - headroom)

    milestones = [t for t in standard_targets if t <= safe_ceiling]
    if not milestones:
        milestones = [min(4000, safe_ceiling)]

    # Include the upper ceiling up to max_ratio if not already present
    if safe_ceiling > milestones[-1]:
        rounded_ceiling = (safe_ceiling // 100) * 100
        milestones.append(rounded_ceiling)
    return sorted(list(set(milestones)))


def extract_python_code(raw_content: str, prefer_class: str = None) -> str:
    """Robustly extracts python code from assistant responses, handling markdown fences and edge cases."""
    fence_pattern = r"```(?:python|python3|py)?\s*\n(.*?)```"
    blocks = re.findall(fence_pattern, raw_content, re.DOTALL | re.IGNORECASE)

    # Check for unclosed block at end of response (e.g. truncated generation)
    unclosed_match = re.search(r"```(?:python|python3|py)?\s*\n((?:(?!```).)*)$", raw_content, re.DOTALL | re.IGNORECASE)
    unclosed_code = unclosed_match.group(1).strip() if unclosed_match else ""
    if unclosed_code:
        unclosed_code = re.sub(r"```\s*$", "", unclosed_code).strip()

    all_blocks = list(blocks)
    if unclosed_code and unclosed_code not in all_blocks:
        all_blocks.append(unclosed_code)

    if prefer_class:
        pref_blocks = [b.strip() for b in all_blocks if prefer_class in b]
        if pref_blocks:
            return max(pref_blocks, key=len)
        if prefer_class in unclosed_code:
            return unclosed_code
        if prefer_class not in raw_content:
            return ""

    if blocks:
        return max(blocks, key=len).strip()
    if unclosed_code:
        return unclosed_code

    if any(k in raw_content for k in ("def ", "class ", "import ", "return ")):
        code = raw_content.strip()
        code = re.sub(r"```\s*$", "", code).strip()
        return code

    return ""


def assemble_python_script(code: str, preamble: str = "", harness: str = "") -> str:
    """
    Assembles model-generated code with an optional import preamble and verification harness.
    Guarantees that any 'from __future__ import ...' statements appear at the very top of the
    final script to satisfy Python syntax requirements.
    """
    future_imports = []

    def strip_future(text: str) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from __future__ import"):
                if stripped not in future_imports:
                    future_imports.append(stripped)
            else:
                lines.append(line)
        return "\n".join(lines)

    clean_preamble = strip_future(preamble).strip()
    clean_code = strip_future(code).strip()
    clean_harness = strip_future(harness).strip()

    sections = []
    if future_imports:
        sections.append("\n".join(future_imports))
    if clean_preamble:
        sections.append(clean_preamble)
    if clean_code:
        sections.append(clean_code)
    if clean_harness:
        sections.append(clean_harness)

    return "\n\n".join(sections)

