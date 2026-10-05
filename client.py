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
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

# ANSI terminal colors (respect NO_COLOR env and non-TTY pipes)
NO_COLOR = bool(os.getenv("NO_COLOR")) or not sys.stdout.isatty()
BOLD = "" if NO_COLOR else "\033[1m"
GREEN = "" if NO_COLOR else "\033[32m"
YELLOW = "" if NO_COLOR else "\033[33m"
RED = "" if NO_COLOR else "\033[31m"
CYAN = "" if NO_COLOR else "\033[36m"
MAGENTA = "" if NO_COLOR else "\033[35m"
RESET = "" if NO_COLOR else "\033[0m"


def log(msg, bold=False, color=""):
    prefix = bold and BOLD or ""
    c = color or ""
    suffix = (bold or color) and RESET or ""
    print(f"{prefix}{c}{msg}{suffix}", flush=True)


class LLMClient:
    """OpenAI-compatible chat completions client (non-streaming + SSE streaming)."""

    def __init__(self, endpoint: str, api_key: str = None, model: str = None):
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            endpoint = endpoint[:-len("/chat/completions")]
        elif not endpoint.endswith("/v1"):
            # Only append /v1 to bare host endpoints (no path component).
            # Custom gateway paths (e.g. /api, /proxy) are left untouched.
            parsed = urlparse(endpoint)
            if not parsed.path:
                endpoint = f"{endpoint}/v1"

        self.base_url = endpoint
        self.completions_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        self.health_url = self.base_url.replace("/v1", "/health")
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

    def fetch_health(self):
        req = urllib.request.Request(self.health_url, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def call(self, messages, max_tokens=1024, temperature=0.0, stream=False,
             tools=None, tool_choice=None, response_format=None, stop=None,
             seed=None, timeout=600, chat_template_kwargs=None, reasoning_effort=None,
             retries=2):
        """
        Sends a chat completion request. Retries transient failures (5xx,
        timeouts, connection errors) up to `retries` additional attempts.
        Client errors (4xx) are returned immediately without retry.
        """
        payload = {
            "model": self.model,
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

        data = json.dumps(payload).encode("utf-8")
        attempt = 0
        while True:
            attempt += 1
            req = urllib.request.Request(self.completions_url, data=data, headers=self._get_headers())
            t0 = time.perf_counter()
            try:
                if not stream:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                    total_time = time.perf_counter() - t0
                    choices = res_json.get("choices") or [{}]
                    choice = choices[0] if choices else {}
                    msg = choice.get("message", {})
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or ""
                    full_text = f"{reasoning}\n{content}".strip() if reasoning else (content or "")
                    tool_calls = msg.get("tool_calls")
                    usage = res_json.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    speed = completion_tokens / total_time if total_time > 0 else 0.0
                    return {
                        "content": content if content else full_text,
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
                            if "usage" in c and c["usage"]:
                                prompt_tokens = c["usage"].get("prompt_tokens", prompt_tokens)
                                completion_tokens = c["usage"].get("completion_tokens", completion_tokens)
                                cached_tokens = (c["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                            choices = c.get("choices", [])
                            if choices:
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
                                reasoning_part = delta.get("reasoning_content") or ""
                                content_part = delta.get("content") or ""
                                chunk_text = reasoning_part + content_part
                                if chunk_text:
                                    now = time.perf_counter()
                                    if t_first is None:
                                        t_first = now
                                    t_last = now
                                    token_count += 1
                                    chunks.append(chunk_text)
                                    if reasoning_part:
                                        reasoning_chunks.append(reasoning_part)
                                    if content_part:
                                        content_chunks.append(content_part)
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
                        "content": content_str if content_str else full_text,
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
                # Retry transient server errors (5xx); client errors (4xx) are final.
                if 500 <= e.code < 600 and attempt <= retries:
                    time.sleep(2)
                    continue
                return {
                    "error": f"HTTPError {e.code}: {e.reason} - {err_body}".strip(),
                    "status_code": e.code,
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
                # Retry transient connection/timeout errors.
                if attempt <= retries:
                    time.sleep(2)
                    continue
                return {
                    "error": str(e),
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


def detect_max_context(model_obj, warn=False):
    """Detects the context window from a /models entry; falls back to 32768."""
    if not isinstance(model_obj, dict):
        if warn:
            log("Warning: Could not detect max context from model metadata; assuming 32,768 tokens.", color=YELLOW)
        return 32768
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
        if isinstance(c, int) and c > 0:
            return c
    if warn:
        log("Warning: Server did not report a context limit; assuming 32,768 tokens (override with --max-context).", color=YELLOW)
    return 32768


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
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", raw_content, re.DOTALL | re.IGNORECASE)
    code = ""
    if blocks:
        if prefer_class:
            filtered = [b for b in blocks if prefer_class in b]
            code = max(filtered, key=len).strip() if filtered else max(blocks, key=len).strip()
        else:
            code = max(blocks, key=len).strip()
    else:
        m = re.search(r"```(?:python)?\s*\n(.*)", raw_content, re.DOTALL | re.IGNORECASE)
        code = m.group(1).strip() if m else raw_content.strip()
        code = re.sub(r"```\s*$", "", code).strip()

    return code
