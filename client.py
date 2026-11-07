#!/usr/bin/env python3
"""
Shared LLM HTTP client & terminal formatting utilities for llm-eval-suite.

Zero external dependencies (Python stdlib only). Used by eval.py, compare.py,
judge_eval.py, and reporting.py to avoid duplicated protocol code.
"""

import errno
import os
import socket
import sys
import time
import json
import re
import random
import secrets
import signal
import subprocess
import tempfile
import threading
from contextlib import contextmanager, nullcontext
import functools
import atexit
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
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


_LOG_LOCK = threading.RLock()
_LOG_LOCAL = threading.local()


def _emit(line: str, msg) -> None:
    with _LOG_LOCK:
        print(line, flush=True)
        if _LOG_FILE_HANDLE:
            try:
                clean_msg = re.sub(r"\033\[[0-9;]*m", "", str(msg))
                _LOG_FILE_HANDLE.write(clean_msg + "\n")
                _LOG_FILE_HANDLE.flush()
            except Exception:
                pass


def log(msg, bold=False, color=""):
    prefix = bold and BOLD or ""
    c = color or ""
    suffix = (bold or color) and RESET or ""
    line = f"{prefix}{c}{msg}{suffix}"
    buffer = getattr(_LOG_LOCAL, "buffer", None)
    if buffer is not None:
        buffer.append((line, msg))
        return
    _emit(line, msg)


@contextmanager
def buffered_log():
    """Collects this thread's log lines and prints them as one block at the
    end, so tests running in parallel do not interleave their output."""
    _LOG_LOCAL.buffer = []
    try:
        yield
    finally:
        lines, _LOG_LOCAL.buffer = _LOG_LOCAL.buffer, None
        with _LOG_LOCK:
            for line, msg in lines:
                _emit(line, msg)


def carry_log_buffer(fn):
    """`fn`, made to log into the calling thread's buffer (see buffered_log)
    when a worker thread runs it, so the thread pool of a buffered test
    prints inside the test's block instead of between other tests' blocks.
    Unchanged when the calling thread is not buffered. The pool must finish
    before the buffered block ends."""
    buffer = getattr(_LOG_LOCAL, "buffer", None)
    if buffer is None:
        return fn

    @functools.wraps(fn)
    def run(*args, **kwargs):
        _LOG_LOCAL.buffer = buffer
        try:
            return fn(*args, **kwargs)
        finally:
            _LOG_LOCAL.buffer = None
    return run


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


_UNREACHABLE_ERRNOS = (errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH)
# A probe URL that answers with one of these does not exist on the server.
_PROBE_ABSENT_STATUS = (404, 405, 501)
# Queue-counter sources, in the order fetch_queue_state tries them.
_QUEUE_SOURCES = ("/metrics", "/health jobs", "/slots")


def is_unreachable_error(exc) -> bool:
    """True when nothing is listening at the endpoint (connection refused,
    no route to the host or network, name not resolved). A timeout or reset
    is not one: a busy server can recover from those."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, socket.gaierror):
        # EAI_AGAIN is a temporary resolver failure: the name may resolve on
        # the next try, so it does not mean nothing is listening.
        return reason.errno != getattr(socket, "EAI_AGAIN", -3)
    if isinstance(reason, ConnectionRefusedError):
        return True
    return isinstance(reason, OSError) and not isinstance(reason, TimeoutError) \
        and reason.errno in _UNREACHABLE_ERRNOS


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


def _positive_int(value):
    """A positive int from a server field (int or digit string), else None.
    Booleans are rejected: `True` must not read as 1 slot."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value) or None
    return None


# Prometheus gauges for the server's request queue, by serving engine.
QUEUE_METRICS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_preemptions_total": "preemptions",
    "sglang:num_running_reqs": "running",
    "sglang:num_queue_reqs": "waiting",
    "tgi_batch_current_size": "running",
    "tgi_queue_size": "waiting",
    "llamacpp:requests_processing": "running",
    "llamacpp:requests_deferred": "waiting",
}
_PROM_LINE = re.compile(r"^([A-Za-z_:][\w:]*)(\{.*\})?\s+(\S+)")


def parse_queue_metrics(raw: str):
    """Running / waiting / preemption totals from Prometheus text (summed
    over engines), or None when no queue gauge is present."""
    totals: dict = {}
    for line in (raw or "").splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line.strip())
        if not m or m.group(1) not in QUEUE_METRICS:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        field = QUEUE_METRICS[m.group(1)]
        totals[field] = totals.get(field, 0) + value
    if "running" not in totals:
        return None
    return {"running": int(totals["running"]),
            "waiting": int(totals["waiting"]) if "waiting" in totals else None,
            "preemptions": int(totals["preemptions"]) if "preemptions" in totals else None}


def parse_kv_capacity(raw: str):
    """vLLM's KV-cache size from the `cache_config_info` labels, or None."""
    for line in (raw or "").splitlines():
        if line.startswith("vllm:cache_config_info{"):
            labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
            tokens = _positive_int(labels.get("kv_cache_size_tokens"))
            if not tokens:
                return None
            try:
                concurrency = round(float(labels.get("kv_cache_max_concurrency", "")), 2)
            except ValueError:
                concurrency = None
            return {"kv_cache_tokens": tokens, "max_concurrency_at_max_len": concurrency}
    return None


# Cache settings worth reporting, from vLLM's `cache_config_info` labels.
_CACHE_CONFIG_LABELS = {
    "kv_cache_size_tokens": "kv_cache_tokens",
    "cache_dtype": "kv_cache_dtype",
    "enable_prefix_caching": "prefix_caching",
    "gpu_memory_utilization": "gpu_memory_utilization",
    "kv_cache_memory_bytes": "kv_cache_memory_bytes",
}


def parse_cache_config(raw: str) -> Dict[str, Any]:
    """Selected labels of vLLM's `cache_config_info` metric ({} when absent)."""
    for line in (raw or "").splitlines():
        if line.startswith("vllm:cache_config_info{"):
            labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
            out: Dict[str, Any] = {}
            for label, name in _CACHE_CONFIG_LABELS.items():
                raw_value = labels.get(label)
                if not raw_value or raw_value == "None":
                    continue
                try:
                    if name in ("kv_cache_tokens", "kv_cache_memory_bytes"):
                        value = _positive_int(raw_value)  # same check as parse_kv_capacity
                        if value is None:
                            continue
                        out[name] = value
                    elif name == "gpu_memory_utilization":
                        out[name] = float(raw_value)
                    elif name == "prefix_caching":
                        out[name] = raw_value.lower() == "true"
                    else:
                        out[name] = raw_value
                except ValueError:
                    continue
            return out
    return {}


def weights_repo_from_path(path: str) -> Optional[str]:
    """'org/name' from a Hugging Face cache path ('.../models--org--name/...'),
    else None."""
    m = re.search(r"models--([^/]+?)--([^/]+)", path or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


# Request fields that `sampling` may set (vLLM, SGLang and llama.cpp accept
# the non-OpenAI ones).
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                 "repetition_penalty")


class LLMClient:
    """OpenAI-compatible chat completions client (non-streaming + SSE streaming)."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None, model: Optional[str] = None):
        self.base_url = normalize_endpoint(endpoint)
        self.completions_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        self.root_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.health_url = f"{self.root_url}/health"
        self.metrics_url = f"{self.root_url}/metrics"
        # Probe URLs known not to exist on this server (404/405/501, or
        # nothing listening). They are never requested again.
        self._unavailable_urls: "set[str]" = set()
        # The queue-counter source that answered first; "none" when the
        # server has none. Later polls ask only that source.
        self._queue_source = None
        # Set when a chat request finds nothing listening at the endpoint
        # (after its retries). Every later call fails at once without a
        # request, and the scheduler ends the run as incomplete.
        self.server_down: Optional[str] = None
        self.api_key = api_key
        self.model = model
        # Caps requests in flight across all threads (set by the scheduler
        # for the parallel phase; None = unlimited).
        self.request_limiter = None

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

    def _get_text(self, url: str, timeout: float = 5):
        """GET a probe URL (/health, /metrics, /slots, /props); the body, or
        None on any failure. A URL that answers 404/405/501, or where nothing
        listens, is remembered and not requested again. Timeouts and 5xx are
        not remembered: a busy server can answer the next time."""
        if url in self._unavailable_urls:
            return None
        req = urllib.request.Request(url, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            e.close()
            if e.code in _PROBE_ABSENT_STATUS:
                self._unavailable_urls.add(url)
            return None
        except Exception as e:
            if is_unreachable_error(e):
                self._unavailable_urls.add(url)
            return None

    def _get_json(self, url: str, timeout: float = 5):
        """GET a JSON document; None when unavailable, non-2xx, or not JSON."""
        raw = self._get_text(url, timeout)
        try:
            return json.loads(raw) if raw is not None else None
        except ValueError:
            return None

    def fetch_health(self):
        return self._get_json(self.health_url)

    def fetch_queue_state(self):
        """The server's own view of its request queue: {"running", "waiting",
        "preemptions", "source"} (waiting/preemptions may be None), or None
        when the server exposes no queue counters. Sources, in order:
        Prometheus /metrics (vLLM, SGLang, TGI, llama.cpp), the exl3-style
        /health job counts, then llama.cpp /slots.

        The first source that gives counters is remembered, and later calls
        (QueueSampler polls every 0.5 s) ask only that one. When every source
        answered without counters or is unavailable, the server has none, and
        no more requests are sent. A timeout leaves the question open."""
        if self._queue_source == "none":
            return None
        sources = [self._queue_source] if self._queue_source else list(_QUEUE_SOURCES)
        settled = True  # every source answered (or is known to be absent)
        for source in sources:
            answered, state = self._read_queue_source(source)
            if state:
                self._queue_source = source
                return state
            settled = settled and answered
        if self._queue_source is None and settled:
            self._queue_source = "none"
        return None

    def _read_queue_source(self, source: str):
        """(answered, state) for one queue-counter source. answered is False
        when the request failed in a way that may not repeat (timeout, 5xx)."""
        if source == "/metrics":
            url = self.metrics_url
            raw = self._get_text(url)
            state = parse_queue_metrics(raw) if raw else None
            state = dict(state, source=source) if state else None
            body = raw
        elif source == "/health jobs":
            url = self.health_url
            body = health = self._get_json(url)
            state = None
            if isinstance(health, dict) and "gpu_active_jobs" in health:
                running = _positive_int(health.get("gpu_active_jobs")) or 0
                waiting = _positive_int(health.get("gpu_pending_jobs")) or 0
                state = {"running": running, "waiting": waiting, "preemptions": None, "source": source}
        else:
            url = f"{self.root_url}/slots"
            body = slots = self._get_json(url)
            state = None
            if isinstance(slots, list) and slots and all(isinstance(s, dict) for s in slots):
                state = {"running": sum(1 for s in slots if s.get("is_processing")), "waiting": None,
                         "preemptions": None, "source": source}
        return (body is not None or url in self._unavailable_urls), state

    def fetch_server_info(self) -> Dict[str, Any]:
        """What the server says about itself: inference engine and version,
        model weights, context length and KV-cache settings. GPU type is not
        exposed by these APIs; it comes from the `hardware` config field.
        Keys are present only when the server reports them."""
        info: Dict[str, Any] = {}
        version = self._get_json(f"{self.root_url}/version")  # vLLM
        if isinstance(version, dict) and version.get("version"):
            info.update(engine="vLLM", version=str(version["version"]))
        else:
            for url, engine, field in ((f"{self.root_url}/get_server_info", "SGLang", "version"),
                                       (f"{self.root_url}/info", "TGI", "version"),
                                       (f"{self.root_url}/props", "llama.cpp", "build_info")):
                body = self._get_json(url)
                if isinstance(body, dict) and body.get(field):
                    info.update(engine=engine, version=str(body[field]))
                    break
        for m in self.fetch_models():
            if isinstance(m, dict) and (m.get("id") == self.model or not self.model):
                if m.get("owned_by") and "engine" not in info:
                    info["engine"] = str(m["owned_by"])
                repo = weights_repo_from_path(str(m.get("root") or ""))
                if repo:
                    info["weights"] = repo
                if m.get("max_model_len"):
                    info["max_model_len"] = m["max_model_len"]
                break
        info.update(parse_cache_config(self._get_text(self.metrics_url) or ""))
        return info

    def count_prompt_tokens(self, messages) -> Optional[int]:
        """Exact prompt tokens for `messages`, chat template included, from
        the server's tokenizer (vLLM and SGLang POST /tokenize; it does not
        run the model). None when the server has no such endpoint."""
        url = f"{self.root_url}/tokenize"
        if url in self._unavailable_urls:
            return None
        body = json.dumps({"model": self.model, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                count = json.loads(resp.read().decode("utf-8")).get("count")
        except urllib.error.HTTPError as e:
            e.close()
            if e.code in _PROBE_ABSENT_STATUS:
                self._unavailable_urls.add(url)
            # 400/422 concern this one request (for example a prompt over the
            # context): count it as unknown, but keep using /tokenize.
            return None
        except Exception as e:
            if is_unreachable_error(e):
                self._unavailable_urls.add(url)
            return None
        return count if isinstance(count, int) and count > 0 else None

    def fetch_kv_capacity(self):
        """KV-cache size the server reports (vLLM `cache_config_info`):
        {"kv_cache_tokens", "max_concurrency_at_max_len"}, or None."""
        raw = self._get_text(self.metrics_url)
        return parse_kv_capacity(raw) if raw else None

    def detect_server_slots(self):
        """(slots, source): how many requests the server says it serves in
        parallel. Tries /health (`parallel` / `max_slots`), then llama.cpp's
        /props (`total_slots`) and /slots (one entry per slot). Returns
        (None, "not reported") when no endpoint says -- e.g. vLLM, which
        batches up to an internal limit it does not expose."""
        health = self.fetch_health()
        if isinstance(health, dict):
            for key in ("parallel", "max_slots"):
                slots = _positive_int(health.get(key))
                if slots:
                    return slots, f"/health {key}"
        props = self._get_json(f"{self.root_url}/props")
        if isinstance(props, dict):
            slots = _positive_int(props.get("total_slots"))
            if slots:
                return slots, "/props total_slots"
        slot_list = self._get_json(f"{self.root_url}/slots")
        if isinstance(slot_list, list) and slot_list:
            return len(slot_list), "/slots"
        return None, "not reported"

    def fetch_server_metrics(self):
        """
        Scrapes and parses Prometheus metrics from /metrics (e.g. vLLM, TGI, SGLang).
        Returns a dictionary of key metrics or None if not supported/available.
        """
        raw = self._get_text(self.metrics_url)
        if raw is None:
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

    def call(self, *args, **kwargs):
        """Sends a chat completion request, waiting for a free request slot
        when the scheduler has set a limit (see _call for the parameters).
        Once the endpoint is known to be down, no request is sent: the call
        returns the same infra failure at once."""
        if self.server_down:
            return _server_down_result(self.server_down)
        with (self.request_limiter or nullcontext()):
            if self.server_down:  # went down while this call waited for a slot
                return _server_down_result(self.server_down)
            res = self._call(*args, **kwargs)
        if isinstance(res, dict) and res.get("endpoint_unreachable") and not self.server_down:
            self.server_down = str(res.get("error"))[:300]
            log(f"\n[SERVER DOWN] {self.base_url} is not reachable: {self.server_down}. "
                "No more requests will be sent.", bold=True, color=RED)
        return res

    def _call(self, messages, max_tokens=1024, temperature=0.0, stream=False,
              tools=None, tool_choice=None, response_format=None, stop=None,
              seed=None, timeout=None, chat_template_kwargs=None, reasoning_effort=None,
              retries=2, chunk_callback=None, model=None, sampling=None):
        """
        Sends a chat completion request. Retries transient failures (5xx,
        timeouts, connection errors) up to `retries` additional attempts.
        Client errors (4xx) are returned immediately without retry.

        `timeout` (seconds) bounds each socket read. Streaming, that is the
        longest gap between chunks. Without streaming the whole answer is one
        read, so the default grows with `max_tokens` (see request_timeout).
        """
        if timeout is None:
            timeout = request_timeout(max_tokens, stream)
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        if sampling:  # explicit settings (for example a model card's) override `temperature`
            payload.update({k: v for k, v in sampling.items() if k in SAMPLING_KEYS and v is not None})
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
                        return _failure_result(f"API Error: {err_msg}",
                                               "client" if any(w in err_msg.lower() for w in ["invalid", "not found", "bad request"]) else "infra",
                                               total_time, raw=res_json)
                    choices = res_json.get("choices") or [{}]
                    choice = choices[0] if choices else {}
                    if choice.get("finish_reason") == "error":
                        err_msg = choice.get("message", {}).get("content", "Server error during generation")
                        return _failure_result(f"Server finish_reason error: {err_msg}", "infra", total_time, raw=res_json)
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
                        "finish_reason": choice.get("finish_reason"),
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
                    t_first_content = None
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
                    stream_finish_reason = None
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
                                return _failure_result(
                                    f"Stream error: {err_msg}", "infra", time.perf_counter() - t0,
                                    ttft=(t_first - t0) if t_first else (time.perf_counter() - t0),
                                    text="".join(chunks), content="".join(content_chunks),
                                    reasoning="".join(reasoning_chunks), prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens or token_count, cached_tokens=cached_tokens)
                            if "usage" in c and c["usage"]:
                                prompt_tokens = c["usage"].get("prompt_tokens", prompt_tokens)
                                completion_tokens = c["usage"].get("completion_tokens", completion_tokens)
                                cached_tokens = (c["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                            choices = c.get("choices", [])
                            if choices:
                                if choices[0].get("finish_reason") == "error":
                                    err_msg = choices[0].get("message", {}).get("content", "Server error during streaming")
                                    return _failure_result(
                                        f"Stream finish_reason error: {err_msg}", "infra", time.perf_counter() - t0,
                                        ttft=(t_first - t0) if t_first else (time.perf_counter() - t0),
                                        text="".join(chunks), content="".join(content_chunks),
                                        reasoning="".join(reasoning_chunks), prompt_tokens=prompt_tokens,
                                        completion_tokens=completion_tokens or token_count, cached_tokens=cached_tokens)
                                if choices[0].get("finish_reason"):
                                    stream_finish_reason = choices[0]["finish_reason"]
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
                                            if t_first_content is None:
                                                t_first_content = now
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
                        "finish_reason": stream_finish_reason,
                        "tool_calls": tool_calls,
                        "total_time": total_time,
                        "ttft": ttft,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": comp_tok,
                        "cached_tokens": cached_tokens,
                        "decode_speed": speed,
                        # Thinking models: time to the first answer (non-reasoning)
                        # token, and how many streamed chunks were reasoning.
                        "answer_ttft": (t_first_content - t0) if t_first_content else None,
                        "reasoning_chunks": len(reasoning_chunks),
                        "content_chunks": len(content_chunks),
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
                return _failure_result(f"HTTPError {e.code}: {e.reason} - {err_body}".strip(),
                                       "rate_limit" if e.code == 429 else ("infra" if 500 <= e.code < 600 else "client"),
                                       total_time, status_code=e.code)
            except Exception as e:
                total_time = time.perf_counter() - t0
                # Transient connection/timeout errors use the short backoff.
                if retry_index < retries:
                    retry_index += 1
                    time.sleep(_backoff_delay(retry_index - 1))
                    continue
                return _failure_result(str(e), "infra", total_time, endpoint_unreachable=is_unreachable_error(e))


def _failure_result(error: str, kind: str, total_time: float = 0.0, ttft: Optional[float] = None,
                    **extra: Any) -> Dict[str, Any]:
    """The result of a failed request, in the same shape as a successful one
    (empty text, zero tokens), plus `extra` fields (partial stream text,
    status_code, raw, endpoint_unreachable)."""
    res: Dict[str, Any] = {"error": error, "failure_kind": kind, "text": "", "content": "", "reasoning": "",
                           "tool_calls": None, "total_time": total_time,
                           "ttft": total_time if ttft is None else ttft, "prompt_tokens": 0,
                           "completion_tokens": 0, "cached_tokens": 0, "decode_speed": 0.0}
    res.update(extra)
    return res


def _server_down_result(reason: str) -> dict:
    """The result of a call that was not sent because the endpoint is down."""
    return _failure_result(f"not sent: server unreachable ({reason})", "infra", endpoint_unreachable=True)


# failure_kind values (set by LLMClient.call) that mean the serving
# environment failed, not the model.
INFRA_FAILURE_KINDS = ("infra", "rate_limit")


class InfraTrackingClient:
    """Wraps a client and records every call that failed for infrastructure
    reasons (network / timeout / 5xx / 429). All other attributes delegate to
    the wrapped client, so test code uses it exactly like an LLMClient."""

    def __init__(self, inner):
        self._inner = inner
        self.infra_errors = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def call(self, *args, **kwargs):
        res = self._inner.call(*args, **kwargs)
        if isinstance(res, dict) and res.get("failure_kind") in INFRA_FAILURE_KINDS:
            self.infra_errors.append(str(res.get("error") or res["failure_kind"]))
        return res


# Default socket timeout, and the slowest decode a non-streaming request is
# given time for: a 32,768-token answer at 10 tok/s needs ~55 minutes.
DEFAULT_TIMEOUT_S = 600
MIN_DECODE_TOK_S = 10


def request_timeout(max_tokens: int, stream: bool) -> float:
    """Socket timeout for a request: DEFAULT_TIMEOUT_S between stream chunks,
    or, without streaming, enough to decode `max_tokens` at MIN_DECODE_TOK_S
    (never less than DEFAULT_TIMEOUT_S)."""
    if stream:
        return DEFAULT_TIMEOUT_S
    return max(DEFAULT_TIMEOUT_S, (max_tokens or 0) / MIN_DECODE_TOK_S + 120)


# Environment variables never passed to model-generated code: API keys for the
# tested server and the judge (env_loader copies .env into os.environ) and any
# other credential the operator's shell holds.
_SECRET_ENV_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH", re.IGNORECASE)


def model_code_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment for running model-generated code: os.environ without
    variables whose names look like credentials, plus `extra`."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV_NAME.search(k)}
    env.update(extra or {})
    return env


# Every run of model-written code gets a random id in this variable. A
# verification harness prints its success marker followed by the id
# (harness_passed), so an answer that prints the marker itself does not pass.
RUN_ID_VAR = "EVAL_RUN_ID"


def harness_passed(proc, marker: str) -> bool:
    """The run exited 0 and printed `marker` followed by its run id."""
    return proc.returncode == 0 and f"{marker} {proc.run_id}" in proc.stdout.splitlines()


# Output kept from each stream (stdout, stderr) of model-written code. The
# rest is read and dropped, so code that prints without end cannot fill the
# memory of the grader.
MODEL_OUTPUT_LIMIT = 1_000_000


def _drain(stream, chunks: list) -> None:
    """Reads `stream` to its end; keeps the first MODEL_OUTPUT_LIMIT bytes in
    `chunks`, then a None that marks the cut."""
    kept, cut = 0, False
    for chunk in iter(lambda: stream.read(65536), b""):
        room = MODEL_OUTPUT_LIMIT - kept
        if room > 0:
            chunks.append(chunk[:room])
            kept += len(chunks[-1])
        if len(chunk) > room and not cut:
            chunks.append(None)
            cut = True
    stream.close()


def _decode(chunks: list) -> str:
    text = b"".join(c for c in chunks if c is not None).decode("utf-8", errors="replace")
    if None in chunks:
        text += f"\n[output cut at {MODEL_OUTPUT_LIMIT:,} bytes]"
    return text


def _kill_group(proc) -> None:
    """Kills the process and everything it started (its process group)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass  # the group has already exited
    proc.wait()


def run_model_code(cmd, timeout: float, cwd: Optional[str] = None,
                   env: Optional[Dict[str, str]] = None) -> "subprocess.CompletedProcess":
    """Runs a command that executes model-written code (or builds it), like
    subprocess.run(capture_output=True, text=True, timeout=...), with three
    guards: the code runs in its own process group, and that group is killed
    when the command ends or times out, so background processes it started
    do not outlive the grade; at most MODEL_OUTPUT_LIMIT bytes of each stream
    are kept; and the default working folder and environment carry no files
    or credentials of the suite (cwd: the temp folder; env: model_code_env()).
    The environment also holds a random run id (RUN_ID_VAR, kept when `env`
    already sets it), returned as `.run_id` for harness_passed.
    Raises subprocess.TimeoutExpired on timeout, after the kill."""
    env = dict(env if env is not None else model_code_env())
    env.setdefault(RUN_ID_VAR, secrets.token_hex(16))
    proc = subprocess.Popen(cmd, cwd=cwd if cwd is not None else tempfile.gettempdir(), env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    out: list = []
    err: list = []
    readers = [threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)]
    for t in readers:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        _kill_group(proc)
    for t in readers:
        t.join(timeout=10)
    stdout, stderr = _decode(out), _decode(err)
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
    done = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    done.run_id = env[RUN_ID_VAR]  # type: ignore[attr-defined]
    return done


def was_truncated(res) -> bool:
    """True when generation stopped at the max_tokens limit, so the answer
    may be incomplete (finish_reason 'length')."""
    return isinstance(res, dict) and res.get("finish_reason") == "length"


def mark_infra_failure(result, infra_errors):
    """Marks a test result as an infrastructure failure when any of its
    requests failed for infra reasons. A score computed with missing answers
    is not a model-quality measurement, so the result becomes FAIL with
    failure_kind='infra' (excluded from effectiveness, counted against the
    completion rate). The score the model reached is kept as 'model_status'."""
    if not infra_errors or not isinstance(result, dict) or result.get("status") == "SKIPPED":
        return result
    marked = dict(result)
    marked["model_status"] = result.get("status")
    marked["status"] = "FAIL"
    marked["failure_kind"] = "infra"
    marked["error"] = f"{len(infra_errors)} request(s) failed (infra): {infra_errors[0]}"
    return marked


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

