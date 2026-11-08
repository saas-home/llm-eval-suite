"""Shared standalone benchmark clients, metrics and answer parsing."""
import json
import re
from decimal import Decimal

from client import LLMClient, log


class ChunkMetrics:
    def __init__(self, display=False):
        self.first = None
        self.last = None
        self.display = display

    def __call__(self, text, first, now, _count):
        if self.first is None:
            self.first = first
        self.last = now
        if self.display:
            print(text, end="", flush=True)

    @property
    def duration(self):
        return self.last - self.first if self.first is not None and self.last is not None else 0.0


def aggregate_decode_rate(results, start_key="_decode_start", end_key="_decode_end"):
    timed = [r for r in results if not r.get("error")
             and r.get(start_key) is not None and r.get(end_key) is not None
             and r[end_key] > r[start_key] and r["completion_tokens"] > 1]
    if not timed:
        return 0.0
    span = max(r[end_key] for r in timed) - min(r[start_key] for r in timed)
    return sum(r["completion_tokens"] - 1 for r in timed) / span


def benchmark_response(result):
    """Expose the shared client's response under the probes' existing keys."""
    tokens = result.get("completion_tokens", 0)
    speed = result.get("decode_speed") or 0
    response = {
        "text": result.get("text", ""),
        "content": result.get("content", ""),
        "reasoning": result.get("reasoning", ""),
        "tool_calls": result.get("tool_calls"),
        "ttft_ms": (result.get("ttft") or 0) * 1000,
        "total_time": result.get("total_time", 0),
        "total_time_s": result.get("total_time", 0),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": tokens,
        "tokens": tokens,
        "tok_per_sec": speed,
        "tok_s": speed,
    }
    for key in ("error", "failure_kind", "status_code", "endpoint_unreachable"):
        if key in result:
            response[key] = result[key]
    return response


def benchmark_error(name, result):
    """Keep request errors distinct from answers that fail a benchmark."""
    if not result.get("error"):
        return None
    log(f"  [ERROR] {name}: {result['error']}")
    return {
        "name": name,
        "passed": False,
        "status": "ERROR",
        "error": result["error"],
        "failure_kind": result.get("failure_kind"),
        "endpoint_unreachable": result.get("endpoint_unreachable", False),
        "tok_s": result.get("tok_s", 0),
        "tok_per_sec": result.get("tok_per_sec", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "tokens": result.get("tokens", 0),
    }


def call_nonstream(url, messages, max_tokens, temperature, enable_thinking, timeout, model, api_key):
    client = LLMClient(url, api_key=api_key, model=model)
    kwargs = {"enable_thinking": True} if enable_thinking else None
    result = client.call(messages, max_tokens=max_tokens, temperature=temperature, stream=False,
                         timeout=timeout, chat_template_kwargs=kwargs)
    if (kwargs and result.get("status_code") in (400, 422)
            and "chat_template_kwargs" in result.get("error", "").lower()):
        log("  [WARNING] Server rejected chat_template_kwargs; retrying without it.")
        result = client.call(messages, max_tokens=max_tokens, temperature=temperature,
                             stream=False, timeout=timeout)
    return benchmark_response(result)


def declared_value(text, name):
    pattern = (rf"(?i)\b{re.escape(name)}\b[`*_]*[ \t]*(?::|=|\bis\b)[ \t]*"
               r"[`*_]*([^\s;]+)")
    matches = re.findall(pattern, text)
    return matches[-1].strip("`*_.:),") if matches else None


def declared_integer(text, name):
    value = declared_value(text, name)
    return int(value) if value and re.fullmatch(r"[+-]?\d+", value) else None


def declared_money(text, name):
    value = declared_value(text, name)
    if value and re.fullmatch(r"\$[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}", value):
        return Decimal(value[1:].replace(",", ""))
    return None


def strict_json_loads(text):
    def reject_constant(value):
        raise ValueError(f"{value} is not valid JSON")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
