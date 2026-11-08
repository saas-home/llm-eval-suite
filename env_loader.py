"""
Standard Library Environment & Configuration Loader for llm-eval-suite.

Zero external dependencies. Manages config.json and .env for server settings.
Supports single-server benchmark and two-server Live Arena comparison mode.
"""

import os
import json
import sys

from client import OUTPUT_BASE_DIR

BASE_DIR = OUTPUT_BASE_DIR
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def load_env_file(env_file=None, override=False):
    """
    Parses a .env file into os.environ without external dependencies.
    """
    candidates = []
    if env_file:
        candidates.append(os.path.abspath(env_file))
    else:
        cwd = os.getcwd()
        candidates.extend([
            os.path.join(cwd, ".env"),
            os.path.join(cwd, ".env.local"),
            os.path.join(BASE_DIR, ".env"),
            os.path.join(BASE_DIR, ".env.local"),
        ])

    loaded_path = None
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[len("export "):].strip()
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip()
                            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                                val = val[1:-1]
                            if key and (override or key not in os.environ):
                                os.environ[key] = val
                loaded_path = path
                break
            except Exception:
                pass

    return loaded_path


def load_config(config_file=None):
    """
    Loads benchmark server configuration: single server or two servers.
    Returns (config_dict, config_path, loaded_from_file).
    """
    target_path = os.path.abspath(config_file) if config_file else DEFAULT_CONFIG_PATH
    config = {
        "endpoint": None,
        "model": None,
        "api_key": None,
        "max_context": None,
        "endpoint1": None,
        "model1": None,
        "api_key1": None,
        "max_context1": None,
        "endpoint2": None,
        "model2": None,
        "api_key2": None,
        "max_context2": None,
    }
    loaded_from_file = False

    # 1. Load from .env if present
    load_env_file()
    if os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT"):
        config["endpoint"] = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_ENDPOINT")
    if os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL"):
        config["model"] = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
    if os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"):
        config["api_key"] = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if os.getenv("MAX_CONTEXT") or os.getenv("EVAL_MAX_CONTEXT"):
        try:
            config["max_context"] = int(os.getenv("MAX_CONTEXT") or os.getenv("EVAL_MAX_CONTEXT"))
        except ValueError:
            pass

    if os.getenv("ENDPOINT1"):
        config["endpoint1"] = os.getenv("ENDPOINT1")
    if os.getenv("MODEL1"):
        config["model1"] = os.getenv("MODEL1")
    if os.getenv("API_KEY1"):
        config["api_key1"] = os.getenv("API_KEY1")

    if os.getenv("ENDPOINT2"):
        config["endpoint2"] = os.getenv("ENDPOINT2")
    if os.getenv("MODEL2"):
        config["model2"] = os.getenv("MODEL2")
    if os.getenv("API_KEY2"):
        config["api_key2"] = os.getenv("API_KEY2")

    # 2. Load from JSON config file if present
    if os.path.isfile(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            s1 = data.get("server1") or data.get("server_1") or data.get("server")
            s2 = data.get("server2") or data.get("server_2")

            nested_first = set()
            if isinstance(s1, dict):
                for key, value in (
                    ("endpoint", s1.get("endpoint") or s1.get("url") or s1.get("base_url")),
                    ("model", s1.get("model")),
                    ("api_key", s1.get("api_key")),
                    ("max_context", s1.get("max_context")),
                ):
                    if value:
                        config[key + "1"] = value
                        config[key] = value
                        nested_first.add(key)

            if isinstance(s2, dict):
                for key, value in (
                    ("endpoint", s2.get("endpoint") or s2.get("url") or s2.get("base_url")),
                    ("model", s2.get("model")),
                    ("api_key", s2.get("api_key")),
                    ("max_context", s2.get("max_context")),
                ):
                    if value:
                        config[key + "2"] = value

            for key, value in (
                ("endpoint", data.get("endpoint") or data.get("base_url") or data.get("url")),
                ("model", data.get("model")),
                ("api_key", data.get("api_key")),
                ("max_context", data.get("max_context") or data.get("max_model_len")),
            ):
                if value:
                    config[key] = value
                    if key not in nested_first and not data.get(key + "1"):
                        config[key + "1"] = value

            for key in ("endpoint", "model", "api_key", "max_context"):
                if data.get(key + "1"):
                    config[key + "1"] = data[key + "1"]
                    aliases = ("endpoint", "base_url", "url") if key == "endpoint" else (
                        ("max_context", "max_model_len") if key == "max_context" else (key,))
                    if not any(data.get(alias) for alias in aliases):
                        config[key] = data[key + "1"]
                if data.get(key + "2"):
                    config[key + "2"] = data[key + "2"]

            for key, value in data.items():
                if key not in config:
                    config[key] = value

            loaded_from_file = True
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to read config from {target_path}: {e}\n")

    return config, target_path, loaded_from_file


def save_config(endpoint, model, api_key="", max_context=None,
                endpoint2=None, model2=None, api_key2="", max_context2=None,
                config_file=None):
    """
    Saves single-server or two-server configuration to config.json.
    """
    target_path = os.path.abspath(config_file) if config_file else DEFAULT_CONFIG_PATH

    if endpoint2:
        data = {
            "server1": {
                "endpoint": endpoint or "",
                "model": model or "",
                "api_key": api_key or "",
                "max_context": max_context
            },
            "server2": {
                "endpoint": endpoint2 or "",
                "model": model2 or "",
                "api_key": api_key2 or "",
                "max_context": max_context2
            }
        }
    else:
        data = {
            "endpoint": endpoint or "",
            "model": model or "",
            "api_key": api_key or "",
            "max_context": max_context
        }

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Config may contain API keys and internal endpoints; restrict to owner.
        try:
            os.chmod(target_path, 0o600)
        except Exception:
            pass
        return target_path
    except Exception as e:
        sys.stderr.write(f"Error saving configuration to {target_path}: {e}\n")
        return None
