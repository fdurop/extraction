from __future__ import annotations

import os
from typing import Any, Dict


def find_project_root(start_path: str | None = None) -> str:
    current = os.path.abspath(start_path or os.getcwd())
    for _ in range(8):
        if os.path.exists(os.path.join(current, "run_src2.py")) and os.path.isdir(
            os.path.join(current, "src2")
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.abspath(start_path or os.getcwd())


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0:1] in {'"', "'"} and value[-1:] == value[0]:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: str) -> Dict[str, Any]:
    """Small YAML reader for the simple key/value config used by this project."""
    if not os.path.exists(path):
        return {}

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            text = line.strip()
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value == "":
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_scalar(value)
    return root


def load_vlm_config(config_path: str | None = None) -> Dict[str, Any]:
    project_root = find_project_root(os.path.dirname(__file__))
    path = config_path or os.getenv(
        "EXTRACTION_VLM_CONFIG",
        os.path.join(project_root, "config", "vlm_api.yaml"),
    )
    data = load_simple_yaml(path)
    vlm = dict(data.get("vlm") or {})

    env_provider = os.getenv("VLM_PROVIDER")
    env_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("VLM_API_KEY")
    env_model = os.getenv("QWEN_VL_MODEL") or os.getenv("VLM_MODEL")

    if env_provider:
        vlm["provider"] = env_provider
    if env_key:
        vlm["api_key"] = env_key
    if env_model:
        vlm["model"] = env_model

    vlm["config_path"] = path
    return vlm


def load_embedding_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load the embedding API config, reusing the VLM key by default."""
    project_root = find_project_root(os.path.dirname(__file__))
    path = config_path or os.getenv(
        "EXTRACTION_VLM_CONFIG",
        os.path.join(project_root, "config", "vlm_api.yaml"),
    )
    data = load_simple_yaml(path)
    vlm = dict(data.get("vlm") or {})
    embedding = dict(data.get("embedding") or {})

    env_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    env_model = os.getenv("QWEN_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL")
    if env_key:
        embedding["api_key"] = env_key
    elif not embedding.get("api_key"):
        embedding["api_key"] = vlm.get("api_key", "")
    if env_model:
        embedding["model"] = env_model

    embedding.setdefault("enabled", True)
    embedding.setdefault("provider", "qwen")
    embedding.setdefault("model", "qwen3.7-text-embedding")
    embedding.setdefault("base_url", "https://dashscope.aliyuncs.com/api/v1")
    embedding.setdefault("dimension", 1024)
    embedding.setdefault("batch_size", 10)
    embedding.setdefault("timeout_seconds", 120)
    embedding.setdefault("retry_count", 2)
    embedding.setdefault("output_type", "dense")
    embedding.setdefault(
        "query_instruction",
        "Retrieve relevant evidence from Chinese educational course materials.",
    )
    embedding["config_path"] = path
    return embedding
