from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from typing import Any, Dict, Optional

import requests

from .config_loader import load_vlm_config


DEFAULT_QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class ApiVLMClient:
    """OpenAI-compatible vision-language API client used by extraction."""

    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger
        self.provider = str(config.get("provider") or "qwen").strip().lower()
        self.provider_name = f"{self.provider}_api"
        self.api_key = str(config.get("api_key") or "").strip()
        self.model = str(config.get("model") or "qwen-vl-plus").strip()
        self.base_url = str(config.get("base_url") or DEFAULT_QWEN_URL).strip()
        self.timeout_seconds = int(config.get("timeout_seconds") or 120)
        self.retry_count = int(config.get("retry_count") or 2)
        self.max_tokens = int(config.get("max_tokens") or 700)
        self.temperature = float(config.get("temperature") or 0.1)
        self.image_detail = str(config.get("image_detail") or "auto")

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    def generate_description(self, image_path: str, prompt: Optional[str] = None) -> str:
        prompt = prompt or (
            "请分析这张教学材料图片。请提取核心概念、关键文字、图表/公式/代码信息、"
            "与课程知识点的关系，并给出适合构建知识图谱和向量检索的结构化中文描述。"
        )
        return self._chat_with_image(image_path, prompt)

    def recognize_formula(self, image_path: str) -> Dict[str, Any]:
        prompt = (
            "请识别图片中的数学、物理或工程公式。按以下格式回答：\n"
            "LaTeX: [公式]\n"
            "名称: [公式名称]\n"
            "含义: [公式含义]\n"
            "变量: [变量说明]\n"
            "条件: [适用条件]"
        )
        response = self._chat_with_image(image_path, prompt, max_tokens=500)
        latex = ""
        description = ""
        for line in response.splitlines():
            if "latex:" in line.lower():
                latex = line.split(":", 1)[1].strip()
            elif line.startswith(("含义:", "意义:", "说明:")):
                description = line.split(":", 1)[1].strip()
        return {"latex": latex or response.strip(), "description": description, "raw_response": response}

    def recognize_code(self, image_path: str) -> Dict[str, Any]:
        prompt = (
            "请判断图片中是否包含程序代码。若没有代码，只回答：无代码。\n"
            "若有代码，请按以下格式回答：\n"
            "语言: [编程语言]\n"
            "```[语言]\n"
            "[完整代码]\n"
            "```\n"
            "功能: [功能说明]"
        )
        response = self._chat_with_image(image_path, prompt, max_tokens=700)
        lower = response.lower()
        result = {
            "code": "",
            "language": "txt",
            "description": "",
            "raw_response": response,
            "has_code": False,
        }
        if "无代码" in response or "no code" in lower:
            return result

        import re

        lang_match = re.search(r"语言\s*[:：]\s*(.+)", response)
        if lang_match:
            result["language"] = lang_match.group(1).strip()
        code_match = re.search(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            if code_match.group(1).strip():
                result["language"] = code_match.group(1).strip()
            result["code"] = code_match.group(2).strip()
        desc_match = re.search(r"功能\s*[:：]\s*(.+)", response, re.DOTALL)
        if desc_match:
            result["description"] = desc_match.group(1).strip()
        result["has_code"] = len(result["code"]) >= 10
        return result

    def recognize_table(self, image_path: str) -> Dict[str, Any]:
        prompt = (
            "请识别图片中的表格。尽量保留表头、行列关系和单元格内容。"
            "请按以下格式回答：\n"
            "表头: [列1, 列2, ...]\n"
            "数据:\n"
            "[用Markdown表格或逐行文本表示]\n"
            "说明: [表格主题和用途]"
        )
        response = self._chat_with_image(image_path, prompt, max_tokens=800)
        headers = ""
        description = ""
        for line in response.splitlines():
            if line.startswith(("表头:", "Headers:")):
                headers = line.split(":", 1)[1].strip()
            elif line.startswith(("说明:", "Description:")):
                description = line.split(":", 1)[1].strip()
        return {
            "table_data": response,
            "headers": headers,
            "description": description,
            "raw_response": response,
        }

    def _chat_with_image(self, image_path: str, prompt: str, max_tokens: Optional[int] = None) -> str:
        if not self.available:
            raise RuntimeError("VLM API is not configured. Please fill config/vlm_api.yaml or set env vars.")
        image_url = self._image_to_data_url(image_path)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": self.image_detail},
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
                return str(content).strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_count:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"VLM API request failed: {last_error}")

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


def create_vlm_client(logger=None) -> Optional[ApiVLMClient]:
    config = load_vlm_config()
    enabled = config.get("enabled", True)
    if str(enabled).lower() in {"0", "false", "no", "off"}:
        return None
    client = ApiVLMClient(config, logger=logger)
    if not client.available:
        if logger:
            logger.warning("VLM API is enabled but api_key/model/base_url is incomplete.")
        return None
    return client
