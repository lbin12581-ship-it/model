from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class LLMConfig:
    api_key: str
    model: str
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    timeout: int = 90

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        load_env_file()
        has_deepseek_key = bool(os.getenv("DEEPSEEK_API_KEY"))
        provider = os.getenv("LLM_PROVIDER") or ("deepseek" if has_deepseek_key else "openai")
        provider = provider.lower()
        api_key = os.getenv("DEEPSEEK_API_KEY") if provider == "deepseek" else None
        api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        default_base_url = {
            "deepseek": "https://api.deepseek.com",
            "openai": "https://api.openai.com/v1",
        }.get(provider, "https://api.deepseek.com")
        default_model = {
            "deepseek": "deepseek-v4-flash",
            "openai": "gpt-4o-mini",
        }.get(provider, "deepseek-v4-flash")
        base_url = os.getenv("LLM_BASE_URL")
        model = os.getenv("LLM_MODEL")
        if provider == "deepseek" and base_url == "https://api.openai.com/v1":
            base_url = None
        if provider == "deepseek" and model == "gpt-4o-mini":
            model = None
        return cls(
            api_key=api_key,
            provider=provider,
            model=model or default_model,
            base_url=base_url or default_base_url,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            timeout=int(os.getenv("LLM_TIMEOUT", "90")),
        )


class LLMClient:
    """Small OpenAI-compatible chat completions client.

    It works with OpenAI and most OpenAI-compatible providers, such as
    DeepSeek, Qwen-compatible gateways, local one-api/new-api gateways, etc.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.last_error: str | None = None

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        config = LLMConfig.from_env()
        return cls(config) if config else None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | list[Any] | None:
        try:
            content = self.chat(system_prompt, user_prompt)
            return self._parse_json(content)
        except Exception as exc:  # Keep the app usable when the model is unavailable.
            self.last_error = str(exc)
            return None

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        result = json.loads(raw)
        return result["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any] | list[Any] | None:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
            return json.loads(match.group(1)) if match else None


def llm_status_text() -> str:
    config = LLMConfig.from_env()
    if not config:
        return "未配置 DeepSeek API，当前使用本地规则智能体。"
    return f"已配置大模型：{config.provider} / {config.model} ({config.base_url})"
