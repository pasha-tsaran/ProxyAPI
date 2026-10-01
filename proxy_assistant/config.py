from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Ошибка пользовательской конфигурации."""


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должен быть целым числом") from exc
    if value <= 0:
        raise ConfigError(f"{name} должен быть больше нуля")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должен быть числом") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} должен быть конечным числом больше нуля")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str
    openai_model: str
    claude_model: str
    openai_base_url: str
    anthropic_base_url: str
    timeout: float
    max_output_tokens: int
    thinking_budget_tokens: int
    history_file: Path
    system_prompt: str = "Вы полезный ассистент. Отвечайте на русском языке."

    @classmethod
    def from_env(cls, require_key: bool = True) -> "Settings":
        load_dotenv()
        api_key = os.getenv("PROXYAPI_API_KEY", "").strip()
        if require_key and not api_key:
            raise ConfigError(
                "В файле .env не заполнен PROXYAPI_API_KEY. "
                "Создайте ключ в ProxyAPI и вставьте его после знака '='."
            )

        max_tokens = _positive_int("MAX_OUTPUT_TOKENS", 4096)
        thinking_budget = _positive_int("THINKING_BUDGET_TOKENS", 1500)
        if thinking_budget >= max_tokens:
            raise ConfigError(
                "THINKING_BUDGET_TOKENS должен быть меньше MAX_OUTPUT_TOKENS"
            )

        return cls(
            api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini").strip(),
            claude_model=os.getenv(
                "CLAUDE_MODEL", "anthropic/claude-sonnet-4-5"
            ).strip(),
            openai_base_url=os.getenv(
                "PROXYAPI_OPENAI_BASE_URL", "https://api.proxyapi.ru/v1"
            ).rstrip("/"),
            anthropic_base_url=os.getenv(
                "PROXYAPI_ANTHROPIC_BASE_URL", "https://api.proxyapi.ru"
            ).rstrip("/"),
            timeout=_positive_float("REQUEST_TIMEOUT_SECONDS", 60),
            max_output_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget,
            history_file=Path(os.getenv("HISTORY_FILE", "data/history.json")),
        )

