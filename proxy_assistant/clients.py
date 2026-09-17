from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from anthropic import Anthropic
from openai import OpenAI

from .config import Settings
from .history import Message


class AssistantClient(Protocol):
    model: str

    def reply(self, history: list[Message], user_text: str) -> "AssistantReply": ...


@dataclass(frozen=True)
class AssistantReply:
    text: str
    reasoning: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class OpenAIChatClient:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.openai_model
        self.system_prompt = settings.system_prompt
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.openai_base_url,
            timeout=settings.timeout,
            max_retries=2,
        )

    def reply(self, history: list[Message], user_text: str) -> AssistantReply:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_text},
        ]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("Модель вернула пустой ответ")
        usage = response.usage
        return AssistantReply(
            text=text,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


class ClaudeThinkingClient:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.claude_model
        self.system_prompt = settings.system_prompt
        self.max_tokens = settings.max_output_tokens
        self.thinking_budget = settings.thinking_budget_tokens
        self._client = Anthropic(
            api_key=settings.api_key,
            base_url=settings.anthropic_base_url,
            timeout=settings.timeout,
            max_retries=2,
        )

    def reply(self, history: list[Message], user_text: str) -> AssistantReply:
        response = self._client.messages.create(
            model=self.model,
            system=self.system_prompt,
            max_tokens=self.max_tokens,
            thinking={"type": "enabled", "budget_tokens": self.thinking_budget},
            messages=[*history, {"role": "user", "content": user_text}],  # type: ignore[arg-type]
        )

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)

        text = "\n".join(text_parts).strip()
        if not text:
            raise RuntimeError("Модель вернула ответ без текстового блока")
        reasoning = "\n".join(thinking_parts).strip() or None
        return AssistantReply(
            text=text,
            reasoning=reasoning,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

