from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[Message]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Не удалось прочитать историю {self.path}: {exc}") from exc

        if not isinstance(raw, list):
            raise ValueError("Файл истории должен содержать JSON-массив")

        messages: list[Message] = []
        for item in raw:
            if (
                not isinstance(item, dict)
                or item.get("role") not in {"user", "assistant"}
                or not isinstance(item.get("content"), str)
            ):
                raise ValueError("В файле истории найдено некорректное сообщение")
            messages.append({"role": item["role"], "content": item["content"]})
        return messages

    def save(self, messages: list[Message]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

