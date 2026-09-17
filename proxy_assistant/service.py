from __future__ import annotations

from .clients import AssistantClient, AssistantReply
from .history import HistoryStore, Message


class ChatService:
    def __init__(self, client: AssistantClient, store: HistoryStore) -> None:
        self.client = client
        self.store = store
        self.history = store.load()

    def ask(self, user_text: str) -> AssistantReply:
        reply = self.client.reply(self.history, user_text)
        self.history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply.text},
            ]
        )
        self.store.save(self.history)
        return reply

    def clear(self) -> None:
        self.history.clear()
        self.store.clear()

    def formatted_history(self) -> str:
        if not self.history:
            return "История пуста."
        labels = {"user": "Вы", "assistant": "Ассистент"}
        return "\n".join(
            f"{labels[message['role']]}: {message['content']}"
            for message in self.history
        )

