from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from proxy_assistant.clients import AssistantReply, ClaudeThinkingClient, OpenAIChatClient
from proxy_assistant.config import ConfigError, Settings
from proxy_assistant.history import HistoryStore
from proxy_assistant.service import ChatService


class FakeClient:
    model = "fake/model"

    def __init__(self) -> None:
        self.received_history = []

    def reply(self, history, user_text):
        self.received_history = list(history)
        return AssistantReply(text=f"Ответ на: {user_text}", input_tokens=3, output_tokens=4)


class FailingClient:
    model = "fake/failing"

    def reply(self, history, user_text):
        raise RuntimeError("network failed")


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
def test_settings_reject_nonfinite_timeout(monkeypatch, raw):
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", raw)
    with pytest.raises(ConfigError):
        Settings.from_env(require_key=False)


def test_failed_save_preserves_in_memory_history(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "history.json")
    service = ChatService(FakeClient(), store)
    monkeypatch.setattr(store, "save", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        service.ask("Не сохраняй меня")
    assert service.history == []


def test_failed_clear_preserves_in_memory_history(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "history.json")
    service = ChatService(FakeClient(), store)
    service.ask("Привет")
    previous = list(service.history)
    monkeypatch.setattr(store, "clear", Mock(side_effect=OSError("access denied")))
    with pytest.raises(OSError):
        service.clear()
    assert service.history == previous
    assert store.load() == previous


def test_history_is_saved_and_loaded_between_services(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.json")
    first = ChatService(FakeClient(), store)
    first.ask("Привет")

    fake = FakeClient()
    second = ChatService(fake, store)
    second.ask("Ты помнишь меня?")

    assert fake.received_history == [
        {"role": "user", "content": "Привет"},
        {"role": "assistant", "content": "Ответ на: Привет"},
    ]
    assert len(second.history) == 4


def test_failed_request_does_not_change_history(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.json")
    service = ChatService(FailingClient(), store)

    with pytest.raises(RuntimeError):
        service.ask("Не сохраняй меня")

    assert service.history == []
    assert not store.path.exists()


def test_clear_removes_saved_history(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.json")
    service = ChatService(FakeClient(), store)
    service.ask("Привет")

    service.clear()

    assert service.history == []
    assert not store.path.exists()


def test_settings_reject_invalid_thinking_budget(monkeypatch) -> None:
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "100")
    monkeypatch.setenv("THINKING_BUDGET_TOKENS", "100")

    with pytest.raises(ConfigError):
        Settings.from_env(require_key=False)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        openai_model="openai/gpt-4o-mini",
        claude_model="anthropic/claude-sonnet-4-5",
        openai_base_url="https://api.proxyapi.ru/v1",
        anthropic_base_url="https://api.proxyapi.ru",
        timeout=10,
        max_output_tokens=4096,
        thinking_budget_tokens=1500,
        history_file=tmp_path / "history.json",
    )


def test_openai_client_sends_history_to_chat_completions(tmp_path: Path) -> None:
    client = OpenAIChatClient(make_settings(tmp_path))
    create = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Готово"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )
    )
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    reply = client.reply([{"role": "user", "content": "Старый вопрос"}], "Новый")

    assert reply.text == "Готово"
    sent = create.call_args.kwargs
    assert sent["model"] == "openai/gpt-4o-mini"
    assert sent["messages"][-2:] == [
        {"role": "user", "content": "Старый вопрос"},
        {"role": "user", "content": "Новый"},
    ]


def test_claude_client_extracts_thinking_and_usage(tmp_path: Path) -> None:
    client = ClaudeThinkingClient(make_settings(tmp_path))
    create = Mock(
        return_value=SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="Проверяю решение"),
                SimpleNamespace(type="text", text="Ответ"),
            ],
            usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        )
    )
    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))

    reply = client.reply([], "Задача")

    assert reply.text == "Ответ"
    assert reply.reasoning == "Проверяю решение"
    assert (reply.input_tokens, reply.output_tokens) == (12, 8)
    sent = create.call_args.kwargs
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 1500}
    assert sent["model"] == "anthropic/claude-sonnet-4-5"
