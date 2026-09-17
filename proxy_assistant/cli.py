from __future__ import annotations

import logging
import sys

import anthropic
import openai

from .clients import ClaudeThinkingClient, OpenAIChatClient
from .config import ConfigError, Settings
from .history import HistoryStore
from .service import ChatService

EXIT_COMMANDS = {"exit", "quit", "выход"}


def choose_mode() -> str:
    print("\nВыберите режим:")
    print("  1 — обычная модель (OpenAI Chat Completions)")
    print("  2 — Claude 4.5 Sonnet с reasoning [по умолчанию]")
    while True:
        choice = input("Режим [2]: ").strip()
        if choice in {"", "2"}:
            return "thinking"
        if choice == "1":
            return "standard"
        print("Введите 1 или 2.")


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, (openai.APITimeoutError, anthropic.APITimeoutError)):
        return "ProxyAPI не ответил вовремя. Повторите запрос позже."
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return "Ключ ProxyAPI отклонён. Проверьте PROXYAPI_API_KEY в .env."
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return "Превышен лимит запросов или недостаточно средств на балансе."
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return "Нет соединения с ProxyAPI. Проверьте интернет и повторите попытку."
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return f"ProxyAPI вернул ошибку HTTP {exc.status_code}: {exc.message}"
    return f"Не удалось получить ответ: {exc}"


def run_chat(settings: Settings, mode: str) -> None:
    client = (
        ClaudeThinkingClient(settings)
        if mode == "thinking"
        else OpenAIChatClient(settings)
    )
    service = ChatService(client, HistoryStore(settings.history_file))
    mode_name = "Claude с reasoning" if mode == "thinking" else "обычная модель"
    logging.info("Режим: %s; модель: %s", mode_name, client.model)
    logging.info("Загружено сообщений из истории: %d", len(service.history))
    print("Команды: /history — история, /clear — очистить, exit — выйти.")

    while True:
        try:
            user_text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            break

        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            print("\nИстория диалога:")
            print(service.formatted_history())
            print("\nДо свидания!")
            break
        if user_text.lower() == "/history":
            print(service.formatted_history())
            continue
        if user_text.lower() == "/clear":
            service.clear()
            print("История очищена.")
            continue

        try:
            reply = service.ask(user_text)
        except Exception as exc:  # Ошибку API показываем без аварийного завершения чата.
            logging.error(_friendly_error(exc))
            continue

        if reply.reasoning:
            print(f"\nReasoning:\n{reply.reasoning}")
        print(f"\nАссистент: {reply.text}")
        if reply.input_tokens is not None and reply.output_tokens is not None:
            print(
                f"[Токены: вход {reply.input_tokens}, выход {reply.output_tokens}]"
            )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.info("Запуск консольного ассистента ProxyAPI")
    try:
        settings = Settings.from_env()
        mode = choose_mode()
        run_chat(settings, mode)
    except (ConfigError, ValueError) as exc:
        logging.error("Ошибка настройки: %s", exc)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nЗапуск отменён.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

