"""Regression test: the Telegram frontend enforces telegram.allowed_user_ids.

aiogram, anthropic and the AI modules are stubbed via sys.modules so the test
runs with only stdlib + pytest. The fake Dispatcher records (handler, filters)
and applies filters in order, like aiogram 3: a filter returning False means
the update is not handled by that handler.
"""
import asyncio
import json
import sys
import types

import pytest


# ---- stubs (installed before importing system.platforms.telegram) ----------
class _Registry:
    def __init__(self):
        self.handlers = []

    def register(self, handler, *filters, **kwargs):
        self.handlers.append((handler, filters))


class _Dispatcher:
    def __init__(self):
        self.message = _Registry()

    async def start_polling(self, bot):  # pragma: no cover - never called
        raise AssertionError("polling must not run in tests")


class _Bot:
    def __init__(self, token=None):
        self.token = token


class _Command:
    def __init__(self, name):
        self.name = name

    def __call__(self, message):
        text = getattr(message, "text", None) or ""
        return bool(text) and text.split()[0].split("@")[0] == "/" + self.name


_aiogram = types.ModuleType("aiogram")
_aiogram.Bot = _Bot
_aiogram.Dispatcher = _Dispatcher
_aiogram.types = types.SimpleNamespace(Message=object)
_filters = types.ModuleType("aiogram.filters")
_filters.Command = _Command
_aiogram.filters = _filters
_context = types.ModuleType("system.ai.context")
_context.LLMContext = object
_memory = types.ModuleType("system.ai.memory")
_memory.MemoryManager = object
sys.modules["aiogram"] = _aiogram
sys.modules["aiogram.filters"] = _filters
sys.modules["system.ai.context"] = _context
sys.modules["system.ai.memory"] = _memory

from system.platforms.telegram import TelegramBot  # noqa: E402


# ---- fakes -----------------------------------------------------------------
class FakeLLM:
    def __init__(self):
        self.calls = {"get_response": 0, "update_context": 0, "clear_context": 0,
                      "get_current_context": 0}
        self.context = [{"role": "user", "content": "OWNER_CONTEXT"}]
        self.xml_processor = types.SimpleNamespace(set_telegram_handler=lambda h: None)

    async def get_response(self, text):
        self.calls["get_response"] += 1
        return "ok", []

    def update_context(self, user_text, response):
        self.calls["update_context"] += 1
        self.context.append({"role": "user", "content": user_text})

    def clear_context(self):
        self.calls["clear_context"] += 1
        self.context = []

    def get_current_context(self):
        self.calls["get_current_context"] += 1
        return self.context

    def get_active_tasks(self):
        return {}

    def get_task_history(self):
        return {}


class FakeMemory:
    def __init__(self):
        self.stored = []
        self.retrieved = 0

    async def store_memory(self, content, tags):
        self.stored.append((content, tags))

    async def retrieve_memories(self, limit=1):
        self.retrieved += 1
        return []


class FakeMessage:
    def __init__(self, uid, text):
        self.from_user = None if uid is None else types.SimpleNamespace(id=uid)
        self.chat = types.SimpleNamespace(id=uid)
        self.text = text
        self.replies = []

    async def reply(self, text):
        self.replies.append(text)


def dispatch(bot, message):
    """Route like aiogram: first handler whose filters all pass, in order."""
    async def run():
        for handler, filters in bot.dp.message.handlers:
            if all(f(message) for f in filters):
                await handler(message)
                return handler.__name__
        return None
    return asyncio.run(run())


def make_bot(tmp_path, monkeypatch, allowed):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "platforms.json").write_text(json.dumps({"telegram": {
        "message_timeout": 60, "max_message_length": 4096,
        "allowed_user_ids": allowed,
        "commands": {"start": "s", "help": "h", "status": "st", "clear": "c"}}}))
    tool = {"enabled": True, "max_execution_time": 30}
    (cfg / "config.json").write_text(json.dumps({"system": {"tools": {
        "python": tool, "terminal": tool, "perplexity": tool}}}))
    monkeypatch.chdir(tmp_path)
    llm, mem = FakeLLM(), FakeMemory()
    return TelegramBot(llm, mem), llm, mem


ALL_TEXTS = ["/start", "/help", "/status", "/clear", "/tasks", "please run echo hi"]


# ---- tests -----------------------------------------------------------------
def test_empty_allowlist_refuses_to_start(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="allowed_user_ids"):
        make_bot(tmp_path, monkeypatch, [])


def test_allowed_sender_reaches_clear_and_message(tmp_path, monkeypatch):
    bot, llm, mem = make_bot(tmp_path, monkeypatch, [101])
    assert dispatch(bot, FakeMessage(101, "/clear")) == "cmd_clear"
    assert llm.calls["clear_context"] == 1
    assert dispatch(bot, FakeMessage(101, "hello")) == "handle_message"
    assert llm.calls["get_response"] == 1
    assert mem.stored


@pytest.mark.parametrize("sender", [202, None], ids=["excluded_202", "no_from_user"])
@pytest.mark.parametrize("text", ALL_TEXTS)
def test_unauthorized_sender_is_not_handled(tmp_path, monkeypatch, sender, text):
    bot, llm, mem = make_bot(tmp_path, monkeypatch, [101])
    calls_before = dict(llm.calls)
    context_before = list(llm.context)
    msg = FakeMessage(sender, text)
    assert dispatch(bot, msg) is None
    assert llm.calls == calls_before
    assert llm.context == context_before
    assert mem.stored == [] and mem.retrieved == 0
    assert msg.replies == []
    assert bot.current_message is None
