"""V0.9.9.2 end-to-end checks on the real NiceGUI chat page.

These drive the actual ``/chat`` page through NiceGUI's simulated user: the
page subscribes to the shared chat runtime, sends a real utterance through the
orchestrator, renders streamed deltas, writes an explicit self-report without
asking a confirmation question, and renders proactive events through the same
bubble function. The DeepSeek
client is replaced by an offline fake; nothing touches the network.

If NiceGUI's testing plugin is unavailable the module is skipped.
"""

from __future__ import annotations

import pytest

nicegui_testing = pytest.importorskip("nicegui.testing")
from nicegui import ui  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]

from companion.core.response_events import (  # noqa: E402
    RESPONSE_COMPLETED,
    ResponseEvent,
)
from companion.domain.models import utcnow  # noqa: E402
from companion.providers.deepseek_client import ChatResponse, StreamDelta  # noqa: E402
from companion.storage.repositories import ProactiveMessageRecord  # noqa: E402
from companion.ui import app as ui_app  # noqa: E402

LIVE_TEXT = "夜深了，记得早点休息。"


class FakeDeepSeek:
    """Offline DeepSeek: judge/analysis say no, replies stream a short text."""

    def __init__(self, *, reply="好的。", chunks=None):
        self.reply = reply
        self.chunks = list(chunks) if chunks is not None else ["好", "的。"]

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        content = (
            self.reply
            if request_type == "response"
            else '{"should_initiate": false, "reason": "offline"}'
        )
        return ChatResponse(
            content=content,
            model="fake-model",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    async def stream_completion(self, messages, *, request_type="response", **kwargs):
        for chunk in self.chunks:
            yield StreamDelta(content=chunk)
        yield StreamDelta(
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            usage_reported=True,
        )


@pytest.fixture(autouse=True)
async def web_app(tmp_path, monkeypatch, nicegui_reset_globals):
    """Configure the real app after NiceGUI's reset and before startup."""
    monkeypatch.setattr(ui_app, "_shared_deepseek_client", lambda: FakeDeepSeek())
    # The fake client cannot answer the semantic router's JSON contract; this
    # test is about page delivery, not routing. Production keeps the default.
    monkeypatch.setenv("AVA_ROUTER_MODE", "legacy")
    await ui_app._stop_chat_coordinator()
    ui_app.create_app(tmp_path / "ui.db", provider="deepseek")
    # NiceGUI invokes async startup handlers fire-and-forget; start the shared
    # runtime eagerly so the test body sees a ready runtime. The handler that
    # runs during the lifespan is idempotent.
    await ui_app._start_chat_coordinator()
    yield ui_app
    await ui_app._stop_chat_coordinator()


async def _owner(rt):
    return await rt.registry.register(
        "Web",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


def _labels_with(user, text: str) -> list:
    return [
        element
        for element in user.client.elements.values()
        if isinstance(element, ui.label) and element.text == text
    ]


def _labels_containing(user, text: str) -> list:
    return [
        element
        for element in user.client.elements.values()
        if isinstance(element, ui.label) and text in (element.text or "")
    ]


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition was not met in time")


async def _type_and_send(user, text: str) -> None:
    box = user.find(ui.textarea).elements.pop()
    box.value = text
    user.find("发送").click()


async def test_chat_page_streams_a_direct_self_report_without_a_question(
    user, web_app
):
    rt = web_app._COORDINATOR.runtime
    person = await _owner(rt)
    await user.open("/chat")

    await _type_and_send(user, "我叫李约")
    await user.should_see("好的。", retries=50)
    # No confirmation question is added to the reply.
    assert _labels_containing(user, "确认一下") == []

    action = await rt.repos.analysis.get_action(
        (await rt.repos.utterance.recent_context(
            f"chat:{person.person_id}", None, limit=1
        ))[-1].event_id
    )
    assert "确认一下" not in (action.response_text or "")
    entries = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="identity.name"
    )
    assert [entry["content"] for entry in entries] == ["李约"]

    await _type_and_send(user, "对")
    await _wait_until(
        lambda: len(_labels_with(user, "好的。")) >= 2
    )
    assert _labels_containing(user, "确认一下") == []
    entries = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="identity.name"
    )
    assert [entry["content"] for entry in entries] == ["李约"]


async def test_chat_page_renders_live_proactive_events_without_duplicates(
    user, web_app
):
    rt = web_app._COORDINATOR.runtime
    person = await _owner(rt)
    conversation = f"chat:{person.person_id}"
    await user.open("/chat")

    await _type_and_send(user, "今天天气不错")
    await user.should_see("好的。", retries=50)

    # A delivered proactive event is routed to the page's subscription and
    # rendered as a normal Ava bubble (no status label, no second surface).
    coordinator = web_app._COORDINATOR
    client = user.client
    callback = coordinator._subscribers[conversation][client.id]
    event = ResponseEvent(
        type=RESPONSE_COMPLETED,
        event_id="event-live",
        conversation_id=conversation,
        response_id="pmsg-live",
        sequence=1,
        status="COMPLETED",
        text=LIVE_TEXT,
        report_status="COMPLETED",
        action="RESPOND",
    )
    await callback(event)
    await user.should_see(LIVE_TEXT, retries=50)
    assert len(_labels_with(user, LIVE_TEXT)) == 1

    # The same response id is never rendered twice (live replay protection).
    await callback(event)
    assert len(_labels_with(user, LIVE_TEXT)) == 1

    # Persist the same style of message and reload: history shows it once,
    # and a live delivery of that exact message is suppressed as a duplicate.
    previous = (
        await rt.repos.utterance.recent_context(conversation, None, limit=1)
    )[-1]
    await rt.repos.proactive.save(
        ProactiveMessageRecord(
            message_id="pmsg-history",
            conversation_id=conversation,
            person_id=person.person_id,
            source_event_id=previous.event_id,
            mode="LONG_IDLE",
            origin="AVA_INITIATED",
            text="该休息了。",
            status="COMPLETED",
            finish_reason="stop",
            model="fake-model",
            created_at=utcnow(),
        )
    )

    await user.open("/chat")
    await user.should_see("该休息了。", retries=50)
    assert len(_labels_with(user, "该休息了。")) == 1

    reloaded = web_app._COORDINATOR._subscribers[conversation][user.client.id]
    replay = ResponseEvent(
        type=RESPONSE_COMPLETED,
        event_id="event-history",
        conversation_id=conversation,
        response_id="pmsg-history",
        sequence=1,
        status="COMPLETED",
        text="该休息了。",
        report_status="COMPLETED",
        action="RESPOND",
    )
    await reloaded(replay)
    assert len(_labels_with(user, "该休息了。")) == 1
