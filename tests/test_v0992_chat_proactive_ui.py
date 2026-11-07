"""V0.9.9.2 checks: streaming confirmation, stop boundaries, web chat runtime.

The web chat surface must use exactly the same core as the CLI. These tests
therefore exercise the real ``Orchestrator``, ``ProfileConfirmationService`` and
dual-stage ``ProactiveManager`` through :class:`ChatCoordinator` — the only new
object is the long-lived runtime that keeps proactive cycles alive after the
HTTP request that started them has ended.

Coverage map (T01-T25 in the V0.9.9.2 brief):

T01  deltas == completed == persisted, question appears exactly once
T02  "对" writes the name once
T03  "不是" rejects without a write
T04  interrupted generation never shows a question / false confirmation
T05  explicit departure blocks both stages even when the judge says yes
T06  hypotheticals, quotes and questions are not stops
T07  a delivered web reply invokes the IMMEDIATE judge once
T08  IMMEDIATE YES shows exactly one message to the tab
T09  IMMEDIATE NO shows nothing
T10  LONG_IDLE runs after the configured window and delivers once
T11  new input cancels the pending idle stage
T12  truncated proactive output is not delivered
T13  history restores proactive rows once, in order
T14  two tabs share one generation and both see it
T15  different conversations are isolated
T16  closing one tab leaves the other subscribed
T17  closing the last tab cancels that conversation's cycle
T18  quiet mode blocks generation
T19  revoked permission blocks generation
T20  shutdown releases tasks and the shared database
T21  mock provider adds no proactive manager and makes no extra call
T22  a proactive delivery never schedules another cycle
T23  an open confirmation question blocks only the IMMEDIATE stage
T24  a restart resumes the confirmation timer without a duplicate question
T25  input in conversation A keeps conversation B's queued event
T26  silent WAIT/IGNORE turns never schedule a cycle
T27  a repeated completion schedules only one cycle

Everything is offline: a fake client stands in for DeepSeek and a scripted
judge stands in for the model judge.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from companion.core.response_events import (
    RESPONSE_COMPLETED,
    ResponseEvent,
)
from companion.domain.enums import CognitiveAction, EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_confirmation import (
    STATE_ACCEPTED_SELF_REPORT,
    STATE_AWAITING_REPLY,
    STATE_CONFIRMED,
    STATE_PENDING_QUESTION,
    STATE_REJECTED,
    ProfileConfirmationService,
)
from companion.proactive.departure import detect_explicit_stop
from companion.proactive.judge import ProactiveJudge
from companion.proactive.manager import MODE_IMMEDIATE, MODE_LONG_IDLE
from companion.providers.deepseek_client import ChatResponse, StreamDelta
from companion.providers.deepseek_proactive import ScriptedProactiveJudgeProvider
from companion.runtime import build_runtime
from companion.ui.app import load_chat_history
from companion.ui.chat_coordinator import ChatCoordinator

CONVERSATION = "chat:test-owner"
QUESTION = "确认一下，你的中文名是李约，对吗？"


# -- offline providers -----------------------------------------------------


class FakeClient:
    """Offline client: judge/analysis JSON, normal text for replies."""

    def __init__(self, *, response_text="好的。", chunks=None):
        self.response_text = response_text
        self.chunks = list(chunks) if chunks is not None else [response_text]
        self.calls: list[str] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(request_type)
        content = (
            self.response_text
            if request_type == "response"
            else json.dumps(
                {"should_initiate": False, "reason": "scripted"},
                ensure_ascii=False,
            )
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


class InterruptingClient(FakeClient):
    """Fails after one delta, like a dropped network mid-answer."""

    async def stream_completion(self, messages, *, request_type="response", **kwargs):
        yield StreamDelta(content="好")
        raise RuntimeError("connection lost")


class TruncatingClient(FakeClient):
    """Stops at the token limit: real text, incomplete turn."""

    async def stream_completion(self, messages, *, request_type="response", **kwargs):
        yield StreamDelta(content="我记住了。")
        yield StreamDelta(
            finish_reason="length",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            usage_reported=True,
        )


class CountingConfirmation(ProfileConfirmationService):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resume_calls = 0

    async def resume_pending(self) -> int:
        self.resume_calls += 1
        return await super().resume_pending()


# -- harness ---------------------------------------------------------------


@pytest.fixture
async def chat(tmp_path):
    """Factory for a shared chat runtime over a fresh database."""
    created: list[ChatCoordinator] = []

    async def _make(
        *,
        client=None,
        decisions=(),
        idle_delay: float = 0.02,
        confirmation_timeout: float | None = None,
        provider: str = "deepseek",
    ):
        client = client or FakeClient()
        scripted = ScriptedProactiveJudgeProvider(list(decisions))
        judge = ProactiveJudge(scripted)

        def factory(db):
            rt = build_runtime(
                db,
                provider=provider,
                deepseek_client=client,
                with_worker=False,
                router_mode="legacy",
                proactive_judge=judge,
                proactive_idle_delay_seconds=idle_delay,
            )
            if confirmation_timeout is not None:
                confirmation = CountingConfirmation(
                    profile_service=rt.profile_service,
                    repos=rt.repos,
                    no_reply_timeout_seconds=confirmation_timeout,
                )
                rt.profile_confirmation = confirmation
                rt.orchestrator.profile_confirmation = confirmation
            return rt

        coordinator = ChatCoordinator(
            tmp_path / "chat.db",
            provider=provider,
            runtime_factory=factory,
        )
        created.append(coordinator)
        rt = await coordinator.start()
        return coordinator, rt, client, scripted

    yield _make

    for coordinator in created:
        await coordinator.close()


def _collector():
    received: list[ResponseEvent] = []

    async def callback(event) -> None:
        received.append(event)

    return received, callback


async def _owner(rt, name="Web"):
    return await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


async def _web_turn(
    coordinator,
    rt,
    person,
    text,
    *,
    conversation=CONVERSATION,
    schedule=True,
):
    """Run one page-equivalent turn and return its stream evidence."""
    await rt.conversations.get_or_create(conversation)
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEXT,
    )
    coordinator.note_user_input(conversation, event.event_id)
    events = [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]
    completed = next(
        (item for item in events if item.type == RESPONSE_COMPLETED), None
    )
    deltas = "".join(
        item.delta or "" for item in events if item.type == "response_delta"
    )
    delivered = bool(
        completed is not None
        and completed.status == "COMPLETED"
        and completed.action == CognitiveAction.RESPOND.value
        and (completed.text or "").strip()
    )
    if schedule and delivered:
        coordinator.on_normal_response_delivered(
            conversation_id=conversation,
            person_id=person.person_id,
            source_event_id=event.event_id,
        )
    return event, events, deltas, completed


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met in time")


async def _attach_tab(coordinator, client_id="tab-1"):
    """A live subscriber is required: without one nothing may be generated."""
    received, callback = _collector()
    await coordinator.subscribe(CONVERSATION, client_id, callback)
    return received


async def _name_entries(rt, person):
    return await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="identity.name"
    )


# -- T01-T04: streaming confirmation delivery ------------------------------


async def test_deltas_completed_and_persisted_text_are_the_same_string(chat):
    """T01: the question is visible once and every surface agrees."""
    coordinator, rt, client, _ = await chat(
        client=FakeClient(
            response_text="好的。\n\n我记住了。",
            chunks=["好的。", "\n", "\n", "我记住了。"],
        )
    )
    person = await _owner(rt)
    event, events, deltas, completed = await _web_turn(
        coordinator, rt, person, "我叫李约"
    )

    assert completed is not None
    assert completed.status == "COMPLETED"
    assert deltas == completed.text
    assert deltas.count(QUESTION) == 1
    assert deltas.endswith(QUESTION)
    # The layout normaliser collapsed the paragraph break, and the persisted
    # action carries exactly the visible text.
    assert "好的。 我记住了。" in deltas
    assert "\n\n" not in deltas.removesuffix("\n\n" + QUESTION)

    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.response_text == completed.text

    started = [item for item in events if item.type == "response_started"]
    assert len(started) == 1
    assert {item.response_id for item in events if item.response_id} == {
        completed.response_id
    }
    sequences = [item.sequence for item in events]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)

    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_AWAITING_REPLY
    assert record.question_message_id == action.action_id
    assert await _name_entries(rt, person) == []


async def test_question_already_emitted_by_the_model_is_not_reprinted(chat):
    """T01 (retry case): the model itself produced the exact question."""
    coordinator, rt, client, _ = await chat(
        client=FakeClient(
            response_text=QUESTION,
            chunks=["确认一下，", "你的中文名是李约，", "对吗？"],
        )
    )
    person = await _owner(rt)
    _, _, deltas, completed = await _web_turn(coordinator, rt, person, "我叫李约")
    assert deltas == completed.text == QUESTION
    assert deltas.count(QUESTION) == 1
    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_AWAITING_REPLY


async def test_confirming_reply_writes_the_name_once(chat):
    """T02: "对" resolves the question and never asks again."""
    coordinator, rt, client, _ = await chat(
        client=FakeClient(response_text="好的。", chunks=["好", "的。"])
    )
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "我叫李约")

    _, _, deltas, completed = await _web_turn(coordinator, rt, person, "对")
    assert "确认一下" not in deltas
    assert deltas == completed.text
    assert [entry["content"] for entry in await _name_entries(rt, person)] == [
        "李约"
    ]
    records = await rt.repos.profile_confirmation.list_for_person(person.person_id)
    assert [record.state for record in records] == [STATE_CONFIRMED]

    # A later unrelated turn must not write it again.
    await _web_turn(coordinator, rt, person, "今天好热")
    assert len(await _name_entries(rt, person)) == 1


async def test_denial_rejects_without_writing(chat):
    """T03."""
    coordinator, rt, client, _ = await chat()
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "我叫李约")

    await _web_turn(coordinator, rt, person, "不是")
    assert await _name_entries(rt, person) == []
    records = await rt.repos.profile_confirmation.list_for_person(person.person_id)
    assert records[-1].state == STATE_REJECTED


async def test_deferral_is_not_a_confirmation_and_is_not_re_asked(chat):
    """An ambiguous reply neither confirms nor reprints the question."""
    coordinator, rt, client, _ = await chat()
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "我叫李约")

    _, _, deltas, _ = await _web_turn(coordinator, rt, person, "嗯，先说别的")
    assert "确认一下" not in deltas
    assert await _name_entries(rt, person) == []
    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_AWAITING_REPLY

    # The explicit yes still resolves it exactly once.
    await _web_turn(coordinator, rt, person, "对")
    assert [entry["content"] for entry in await _name_entries(rt, person)] == [
        "李约"
    ]


async def test_truncated_normal_reply_does_not_deliver_the_question(chat):
    """T04 (truncation): an incomplete reply never carries a confirmation."""
    coordinator, rt, client, _ = await chat(client=TruncatingClient())
    person = await _owner(rt)
    _, _, deltas, completed = await _web_turn(
        coordinator, rt, person, "我叫李约", schedule=False
    )
    assert completed.status == "TRUNCATED"
    assert completed.action == "RESPOND"
    assert QUESTION not in deltas
    assert deltas == "我记住了。"
    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_PENDING_QUESTION
    assert await _name_entries(rt, person) == []


async def test_interrupted_stream_shows_no_question_and_writes_nothing(chat):
    """T04: a failed turn is never a delivered confirmation."""
    coordinator, rt, client, _ = await chat(client=InterruptingClient())
    person = await _owner(rt)
    await rt.conversations.get_or_create(CONVERSATION)
    event = make_utterance_event(
        conversation_id=CONVERSATION,
        text="我叫李约",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEXT,
    )
    coordinator.note_user_input(CONVERSATION, event.event_id)
    events = [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]

    deltas = "".join(
        item.delta or "" for item in events if item.type == "response_delta"
    )
    assert QUESTION not in deltas
    assert not any(item.type == RESPONSE_COMPLETED for item in events)
    assert any(item.type == "response_failed" for item in events)

    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_PENDING_QUESTION
    assert await _name_entries(rt, person) == []


# -- T05-T07: explicit stop boundary and judge triggering ------------------


@pytest.mark.parametrize(
    "text",
    ["我要下线了", "我先去睡了", "我不想聊了", "别再跟我说话", "别烦我", "安静点", "拜拜", "晚安"],
)
async def test_explicit_stop_blocks_both_stages_even_with_a_yes_judge(chat, text):
    """T05."""
    coordinator, rt, client, judge = await chat(decisions=[True, True])
    await _attach_tab(coordinator)
    person = await _owner(rt)
    event, _, _, completed = await _web_turn(
        coordinator, rt, person, text, schedule=False
    )
    if completed is not None:
        assert completed.action == "RESPOND"
        coordinator.on_normal_response_delivered(
            conversation_id=CONVERSATION,
            person_id=person.person_id,
            source_event_id=event.event_id,
        )
    else:
        # Some stop phrases are answered with silence by the policy. The floor
        # must hold even if a cycle were scheduled, so drive the manager
        # directly to prove the judge is never consulted.
        rt.proactive_manager.on_response_delivered(
            conversation_id=CONVERSATION,
            person_id=person.person_id,
            source_event_id=event.event_id,
        )
    await asyncio.sleep(0.2)
    assert judge.payloads == []
    assert await rt.repos.proactive.list_for_conversation(CONVERSATION) == []


@pytest.mark.parametrize(
    "text",
    ["如果我走了呢？", "我不走", "他说他要下线了", "真的不聊了吗？", "我们聊点别的吧"],
)
def test_hypotheticals_quotes_and_questions_are_not_stops(text):
    """T06."""
    assert detect_explicit_stop(text) is False


async def test_delivered_web_reply_triggers_exactly_one_immediate_judge(chat):
    """T07: the page schedules the cycle once; long idle is still pending."""
    coordinator, rt, client, judge = await chat(
        decisions=[False], idle_delay=10.0
    )
    await _attach_tab(coordinator)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")
    await _wait_for(lambda: judge.payloads)
    assert [payload.mode for payload in judge.payloads] == [MODE_IMMEDIATE]
    await asyncio.sleep(0.05)
    assert [payload.mode for payload in judge.payloads] == [MODE_IMMEDIATE]


# -- T08-T12: delivery, delay, cancellation, truncation -------------------


async def test_wait_turns_never_schedule_a_cycle(chat):
    """A silent turn (WAIT/IGNORE) has no delivered reply to follow up on."""
    coordinator, rt, client, judge = await chat(
        decisions=[True, True], idle_delay=0.02
    )
    await _attach_tab(coordinator)
    person = await _owner(rt)

    _, events, _deltas, completed = await _web_turn(
        coordinator, rt, person, "嗯"
    )
    assert completed is None
    assert not [item for item in events if item.type == RESPONSE_COMPLETED]
    await asyncio.sleep(0.15)
    assert judge.payloads == []
    assert await rt.repos.proactive.list_for_conversation(CONVERSATION) == []


async def test_duplicate_completion_schedules_only_one_cycle(chat):
    """Repeating the same completion must not double-judge or double-send."""
    received, callback = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[True, False], idle_delay=10.0
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)
    event, _, _, completed = await _web_turn(
        coordinator, rt, person, "今天天气不错", schedule=False
    )
    assert completed is not None and completed.status == "COMPLETED"
    for _ in range(3):
        coordinator.on_normal_response_delivered(
            conversation_id=CONVERSATION,
            person_id=person.person_id,
            source_event_id=event.event_id,
        )

    await _wait_for(lambda: received)
    await asyncio.sleep(0.1)
    assert [payload.mode for payload in judge.payloads] == [MODE_IMMEDIATE]
    assert len(received) == 1
    assert len(await rt.repos.proactive.list_for_conversation(CONVERSATION)) == 1


async def test_immediate_yes_delivers_one_message_to_the_tab(chat):
    """T08."""
    received, callback = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[True, False], idle_delay=10.0
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")

    await _wait_for(lambda: received)
    assert len(received) == 1
    event = received[0]
    assert event.type == RESPONSE_COMPLETED
    assert event.conversation_id == CONVERSATION
    assert event.text == client.response_text
    rows = await rt.repos.proactive.list_for_conversation(CONVERSATION)
    assert len(rows) == 1
    assert rows[0].mode == MODE_IMMEDIATE
    assert rows[0].message_id == event.response_id
    assert judge.payloads and judge.payloads[0].mode == MODE_IMMEDIATE


async def test_immediate_no_delivers_nothing(chat):
    """T09."""
    received, callback = _collector()
    coordinator, rt, client, judge = await chat(decisions=[False])
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")
    await _wait_for(lambda: judge.payloads)
    await asyncio.sleep(0.1)
    assert received == []
    assert await rt.repos.proactive.list_for_conversation(CONVERSATION) == []


async def test_long_idle_runs_after_the_window_and_delivers_once(chat):
    """T10."""
    received, callback = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[False, True], idle_delay=0.02
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")

    await _wait_for(
        lambda: [p.mode for p in judge.payloads] == [MODE_IMMEDIATE, MODE_LONG_IDLE]
    )
    await _wait_for(lambda: received)
    rows = await rt.repos.proactive.list_for_conversation(CONVERSATION)
    assert [row.mode for row in rows] == [MODE_LONG_IDLE]
    assert len(received) == 1
    await asyncio.sleep(0.1)
    assert len(await rt.repos.proactive.list_for_conversation(CONVERSATION)) == 1


async def test_new_user_input_cancels_the_pending_idle_stage(chat):
    """T11."""
    coordinator, rt, client, judge = await chat(
        decisions=[False], idle_delay=10.0
    )
    await _attach_tab(coordinator)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")
    await _wait_for(lambda: judge.payloads)
    assert [payload.mode for payload in judge.payloads] == [MODE_IMMEDIATE]

    coordinator.note_user_input(CONVERSATION, "new-event")
    assert rt.proactive_manager._states[CONVERSATION].task is None
    await asyncio.sleep(0.1)
    assert [payload.mode for payload in judge.payloads] == [MODE_IMMEDIATE]
    assert await rt.repos.proactive.list_for_conversation(CONVERSATION) == []


async def test_stale_completion_cannot_schedule_a_newer_conversation_state(chat):
    """A slower earlier turn must not start a cycle after a newer input."""
    coordinator, rt, client, judge = await chat(
        decisions=[True, True], idle_delay=10.0
    )
    await _attach_tab(coordinator)
    person = await _owner(rt)
    first, _, _, _ = await _web_turn(
        coordinator, rt, person, "第一句", schedule=False
    )
    await _web_turn(coordinator, rt, person, "第二句", schedule=False)

    coordinator.on_normal_response_delivered(
        conversation_id=CONVERSATION,
        person_id=person.person_id,
        source_event_id=first.event_id,
    )
    await asyncio.sleep(0.1)
    assert judge.payloads == []
    state = rt.proactive_manager._states[CONVERSATION]
    assert state.task is None
    assert state.latest_user_event_id != first.event_id


async def test_truncated_proactive_message_is_not_delivered(chat, monkeypatch):
    """T12."""
    received, callback = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[True], idle_delay=10.0
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)

    async def truncated(*, source_event, person_id, mode="IMMEDIATE"):
        return {"text": "半句话", "finish_reason": "length", "model": "fake"}

    monkeypatch.setattr(rt.orchestrator, "generate_proactive_reply", truncated)
    await _web_turn(coordinator, rt, person, "今天天气不错")
    await _wait_for(lambda: judge.payloads)
    await asyncio.sleep(0.1)
    assert received == []
    assert await rt.repos.proactive.list_for_conversation(CONVERSATION) == []


# -- T13: history recovery ------------------------------------------------


async def test_history_contains_proactive_rows_once_in_order(chat):
    """T13."""
    coordinator, rt, client, judge = await chat(
        decisions=[True, False], idle_delay=10.0
    )
    received, callback = _collector()
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")
    await _wait_for(lambda: received)
    rows = await rt.repos.proactive.list_for_conversation(CONVERSATION)
    assert len(rows) == 1

    entries = await load_chat_history(rt, CONVERSATION)
    kinds = [entry["kind"] for entry in entries]
    assert kinds == ["user", "ava", "ava"]
    assert entries[0]["text"] == "今天天气不错"
    assert entries[2]["proactive"] is True
    assert entries[2]["message_id"] == rows[0].message_id
    assert entries[2]["text"] == received[0].text
    # Loading twice is idempotent: the same rows are never duplicated.
    again = await load_chat_history(rt, CONVERSATION)
    assert [entry["message_id"] for entry in again] == [
        entry["message_id"] for entry in entries
    ]


# -- T14-T17: multiple tabs and conversations -----------------------------


async def test_two_tabs_share_one_generation_and_both_see_it(chat):
    """T14."""
    first, first_cb = _collector()
    second, second_cb = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[True, False], idle_delay=10.0
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", first_cb)
    await coordinator.subscribe(CONVERSATION, "tab-2", second_cb)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")

    await _wait_for(lambda: first and second)
    assert len(first) == 1 and len(second) == 1
    assert first[0].response_id == second[0].response_id
    assert len(await rt.repos.proactive.list_for_conversation(CONVERSATION)) == 1
    assert [payload.mode for payload in judge.payloads] == [MODE_IMMEDIATE]


async def test_conversations_are_isolated(chat):
    """T15."""
    coordinator, rt, client, judge = await chat(decisions=[], idle_delay=10.0)
    person = await _owner(rt)
    await rt.conversations.get_or_create("chat:a")
    await rt.conversations.get_or_create("chat:b")
    event_a, _, _, _ = await _web_turn(
        coordinator, rt, person, "A 的话", conversation="chat:a", schedule=False
    )
    event_b, _, _, _ = await _web_turn(
        coordinator, rt, person, "B 的话", conversation="chat:b", schedule=False
    )

    manager = rt.proactive_manager
    manager.on_response_delivered(
        conversation_id="chat:b",
        person_id=person.person_id,
        source_event_id=event_b.event_id,
    )
    manager.note_user_input("chat:a", event_a.event_id)
    # The manager's state for B is untouched by A's new turn.
    state_b = manager._states["chat:b"]
    assert state_b.source_event_id == event_b.event_id
    assert state_b.latest_user_event_id == event_b.event_id

    # Cancelling A's cycle (last tab closes) also leaves B intact.
    manager.cancel_cycle("chat:a")
    assert manager._states["chat:b"].source_event_id == event_b.event_id
    assert manager._states["chat:b"].latest_user_event_id == event_b.event_id


async def test_closing_one_tab_leaves_the_other_subscribed(chat):
    """T16."""
    received, callback = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[True, False], idle_delay=10.0
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    await coordinator.subscribe(CONVERSATION, "tab-2", callback)
    await coordinator.unsubscribe(CONVERSATION, "tab-1")
    assert coordinator.subscriber_count(CONVERSATION) == 1
    assert coordinator.has_subscribers(CONVERSATION) is True

    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")
    await _wait_for(lambda: received)
    assert len(await rt.repos.proactive.list_for_conversation(CONVERSATION)) == 1


async def test_closing_the_last_tab_cancels_the_cycle(chat):
    """T17."""
    _, callback = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[False], idle_delay=10.0
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")
    await _wait_for(lambda: judge.payloads)
    assert rt.proactive_manager._states[CONVERSATION].task is not None

    await coordinator.unsubscribe(CONVERSATION, "tab-1")
    assert coordinator.subscriber_count(CONVERSATION) == 0
    assert rt.proactive_manager._states[CONVERSATION].task is None
    await asyncio.sleep(0.1)
    assert [payload.mode for payload in judge.payloads] == [MODE_IMMEDIATE]


# -- T18-T21: permissions, shutdown, mock ---------------------------------


async def test_quiet_mode_blocks_generation(chat):
    """T18."""
    coordinator, rt, client, judge = await chat(decisions=[True, True])
    await _attach_tab(coordinator)
    person = await _owner(rt)
    event, _, _, _ = await _web_turn(
        coordinator, rt, person, "今天天气不错", schedule=False
    )
    await rt.conversations.set_quiet_mode(CONVERSATION, True)
    rt.proactive_manager.on_response_delivered(
        conversation_id=CONVERSATION,
        person_id=person.person_id,
        source_event_id=event.event_id,
    )
    await asyncio.sleep(0.15)
    assert judge.payloads == []
    assert await rt.repos.proactive.list_for_conversation(CONVERSATION) == []


async def test_revoked_permission_blocks_generation(chat):
    """T19."""
    coordinator, rt, client, judge = await chat(decisions=[True, True])
    await _attach_tab(coordinator)
    person = await _owner(rt)
    event, _, _, _ = await _web_turn(
        coordinator, rt, person, "今天天气不错", schedule=False
    )
    consent = await rt.registry.get_consent(person.person_id)
    await rt.registry.set_consent(
        consent.model_copy(update={"allow_external_ai": False})
    )
    rt.proactive_manager.on_response_delivered(
        conversation_id=CONVERSATION,
        person_id=person.person_id,
        source_event_id=event.event_id,
    )
    await asyncio.sleep(0.15)
    assert judge.payloads == []
    assert await rt.repos.proactive.list_for_conversation(CONVERSATION) == []


async def test_shutdown_releases_tasks_and_the_shared_database(chat):
    """T20."""
    _, callback = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[False], idle_delay=10.0
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")
    await _wait_for(lambda: judge.payloads)
    assert coordinator._dispatcher is not None

    await coordinator.close()
    assert coordinator.started is False
    assert coordinator._dispatcher is None
    assert rt.proactive_manager._closed is True
    assert all(
        state.task is None for state in rt.proactive_manager._states.values()
    )
    with pytest.raises(RuntimeError):
        await rt.repos.utterance.recent_context(CONVERSATION, None, limit=1)


async def test_mock_provider_has_no_proactive_manager_and_no_extra_call(chat):
    """T21."""
    coordinator, rt, client, judge = await chat(provider="mock")
    assert rt.provider == "mock"
    assert rt.proactive_manager is None
    assert rt.profile_confirmation is None

    person = await rt.registry.register(
        "Mock",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=False,
    )
    await _web_turn(coordinator, rt, person, "你好")
    await asyncio.sleep(0.1)
    assert judge.payloads == []
    assert await rt.repos.proactive.list_for_conversation(CONVERSATION) == []
    assert "response" not in client.calls


# -- T22-T25: lifecycle edges ---------------------------------------------


async def test_proactive_delivery_never_schedules_another_cycle(chat):
    """T22: two stages, then the cycle stops. Nothing self-triggers."""
    received, callback = _collector()
    coordinator, rt, client, judge = await chat(
        decisions=[True, True], idle_delay=0.02
    )
    await coordinator.subscribe(CONVERSATION, "tab-1", callback)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "今天天气不错")

    await _wait_for(
        lambda: len(received) == 2
        and [p.mode for p in judge.payloads] == [MODE_IMMEDIATE, MODE_LONG_IDLE]
    )
    await asyncio.sleep(0.15)
    assert [p.mode for p in judge.payloads] == [MODE_IMMEDIATE, MODE_LONG_IDLE]
    rows = await rt.repos.proactive.list_for_conversation(CONVERSATION)
    assert [row.mode for row in rows] == [MODE_IMMEDIATE, MODE_LONG_IDLE]


async def test_pending_confirmation_blocks_only_the_immediate_stage(chat):
    """T23."""
    coordinator, rt, client, judge = await chat(
        decisions=[False], idle_delay=0.02
    )
    await _attach_tab(coordinator)
    person = await _owner(rt)
    await _web_turn(coordinator, rt, person, "我叫李约")
    await _wait_for(lambda: judge.payloads)
    assert [payload.mode for payload in judge.payloads] == [MODE_LONG_IDLE]

    # Answering the question resolves it; no duplicate confirmation appears.
    _, _, deltas, _ = await _web_turn(coordinator, rt, person, "对")
    assert "确认一下" not in deltas
    assert [entry["content"] for entry in await _name_entries(rt, person)] == [
        "李约"
    ]


async def test_restart_resumes_the_timer_without_a_duplicate_question(chat):
    """T24."""
    first, rt, client, _ = await chat(
        confirmation_timeout=0.05, decisions=[], idle_delay=10.0
    )
    person = await _owner(rt)
    await _web_turn(first, rt, person, "我叫李约")
    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_AWAITING_REPLY
    assert await _name_entries(rt, person) == []
    await first.close()

    second, rt2, _, _ = await chat(
        confirmation_timeout=0.05, decisions=[], idle_delay=10.0
    )
    assert rt2.profile_confirmation.resume_calls == 1
    await asyncio.sleep(0.2)
    assert [entry["content"] for entry in await _name_entries(rt2, person)] == [
        "李约"
    ]
    records = await rt2.repos.profile_confirmation.list_for_person(
        person.person_id
    )
    assert len(records) == 1
    assert records[0].state == STATE_ACCEPTED_SELF_REPORT


async def test_start_is_idempotent_and_resumes_pending_only_once(chat):
    """T24 (duplicate timers): a second start must not resume again."""
    coordinator, rt, client, _ = await chat(
        confirmation_timeout=60.0, decisions=[], idle_delay=10.0
    )
    confirmation = rt.profile_confirmation
    assert confirmation.resume_calls == 1
    assert await coordinator.start() is rt
    assert confirmation.resume_calls == 1


async def test_input_in_one_conversation_keeps_the_other_queued_event(chat):
    """T25: ``_discard_pending_events`` is conversation-scoped."""
    coordinator, rt, client, judge = await chat(decisions=[], idle_delay=10.0)
    manager = rt.proactive_manager
    other = ResponseEvent(
        type=RESPONSE_COMPLETED,
        event_id="event-b",
        conversation_id="chat:b",
        response_id="msg-b",
        sequence=1,
        status="COMPLETED",
        text="B 的主动消息",
        report_status="COMPLETED",
        action="RESPOND",
    )
    manager._emit(other)
    manager.note_user_input("chat:a", "event-a-new")
    kept = manager.poll_event()
    assert kept is not None
    assert kept.conversation_id == "chat:b"
    assert kept.text == "B 的主动消息"
