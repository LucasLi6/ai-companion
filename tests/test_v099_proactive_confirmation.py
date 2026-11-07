"""V0.9.9 key checks: direct self-reports and proactive speech boundaries.

Offline and deliberately small. V0.9.9.6 removed the one-turn confirmation
question: an explicit first-person statement is written immediately by local
code. These tests pin that narrow detection and the proactive boundaries that
must never depend on a "yes" reply.
"""

from __future__ import annotations

import asyncio
import json

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_confirmation import ProfileDirectSelfReportService
from companion.people.profile_self_report import detect_self_reports
from companion.proactive.departure import (
    detect_explicit_departure,
    detect_explicit_stop,
)
from companion.proactive.judge import ProactiveJudge
from companion.providers.deepseek_client import ChatResponse, StreamDelta
from companion.providers.deepseek_proactive import ScriptedProactiveJudgeProvider
from companion.runtime import build_runtime


class _FakeClient:
    """Offline client: judge JSON for analysis calls, text for replies."""

    def __init__(self, *, response_text="好的。", chunks=None):
        self.response_text = response_text
        self.chunks = list(chunks) if chunks is not None else [response_text]

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        content = (
            self.response_text
            if request_type == "response"
            else json.dumps(
                {"should_initiate": False, "reason": "scripted"}, ensure_ascii=False
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

    async def stream_completion(
        self, messages, *, request_type="response", **kwargs
    ):
        for chunk in self.chunks:
            yield StreamDelta(content=chunk)
        yield StreamDelta(
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            usage_reported=True,
        )


async def _person(rt, name="Confirm"):
    return await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


async def _event(rt, person, text, conversation="c1", *, save=True):
    await rt.conversations.get_or_create(conversation)
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    if save:
        await rt.repos.utterance.save(event)
    return event


def _service(rt):
    return ProfileDirectSelfReportService(
        profile_service=rt.profile_service,
        repos=rt.repos,
    )


async def _active_name(rt, person):
    return await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="identity.name"
    )


# -- detection -------------------------------------------------------------


def test_explicit_name_fields_are_detected_together():
    detected = {
        item.field_key: item.value for item in detect_self_reports("我叫李约，英文名 Lucas")
    }
    assert detected == {"identity.name": "李约", "identity.preferred_name": "Lucas"}


def test_future_self_report_keeps_only_the_name():
    detected = {
        item.field_key: item.value
        for item in detect_self_reports("我叫李约以后你可以叫我约约")
    }
    assert detected["identity.name"] == "李约"
    assert detected["identity.preferred_name"] == "约约"


def test_questions_hypotheticals_quotes_and_third_person_are_refused():
    for text in (
        "李约是谁？",
        "假如我叫李约",
        "他说他叫李约",
        "他叫我小明",
        "你们可以叫我小明吗？",
    ):
        assert detect_self_reports(text) == (), text


# -- direct write on the same turn ----------------------------------------


async def test_explicit_name_is_written_once_with_the_real_source(rt):
    person = await _person(rt)
    service = _service(rt)
    first = await _event(rt, person, "我叫李约")

    assert await service.begin_turn(event=first, person_id=person.person_id) is None
    entries = await _active_name(rt, person)
    assert [entry["content"] for entry in entries] == ["李约"]
    assert entries[0]["evidence_event_ids"] == [first.event_id]
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 2
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert revisions[-1]["changes"]["source"] == "DIRECT_SELF_REPORT"
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []

    # A later turn never writes the same name again.
    third = await _event(rt, person, "嗯，今天好热")
    await service.begin_turn(event=third, person_id=person.person_id)
    assert len(await _active_name(rt, person)) == 1
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == 2


async def test_denial_text_alone_writes_nothing(rt):
    person = await _person(rt)
    service = _service(rt)
    first = await _event(rt, person, "我叫李约")
    await service.begin_turn(event=first, person_id=person.person_id)
    second = await _event(rt, person, "不是")
    await service.begin_turn(event=second, person_id=person.person_id)
    # "不是" denies nothing because there is no open question; the only
    # evidence in the conversation is the person's own explicit statement.
    assert [entry["content"] for entry in await _active_name(rt, person)] == [
        "李约"
    ]


async def test_new_explicit_name_replaces_the_old_one(rt):
    person = await _person(rt)
    service = _service(rt)
    first = await _event(rt, person, "我叫李约")
    await service.begin_turn(event=first, person_id=person.person_id)

    second = await _event(rt, person, "我叫李明")
    await service.begin_turn(event=second, person_id=person.person_id)
    active = await _active_name(rt, person)
    assert [entry["content"] for entry in active] == ["李明"]
    # The correction updates the one identity entry; history keeps the old
    # value in the revision log rather than as a second active fact.
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert revisions[-1]["changes"]["old"] == ["李约"]
    assert revisions[-1]["changes"]["new"] == "李明"
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []


async def test_unrelated_turn_does_not_touch_a_written_name(rt):
    person = await _person(rt)
    service = _service(rt)
    first = await _event(rt, person, "我叫李约")
    await service.begin_turn(event=first, person_id=person.person_id)
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision

    second = await _event(rt, person, "今天天气不错")
    await service.begin_turn(event=second, person_id=person.person_id)
    assert [entry["content"] for entry in await _active_name(rt, person)] == [
        "李约"
    ]
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == revision


# -- interaction with proactive speech ------------------------------------


def test_clear_departure_is_structural_not_keyword_matching():
    for text in ("我要下线了", "我先睡了，晚安", "今天就聊到这儿吧，拜拜"):
        assert detect_explicit_departure(text) is True, text
    for text in (
        "真的不聊了？",
        "假如我走了呢",
        "他不走",
        "他说他走了",
        "我还没走",
        "今天好累啊",
    ):
        assert detect_explicit_departure(text) is False, text


def test_explicit_refusal_is_a_local_stop():
    for text in ("我不想聊了", "别跟我说话", "安静点", "闭嘴", "别烦我"):
        assert detect_explicit_stop(text) is True, text
    for text in ("真的不聊了？", "假如我不想聊了呢", "我们聊点别的吧"):
        assert detect_explicit_stop(text) is False, text


async def test_long_idle_judge_context_carries_departure_and_mode(db):
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=_FakeClient(),
        with_worker=False,
        router_mode="legacy",
    )
    person = await _person(rt)
    event = await _event(rt, person, "我要下线了", save=False)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"

    immediate = await rt.orchestrator.build_proactive_judge_context(
        source_event=event, person_id=person.person_id, mode="IMMEDIATE"
    )
    long_idle = await rt.orchestrator.build_proactive_judge_context(
        source_event=event, person_id=person.person_id, mode="LONG_IDLE"
    )
    assert immediate["mode"] == "IMMEDIATE"
    assert immediate["explicit_departure"] is False
    assert long_idle["mode"] == "LONG_IDLE"
    assert long_idle["explicit_departure"] is True


async def test_direct_self_report_does_not_add_a_question_to_the_reply(db):
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=_FakeClient(response_text="好的。"),
        with_worker=False,
        router_mode="legacy",
    )
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")
    event = await _event(rt, person, "我叫李约", save=False)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    assert "确认一下" not in (report.response_text or "")
    assert [entry["content"] for entry in await _active_name(rt, person)] == [
        "李约"
    ]


async def test_explicit_refusal_blocks_both_proactive_stages(db):
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=_FakeClient(),
        with_worker=False,
        router_mode="legacy",
    )
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")
    event = await _event(rt, person, "我不想聊了", save=False)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"

    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.01
    judge = ScriptedProactiveJudgeProvider([True, True])
    manager.judge = ProactiveJudge(judge)
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=event.event_id,
    )
    await asyncio.sleep(0.15)

    # The local floor blocks both stages; the judge is never even consulted.
    assert judge.payloads == []
    assert await rt.repos.proactive.get_for_source_event(event.event_id) is None
    await manager.shutdown()


# -- real streaming path ---------------------------------------------------


async def test_self_report_is_written_without_being_reprinted(db):
    client = _FakeClient(response_text="好的。", chunks=["好", "的。"])
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
    )
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")

    first = await _event(rt, person, "我叫李约", save=False)
    events = [
        item async for item in rt.orchestrator.process_utterance_stream(first)
    ]
    deltas = "".join(
        item.delta or "" for item in events if item.type == "response_delta"
    )
    completed = next(
        item for item in events if item.type == "response_completed"
    )
    assert "确认一下" not in deltas
    # What the chat surface showed, the terminal event and the persisted
    # action are the same full reply.
    assert deltas == completed.text
    action = await rt.repos.analysis.get_action(first.event_id)
    assert action.response_text == completed.text
    assert [entry["content"] for entry in await _active_name(rt, person)] == [
        "李约"
    ]

    second = await _event(rt, person, "对", save=False)
    events = [
        item async for item in rt.orchestrator.process_utterance_stream(second)
    ]
    deltas = "".join(
        item.delta or "" for item in events if item.type == "response_delta"
    )
    assert "确认一下" not in deltas
    assert len(await _active_name(rt, person)) == 1
