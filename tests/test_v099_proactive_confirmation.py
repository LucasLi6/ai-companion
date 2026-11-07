"""V0.9.9 key checks: dual-stage proactive speech and one-turn confirmation.

Offline and deliberately small. They pin the behaviours the feature must not
get wrong: identity self-reports are detected narrowly and confirmed in one
turn, silence never overwrites a confirmed identity or counts as a yes, and a
clear departure is passed to the long-idle judge while a question is not.
"""

from __future__ import annotations

import asyncio
import json

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event, utcnow
from companion.people.profile_confirmation import (    KIND_AMBIGUOUS,
    KIND_CONFIRM,
    KIND_CORRECT,
    KIND_DENY,
    KIND_IGNORED,
    STATE_ACCEPTED_SELF_REPORT,
    STATE_AWAITING_REPLY,
    STATE_CONFIRMED,
    STATE_NEEDS_REVIEW,
    STATE_PENDING_QUESTION,
    STATE_REJECTED,
    ProfileConfirmationService,
    build_confirmation_question,
    detect_self_reports,
    interpret_reply,
)
from companion.proactive.departure import (
    detect_explicit_departure,
    detect_explicit_stop,
)
from companion.proactive.judge import ProactiveJudge
from companion.providers.deepseek_client import ChatResponse, StreamDelta
from companion.providers.deepseek_proactive import ScriptedProactiveJudgeProvider
from companion.runtime import build_runtime
from companion.storage.repositories import ProfileConfirmationRecord


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


def _service(rt, *, no_reply_timeout_seconds: float = 60.0):
    return ProfileConfirmationService(
        profile_service=rt.profile_service,
        repos=rt.repos,
        no_reply_timeout_seconds=no_reply_timeout_seconds,
    )


def _record(field_key: str, value: str) -> ProfileConfirmationRecord:
    now = utcnow()
    return ProfileConfirmationRecord(
        confirmation_id="pcfm-test",
        person_id="p1",
        conversation_id="c1",
        source_event_id="e1",
        question_message_id=None,
        candidate_id=None,
        field_key=field_key,
        value=value,
        quote="",
        state=STATE_AWAITING_REPLY,
        created_at=now,
        updated_at=now,
    )


async def _active_name(rt, person):
    return await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="identity.name"
    )


# -- detection -------------------------------------------------------------


def test_explicit_name_fields_are_detected_and_merged_into_one_question():
    detected = {
        item.field_key: item.value for item in detect_self_reports("我叫李约，英文名 Lucas")
    }
    assert detected == {"identity.name": "李约", "identity.preferred_name": "Lucas"}

    records = [
        _record("identity.name", "李约"),
        _record("identity.preferred_name", "Lucas"),
    ]
    assert build_confirmation_question(records) == (
        "确认一下，你的中文名是李约，你的英文名是 Lucas，对吗？"
    )


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


def test_reply_interpretation_is_conservative():
    records = [_record("identity.name", "李约")]

    assert interpret_reply("对", records).kind == KIND_CONFIRM
    assert interpret_reply("我叫李约", records).kind == KIND_CONFIRM
    assert interpret_reply("不是", records).kind == KIND_DENY
    corrected = interpret_reply("我叫李明", records)
    assert corrected.kind == KIND_CORRECT
    assert corrected.detected[0].value == "李明"
    assert interpret_reply("李约是谁？", records).kind == KIND_AMBIGUOUS
    assert interpret_reply("今天天气不错", records).kind == KIND_IGNORED
    # A reply that merely starts with an affirmation token is not agreement.
    assert interpret_reply("嗯，先说别的", records).kind == KIND_AMBIGUOUS
    assert interpret_reply("对了，我们聊其他话题", records).kind == KIND_AMBIGUOUS
    assert interpret_reply("好的，谢谢", records).kind == KIND_IGNORED


# -- confirmation state machine -------------------------------------------


async def test_confirm_writes_once_with_the_real_source(rt):
    person = await _person(rt)
    service = _service(rt)
    first = await _event(rt, person, "我叫李约")

    question = await service.begin_turn(event=first, person_id=person.person_id)
    assert question == "确认一下，你的中文名是李约，对吗？"
    assert await _active_name(rt, person) == []
    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_PENDING_QUESTION

    await service.mark_question_delivered(
        person_id=person.person_id,
        conversation_id="c1",
        question_message_id="act-1",
    )
    assert (
        await rt.repos.profile_confirmation.get(record.confirmation_id)
    ).state == STATE_AWAITING_REPLY

    second = await _event(rt, person, "对")
    assert await service.begin_turn(event=second, person_id=person.person_id) is None

    entries = await _active_name(rt, person)
    assert [entry["content"] for entry in entries] == ["李约"]
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 2
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert revisions[-1]["changes"]["source"] == "DIRECT_SELF_REPORT"
    assert revisions[-1]["changes"]["confirmed"] is True
    assert (
        await rt.repos.profile_confirmation.get(record.confirmation_id)
    ).state == STATE_CONFIRMED

    # A later turn never writes the same name again.
    third = await _event(rt, person, "嗯，今天好热")
    await service.begin_turn(event=third, person_id=person.person_id)
    assert len(await _active_name(rt, person)) == 1
    await service.shutdown()


async def test_deny_rejects_without_a_write(rt):
    person = await _person(rt)
    service = _service(rt)
    first = await _event(rt, person, "我叫李约")
    await service.begin_turn(event=first, person_id=person.person_id)
    await service.mark_question_delivered(
        person_id=person.person_id, conversation_id="c1", question_message_id="act-1"
    )

    second = await _event(rt, person, "不是")
    assert await service.begin_turn(event=second, person_id=person.person_id) is None
    assert await _active_name(rt, person) == []
    records = await rt.repos.profile_confirmation.list_for_person(person.person_id)
    assert records[0].state == STATE_REJECTED
    await service.shutdown()


async def test_correction_replaces_the_old_question(rt):
    person = await _person(rt)
    service = _service(rt)
    first = await _event(rt, person, "我叫李约")
    await service.begin_turn(event=first, person_id=person.person_id)
    await service.mark_question_delivered(
        person_id=person.person_id, conversation_id="c1", question_message_id="act-1"
    )

    second = await _event(rt, person, "我叫李明")
    question = await service.begin_turn(event=second, person_id=person.person_id)
    assert question == "确认一下，你的中文名是李明，对吗？"
    assert await _active_name(rt, person) == []
    active = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert active.value == "李明"
    await service.mark_question_delivered(
        person_id=person.person_id, conversation_id="c1", question_message_id="act-2"
    )

    third = await _event(rt, person, "对")
    assert await service.begin_turn(event=third, person_id=person.person_id) is None
    assert [entry["content"] for entry in await _active_name(rt, person)] == ["李明"]
    await service.shutdown()


async def test_unrelated_reply_accepts_a_clear_self_report_without_confirmation(rt):
    person = await _person(rt)
    service = _service(rt)
    first = await _event(rt, person, "我叫李约")
    await service.begin_turn(event=first, person_id=person.person_id)
    await service.mark_question_delivered(
        person_id=person.person_id, conversation_id="c1", question_message_id="act-1"
    )

    second = await _event(rt, person, "今天天气不错")
    await service.begin_turn(event=second, person_id=person.person_id)
    assert [entry["content"] for entry in await _active_name(rt, person)] == ["李约"]
    record = (
        await rt.repos.profile_confirmation.list_for_person(person.person_id)
    )[0]
    assert record.state == STATE_ACCEPTED_SELF_REPORT
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert revisions[-1]["changes"]["source"] == "DIRECT_SELF_REPORT"
    assert revisions[-1]["changes"]["confirmed"] is False

    # The same value is never written twice.
    third = await _event(rt, person, "嗯")
    await service.begin_turn(event=third, person_id=person.person_id)
    assert len(await _active_name(rt, person)) == 1
    await service.shutdown()


async def test_silence_after_a_clear_self_report_still_writes(rt):
    person = await _person(rt)
    service = _service(rt, no_reply_timeout_seconds=0.02)
    first = await _event(rt, person, "我叫李约")
    await service.begin_turn(event=first, person_id=person.person_id)
    await service.mark_question_delivered(
        person_id=person.person_id, conversation_id="c1", question_message_id="act-1"
    )

    await asyncio.sleep(0.15)
    assert [entry["content"] for entry in await _active_name(rt, person)] == ["李约"]
    record = (
        await rt.repos.profile_confirmation.list_for_person(person.person_id)
    )[0]
    assert record.state == STATE_ACCEPTED_SELF_REPORT
    await service.shutdown()


async def test_silence_never_overwrites_a_confirmed_identity(rt):
    person = await _person(rt)
    service = _service(rt, no_reply_timeout_seconds=0.02)
    first = await _event(rt, person, "我叫李约")
    await service.begin_turn(event=first, person_id=person.person_id)
    await service.mark_question_delivered(
        person_id=person.person_id, conversation_id="c1", question_message_id="act-1"
    )
    confirmed = await _event(rt, person, "对")
    await service.begin_turn(event=confirmed, person_id=person.person_id)
    assert [entry["content"] for entry in await _active_name(rt, person)] == ["李约"]

    # A later, different self-report cannot replace it just because the
    # confirmation question went unanswered.
    second = await _event(rt, person, "我叫李明")
    question = await service.begin_turn(event=second, person_id=person.person_id)
    assert question == "确认一下，你的中文名是李明，对吗？"
    await service.mark_question_delivered(
        person_id=person.person_id, conversation_id="c1", question_message_id="act-2"
    )
    await asyncio.sleep(0.15)

    assert [entry["content"] for entry in await _active_name(rt, person)] == ["李约"]
    records = await rt.repos.profile_confirmation.list_for_person(person.person_id)
    assert records[-1].state == STATE_NEEDS_REVIEW
    await service.shutdown()


async def test_background_review_candidate_is_reused_not_duplicated(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我叫李约")
    # A background Profile Review proposed the same value first.
    shared = await rt.repos.profile.insert_candidate(
        person_id=person.person_id,
        field_key="identity.name",
        value="李约",
        source="INTERACTION_INFERENCE",
        confidence=0.8,
        evidence_memory_ids=[],
        evidence_event_ids=[event.event_id],
        evidence_candidate_ids=[],
        reason="background_review",
        operation="ADD",
        target_entry_ids=[],
        expected_revision=None,
        requires_confirmation=True,
        review_status="PENDING",
    )

    service = _service(rt)
    await service.begin_turn(event=event, person_id=person.person_id)
    pending = await rt.repos.profile.find_pending(person.person_id, "identity.name")
    assert [item["candidate_id"] for item in pending] == [shared]
    assert pending[0]["source"] == "DIRECT_SELF_REPORT"
    assert event.event_id in pending[0]["evidence_event_ids"]
    await service.shutdown()


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


async def test_immediate_judge_does_not_interject_while_confirming(db):
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=_FakeClient(),
        with_worker=False,
        router_mode="legacy",
    )
    person = await _person(rt)
    await rt.conversations.get_or_create("c1")
    event = await _event(rt, person, "我叫李约", save=False)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    assert "确认一下" in report.response_text
    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_AWAITING_REPLY

    manager = rt.proactive_manager
    manager.idle_delay_seconds = 0.01
    judge = ScriptedProactiveJudgeProvider([False])
    manager.judge = ProactiveJudge(judge)
    manager.on_response_delivered(
        conversation_id="c1",
        person_id=person.person_id,
        source_event_id=event.event_id,
    )
    await asyncio.sleep(0.15)

    # The immediate stage is skipped while the question is open; only the
    # long-idle stage is judged.
    assert [payload.mode for payload in judge.payloads] == ["LONG_IDLE"]
    await manager.shutdown()
    await rt.profile_confirmation.shutdown()


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
    await rt.profile_confirmation.shutdown()


async def test_confirmation_timeout_resumes_after_restart(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我叫李约")
    before = _service(rt)
    await before.begin_turn(event=first, person_id=person.person_id)
    await before.mark_question_delivered(
        person_id=person.person_id, conversation_id="c1", question_message_id="act-1"
    )
    # Simulate a process exit: the in-memory timer is gone.
    await before.shutdown()
    assert await _active_name(rt, person) == []

    after = _service(rt, no_reply_timeout_seconds=0.02)
    assert await after.resume_pending() == 1
    await asyncio.sleep(0.15)
    assert [entry["content"] for entry in await _active_name(rt, person)] == ["李约"]
    record = (
        await rt.repos.profile_confirmation.list_for_person(person.person_id)
    )[0]
    assert record.state == STATE_ACCEPTED_SELF_REPORT
    await after.shutdown()


# -- real streaming path ---------------------------------------------------


async def _stream_events(rt, event):
    return [
        item async for item in rt.orchestrator.process_utterance_stream(event)
    ]


async def test_confirmation_question_is_delivered_through_stream_deltas(db):
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
    events = await _stream_events(rt, first)
    deltas = "".join(
        item.delta or "" for item in events if item.type == "response_delta"
    )
    completed = next(
        item for item in events if item.type == "response_completed"
    )
    assert "确认一下，你的中文名是李约，对吗？" in deltas
    # What the chat surface showed, the terminal event and the persisted
    # action are the same full reply.
    assert deltas == completed.text
    action = await rt.repos.analysis.get_action(first.event_id)
    assert action.response_text == completed.text
    record = await rt.repos.profile_confirmation.get_active(person.person_id)
    assert record.state == STATE_AWAITING_REPLY
    assert record.question_message_id == action.action_id

    second = await _event(rt, person, "对", save=False)
    events = await _stream_events(rt, second)
    deltas = "".join(
        item.delta or "" for item in events if item.type == "response_delta"
    )
    completed = next(
        item for item in events if item.type == "response_completed"
    )
    assert "确认一下" not in deltas
    assert deltas == completed.text
    assert [entry["content"] for entry in await _active_name(rt, person)] == ["李约"]
    stored = await rt.repos.profile_confirmation.get(record.confirmation_id)
    assert stored.state == STATE_CONFIRMED
    await rt.profile_confirmation.shutdown()
