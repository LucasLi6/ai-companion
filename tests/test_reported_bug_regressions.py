"""Regression tests for the bugs reported against the real chat pipeline.

Everything here is offline: the DeepSeek transport is a scripted fake that
records which request types were made. No API key, no network.

The reported failures were:

- ``跳过这个话`` produced a confirmation reply ("没问题，翻篇了。").
- ``不回答我说的话`` produced a confirmation reply ("嗯，那就不回。").

Both must end as ``IGNORE`` with no provider call, no stream event, no saved
reply text, and no conversational output.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    MemoryKind,
    MemoryVisibility,
    ProcessingStatus,
    ReasonCode,
)
from companion.domain.models import make_utterance_event
from companion.memory.models import AccessContext
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime
from companion.storage.database import Database

ANALYSIS_JSON = json.dumps(
    {
        "meaning": "用户在表达",
        "action": "RESPOND",
        "reason_code": "direct_agent_address",
        "referenced_memory_ids": [],
        "memory_candidates": [],
        "response_style": "warm",
        "reply_necessity": "USEFUL",
    },
    ensure_ascii=False,
)


def _chat(content: str) -> ChatResponse:
    return ChatResponse(
        content=content,
        model="deepseek-flash",
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )


class RecordingClient:
    """Records every request type; returns a valid analysis or a short reply."""

    def __init__(
        self, analysis: str = ANALYSIS_JSON, response_text: str = "好的，我来回答。"
    ) -> None:
        self.calls: list[str] = []
        self.requests: list[tuple[str, list]] = []
        self.analysis = analysis
        self.response_text = response_text

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(request_type)
        self.requests.append((request_type, messages))
        if request_type == "analysis":
            return _chat(self.analysis)
        return _chat(self.response_text)

    def rendered_requests(self, request_type: str | None = None) -> str:
        return json.dumps(
            [
                messages
                for kind, messages in self.requests
                if request_type is None or kind == request_type
            ],
            ensure_ascii=False,
        )


async def _runtime(
    tmp_path,
    name: str = "Speaker",
    *,
    analysis=ANALYSIS_JSON,
    response_text="好的，我来回答。",
):
    db = await Database(tmp_path / "regressions.db").connect()
    await db.initialize()
    client = RecordingClient(analysis, response_text)
    rt = build_runtime(db, provider="deepseek", deepseek_client=client, router_mode="legacy")
    person = await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    return db, rt, person, client


def _event(person, text, *, to="agent", conversation="c1", occurred_at=None):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint=to,
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
        occurred_at=occurred_at,
    )


async def _turn(
    rt, person, text, *, to="agent", conversation="c1", occurred_at=None
):
    event = _event(
        person, text, to=to, conversation=conversation, occurred_at=occurred_at
    )
    report = await rt.orchestrator.process_utterance(event)
    action = await rt.repos.analysis.get_action(event.event_id)
    return event, report, action


# ---------------------------------------------------------------------------
# The two reported cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_bu_negation_is_silent(tmp_path):
    """``不回答我说的话`` must not be answered, not even with a confirmation."""
    db, rt, person, client = await _runtime(tmp_path)
    try:
        event, report, action = await _turn(rt, person, "不回答我说的话")
        assert report.action == CognitiveAction.IGNORE
        assert report.response_text is None
        assert client.calls == []  # no analysis, no response
        assert action is not None
        assert action.action_type == CognitiveAction.IGNORE
        assert action.reason_code == ReasonCode.EXPLICIT_SILENCE_REQUEST
        assert action.response_text is None
        trace = action.nonresponse_trace
        assert trace["final_action"] == "IGNORE"
        assert trace["cause_code"] == "EXPLICIT_SILENCE"
        assert trace["cause_source"] == "LOCAL_POLICY"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_skip_current_message_is_silent(tmp_path):
    """``跳过这个话`` must skip the current message, with no confirmation."""
    db, rt, person, client = await _runtime(tmp_path)
    try:
        event, report, action = await _turn(rt, person, "跳过这个话")
        assert report.action == CognitiveAction.IGNORE
        assert report.response_text is None
        assert client.calls == []
        assert action is not None and action.action_type == CognitiveAction.IGNORE
        assert action.response_text is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_skip_paragraph_is_silent(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        report = (await _turn(rt, person, "这段先略过"))[1]
        assert report.action == CognitiveAction.IGNORE
        assert client.calls == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_this_message_needs_no_reply_is_silent(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        report = (await _turn(rt, person, "这句话不用回复"))[1]
        assert report.action == CognitiveAction.IGNORE
        assert report.response_text is None
        assert client.calls == []
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# False-positive guards: normal turns must still be processed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "为什么 AI 有时候不回答？",
        "我朋友刚才说“不要回答我”。",
        "我朋友说“别回复我”",
        "如果我让你不回答，你会怎么处理？",
        "跳过刚才的话题，现在帮我解释 Python。",
    ],
)
async def test_silence_lookalikes_are_still_answered(tmp_path, text):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        report = (await _turn(rt, person, text))[1]
        assert report.action == CognitiveAction.RESPOND
        assert report.response_text is not None
        assert "response" in client.calls
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_greeting_still_works_after_a_silent_turn(tmp_path):
    """One explicit-silence request must not become a permanent quiet mode."""
    db, rt, person, client = await _runtime(tmp_path)
    try:
        first = (await _turn(rt, person, "不回答我说的话"))[1]
        assert first.action == CognitiveAction.IGNORE
        assert client.calls == []

        second = (await _turn(rt, person, "你好"))[1]
        assert second.action == CognitiveAction.RESPOND
        assert second.response_text is not None
        assert client.calls == ["response"]
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# WAIT semantics and streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_request_is_silent_and_has_no_stream_events(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        event = _event(person, "等我一下")
        items = [
            item async for item in rt.orchestrator.process_utterance_stream(event)
        ]
        assert items == []  # no response_started/delta/completed
        assert client.calls == []
        action = await rt.repos.analysis.get_action(event.event_id)
        assert action is not None
        assert action.action_type == CognitiveAction.WAIT
        assert action.response_text is None
        assert action.nonresponse_trace["pending_followup"] is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_silent_turn_emits_no_stream_events(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        event = _event(person, "不回答我说的话")
        items = [
            item async for item in rt.orchestrator.process_utterance_stream(event)
        ]
        assert items == []
        assert client.calls == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_wait_phrase_with_a_question_is_processed(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        report = (await _turn(rt, person, "等一下，这个怎么弄？"))[1]
        assert report.action == CognitiveAction.RESPOND
        assert report.response_text is not None
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Addressee boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_addressee_is_ignored_without_any_model_call(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        report = (await _turn(rt, person, "我昨天完成了一个项目。", to="human"))[1]
        assert report.action == CognitiveAction.IGNORE
        assert report.response_text is None
        assert client.calls == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unknown_addressee_uses_deliberation_and_conservatively_waits(tmp_path):
    # An unclear addressee goes to the existing Deliberation Path. The model is
    # allowed to suggest, but a merely OPTIONAL, unaddressed reply does not
    # clear the conservative bar, so the turn stays WAIT.
    optional = json.dumps(
        {
            "meaning": "可能是自言自语",
            "action": "RESPOND",
            "reason_code": "ambiguous_addressee",
            "referenced_memory_ids": [],
            "memory_candidates": [],
            "response_style": "warm",
            "reply_necessity": "OPTIONAL",
            "addressed_to": "unknown",
        },
        ensure_ascii=False,
    )
    db, rt, person, client = await _runtime(tmp_path, analysis=optional)
    try:
        report = (await _turn(rt, person, "这个我觉得可以。", to="unknown"))[1]
        assert report.action == CognitiveAction.WAIT
        assert report.response_text is None
        assert client.calls == ["analysis"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_backchannel_pause_does_not_force_a_reply(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        for text in ("嗯", "...", "……"):
            report = (await _turn(rt, person, text))[1]
            assert report.action == CognitiveAction.WAIT, text
            assert report.response_text is None, text
        assert client.calls == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_filler_inside_a_question_is_still_answered(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        report = (await _turn(rt, person, "嗯，那你觉得呢？"))[1]
        assert report.action == CognitiveAction.RESPOND
        assert report.response_text is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_thinking_pause_is_not_treated_as_must_reply(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        for text in ("我再想想", "让我想想"):
            report = (await _turn(rt, person, text))[1]
            assert report.action == CognitiveAction.WAIT, text
            assert report.response_text is None, text
        assert client.calls == []

        # The pause is per-turn: a later greeting still gets a reply.
        later = (await _turn(rt, person, "你好"))[1]
        assert later.action == CognitiveAction.RESPOND
        assert later.response_text is not None
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Memory recall and truthfulness
# ---------------------------------------------------------------------------


async def _remember(rt, person, content):
    return await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content=content,
        access=AccessContext("admin", is_admin=True),
        status="ACTIVE",
        visibility=MemoryVisibility.PRIVATE,
        subjects=[("PERSON", person.person_id)],
    )


@pytest.mark.asyncio
async def test_memory_question_uses_a_real_stored_memory(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        await _remember(rt, person, "用户正在推进一个机器学习项目，当前进入部署阶段。")
        report = (await _turn(rt, person, "你还记得我的项目吗？"))[1]
        assert report.action == CognitiveAction.RESPOND
        rendered = client.rendered_requests("response")
        assert "机器学习项目" in rendered
        assert "找到相关记忆" in rendered
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_another_persons_private_memory_never_reaches_the_prompt(tmp_path):
    db, rt, alice, client = await _runtime(tmp_path)
    try:
        bob = await rt.registry.register(
            "Bob",
            "friend",
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=True,
            allow_external_ai=True,
        )
        await _remember(rt, bob, "Bob 的银行卡密码是 123456。")
        report = (await _turn(rt, alice, "你还记得我的项目吗？"))[1]
        assert report.action == CognitiveAction.RESPOND
        assert "银行卡密码" not in client.rendered_requests()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_pronoun_follow_up_resolves_through_conversation_history(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        (await _turn(rt, person, "我最近在推进一个机器学习项目。"))[1]
        report = (await _turn(rt, person, "那个现在怎么样了？"))[1]
        assert report.action == CognitiveAction.RESPOND
        assert "机器学习项目" in client.rendered_requests("response")
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Time context
# ---------------------------------------------------------------------------


def _system_prompt(client) -> str:
    response_calls = [
        messages for kind, messages in client.requests if kind == "response"
    ]
    return response_calls[-1][0]["content"]


@pytest.mark.asyncio
async def test_time_question_receives_trusted_runtime_data(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        await rt.repos.settings.set(
            f"runtime.timezone.person.{person.person_id}", "Asia/Shanghai"
        )
        (await _turn(rt, person, "现在几点？"))[1]
        system = _system_prompt(client)
        assert "【运行状态】" in system
        assert "time=" in system
        assert "Asia/Shanghai" in system
        # Runtime data is trusted system context, never a user utterance.
        user_messages = [
            m for m in client.requests[-1][1] if m.get("role") == "user"
        ]
        assert all("【运行状态】" not in m.get("content", "") for m in user_messages)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_how_long_since_we_talked_uses_the_last_real_reply(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    start = datetime(2026, 11, 4, 2, 0, tzinfo=timezone.utc)
    try:
        (await _turn(rt, person, "你好", occurred_at=start))[1]
        report = (
            await _turn(
                rt,
                person,
                "我们多久没聊天了？",
                occurred_at=start + timedelta(hours=2, minutes=30),
            )
        )[1]
        assert report.action == CognitiveAction.RESPOND
        system = _system_prompt(client)
        assert "last_interaction=" in system
        assert "2 小时 30 分钟前" in system
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_recent_reply_is_available_to_a_follow_up_question(tmp_path):
    db, rt, person, client = await _runtime(
        tmp_path, response_text="我建议先做数据清洗。"
    )
    start = datetime(2026, 11, 4, 2, 0, tzinfo=timezone.utc)
    try:
        (await _turn(rt, person, "这个数据要怎么处理？", occurred_at=start))[1]
        (await _turn(
            rt, person, "你刚才说什么？", occurred_at=start + timedelta(minutes=1)
        ))[1]
        rendered = client.rendered_requests("response")
        assert "我建议先做数据清洗。" in rendered
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Conversation counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_count_matches_the_real_sqlite_records(tmp_path):
    db, rt, person, client = await _runtime(tmp_path)
    start = datetime(2026, 11, 4, 2, 0, tzinfo=timezone.utc)
    try:
        for index in range(3):
            (await _turn(rt, person, f"你好 {index}", occurred_at=start))[1]
        (await _turn(rt, person, "换一个话题", conversation="c2"))[1]

        bob = await rt.registry.register(
            "Bob",
            "friend",
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=True,
            allow_external_ai=True,
        )
        await rt.orchestrator.process_utterance(
            _event(bob, "我也在这里", conversation="c1")
        )

        report = (await _turn(rt, person, "到现在你和我有多少条对话了？"))[1]
        assert report.action == CognitiveAction.RESPOND
        system = _system_prompt(client)

        cur = await db.execute(
            "SELECT COUNT(*) AS messages,"
            " COUNT(DISTINCT conversation_id) AS sessions"
            " FROM utterances WHERE speaker_person_id = ?",
            (person.person_id,),
        )
        row = await cur.fetchone()
        cur = await db.execute(
            "SELECT COUNT(*) AS replies FROM agent_actions a"
            " JOIN utterances u ON u.event_id = a.event_id"
            " WHERE u.speaker_person_id = ?"
            " AND a.action_type = 'RESPOND'"
            " AND a.status IN ('COMPLETED', 'TRUNCATED')",
            (person.person_id,),
        )
        replies = (await cur.fetchone())["replies"]

        # The count question itself is an utterance, but its reply is only
        # generated after the stats are read.
        assert f"对方发来的消息={row['messages']} 条" in system
        assert f"Ava 可见回复={replies - 1} 条" in system
        assert f"会话数={row['sessions']} 个" in system
        assert "不得估算" in system
        # Bob's message and reply are not attributed to this speaker.
        assert row["messages"] == 5
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_topical_count_question_does_not_trigger_conversation_stats(
    tmp_path,
):
    db, rt, person, client = await _runtime(tmp_path)
    try:
        (await _turn(rt, person, "这段对话有多少条建议？"))[1]
        assert "【对话统计】" not in _system_prompt(client)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Persistence and retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persisted_wait_is_not_regenerated_on_retry(tmp_path):
    """A durable WAIT must survive a replay as WAIT, never become a reply."""
    db, rt, person, client = await _runtime(tmp_path)
    try:
        event = _event(person, "等我一下")
        first = await rt.orchestrator.process_utterance(event)
        assert first.action == CognitiveAction.WAIT
        assert client.calls == []

        # Simulate a crash after the event was stored but before completion.
        await rt.repos.utterance.update_status(
            event.event_id, ProcessingStatus.PROCESSING
        )
        retry = make_utterance_event(
            conversation_id="c1",
            text="等我一下",
            addressee_hint="agent",
            speaker_person_id=person.person_id,
            source=EventSource.TEST,
            event_id=event.event_id,
        )
        second = await rt.orchestrator.process_utterance(retry)
        assert second.action == CognitiveAction.WAIT
        assert second.response_text is None
        assert client.calls == []

        action = await rt.repos.analysis.get_action(event.event_id)
        assert action is not None
        assert action.action_type == CognitiveAction.WAIT
        assert action.response_text is None
        cur = await db.execute("SELECT COUNT(*) AS c FROM agent_actions")
        assert (await cur.fetchone())["c"] == 1
    finally:
        await db.close()
