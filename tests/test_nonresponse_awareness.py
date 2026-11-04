"""V0.3.2 — the companion can see its own past silence truthfully.

These tests cover the durable action trace, the separate action-history channel
(never fake assistant text), prompt rendering, privacy/truncation boundaries,
retry idempotency, schema-12 compatibility and the end-to-end two-turn case.
All providers are fakes; no network is used.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from companion.cognition.context_builder import ContextBuilder
from companion.cognition.models import (
    ActionHistoryRecord,
    CognitiveContext,
    CognitiveTurn,
)
from companion.cognition.nonresponse import build_action_history_record
from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ProcessingStatus,
    ReasonCode,
    ReplyNecessity,
)
from companion.domain.models import (
    AgentAction,
    CognitiveResult,
    NonresponseMotive,
    make_utterance_event,
    new_id,
    utcnow,
)
from companion.people.registry import PersonRegistry
from companion.prompts.cognitive import build_cognitive_messages
from companion.prompts.response import build_response_messages
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime
from companion.storage.database import Database, _statements
from companion.storage.repositories import make_repositories
from companion.storage.schema import MIGRATIONS

BASE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _at(minutes: int) -> datetime:
    return BASE + timedelta(minutes=minutes)


def _event(person_id, text, *, to="unknown", at=0, conversation="c1"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
        occurred_at=_at(at),
    )


def _analysis(
    event,
    *,
    action=CognitiveAction.WAIT,
    reason=ReasonCode.AMBIGUOUS_ADDRESSEE,
    necessity=ReplyNecessity.OPTIONAL,
    motive=None,
):
    return CognitiveResult(
        event_id=event.event_id,
        meaning="fake analysis",
        action=action,
        reason_code=reason,
        response_style="plain",
        provider="fake",
        reply_necessity=necessity,
        nonresponse_motive=motive,
    )


class ScriptedCognitive:
    def __init__(self, decide):
        self.decide = decide
        self.calls = []

    async def analyze(self, event, context, memories):
        self.calls.append((event, context))
        return self.decide(event, context)


class RecordingResponse:
    def __init__(self, text="收到。"):
        self.text = text
        self.calls = []

    async def generate(self, event, analysis, context):
        self.calls.append(event.event_id)
        return self.text


async def _register(rt, name="Alice", *, external_ai=False, persistence=True):
    return await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=persistence,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
    )


def _runtime(db, cognitive, response):
    rt = build_runtime(db, cognitive=cognitive, response=response)
    rt.orchestrator.context_builder = ContextBuilder(
        repos=rt.repos, registry=rt.registry
    )
    return rt


def _trace(action_type, cause="AMBIGUOUS_ADDRESSEE", source="MODEL_ASSESSMENT", **extra):
    payload = {
        "final_action": action_type.value,
        "cause_code": cause,
        "cause_source": source,
        "model_motive": None,
        "policy_override": None,
        "evidence_span": None,
        "pending_followup": False,
        "reentry_condition": "NONE",
        "spoken": False,
    }
    payload.update(extra)
    return payload


async def _save_turn(
    rt,
    person_id,
    minutes,
    text,
    *,
    action_type=CognitiveAction.WAIT,
    reason=ReasonCode.AMBIGUOUS_ADDRESSEE,
    trace=None,
    response=None,
):
    conversation_id = "c1"
    await rt.conversations.get_or_create(conversation_id)
    event = make_utterance_event(
        conversation_id=conversation_id,
        text=text,
        addressee_hint="unknown",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        occurred_at=_at(minutes),
    )
    await rt.repos.utterance.save(
        event.model_copy(update={"processing_status": ProcessingStatus.COMPLETED})
    )
    await rt.repos.analysis.save_action(
        AgentAction(
            action_id=new_id("act"),
            event_id=event.event_id,
            action_type=action_type,
            response_text=response,
            reason_code=reason,
            status="COMPLETED",
            created_at=utcnow(),
            nonresponse_trace=trace
            if trace is not None
            else _trace(action_type),
        )
    )
    return event


# -- durable WAIT/IGNORE, no fake assistant text -----------------------------


async def test_wait_is_saved_without_fake_assistant_text(db):
    cognitive = ScriptedCognitive(
        lambda event, ctx: _analysis(
            event,
            motive=NonresponseMotive(
                motive="UNCERTAIN_CONTEXT",
                confidence=0.9,
                evidence_span="你在听吗？",
                reentry_hint="WHEN_CONTEXT_CLEAR",
            ),
        )
    )
    response = RecordingResponse()
    rt = _runtime(db, cognitive, response)
    alice = await _register(rt)
    event = _event(alice.person_id, "你在听吗？", at=1)

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.WAIT
    assert report.response_text is None
    assert response.calls == []
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.action_type == CognitiveAction.WAIT
    assert action.response_text is None
    assert action.nonresponse_trace is not None
    assert action.nonresponse_trace["cause_code"] == "AMBIGUOUS_ADDRESSEE"
    assert action.nonresponse_trace["cause_source"] == "MODEL_ASSESSMENT"

    current = _event(alice.person_id, "现在呢？", at=2)
    ctx = await ContextBuilder(repos=rt.repos, registry=rt.registry).build(current, [])
    assert [r.event_id for r in ctx.action_history] == [event.event_id]
    assert [t.role for t in ctx.turns] == ["human"]
    assert "我选择了沉默" not in [t.text for t in ctx.turns]


async def test_ignore_is_saved_without_fake_assistant_text(db):
    cognitive = ScriptedCognitive(
        lambda event, ctx: _analysis(
            event,
            action=CognitiveAction.IGNORE,
            reason=ReasonCode.SOCIAL_RESPONSE_NOT_NEEDED,
        )
    )
    response = RecordingResponse()
    rt = _runtime(db, cognitive, response)
    alice = await _register(rt)
    event = _event(alice.person_id, "嗯。", at=1)

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.IGNORE
    assert response.calls == []
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action.nonresponse_trace["cause_code"] == "NO_REPLY_NEEDED"

    current = _event(alice.person_id, "那你怎么看？", to="agent", at=2)
    ctx = await ContextBuilder(repos=rt.repos, registry=rt.registry).build(current, [])
    assert ctx.action_history[0].final_action == "IGNORE"
    assert ctx.action_history[0].cause_code == "NO_REPLY_NEEDED"
    assert all(t.role != "assistant" for t in ctx.turns)


# -- prompt rendering --------------------------------------------------------


def test_prompts_render_action_history_as_action_not_speech():
    event = _event("person_1", "你刚才怎么没说话？", to="agent", at=2)
    ctx = CognitiveContext(
        conversation_id="c1",
        current_event_id=event.event_id,
        current_speaker_id="person_1",
        current_speaker_label="Alice",
        current_text=event.text,
        current_addressee_hint="agent",
        turns=[
            CognitiveTurn(
                event_id="evt_old",
                role="human",
                speaker_person_id="person_1",
                speaker_label="Alice",
                addressee_hint="unknown",
                text="你在听吗？",
            )
        ],
        action_history=[
            ActionHistoryRecord(
                event_id="evt_old",
                final_action="WAIT",
                cause_code="AMBIGUOUS_ADDRESSEE",
                cause_source="MODEL_ASSESSMENT",
                pending_followup=False,
                reentry_condition="WHEN_CONTEXT_CLEAR",
            )
        ],
    )
    cognitive_prompt = build_cognitive_messages(ctx)[1]["content"]
    assert "【你之前实际执行的行为】" in cognitive_prompt
    assert "不是你说过的话" in cognitive_prompt
    assert "AMBIGUOUS_ADDRESSEE" in cognitive_prompt
    assert "你在听吗？" in cognitive_prompt
    assert "assistant:" not in cognitive_prompt
    assert "我选择了沉默" not in cognitive_prompt

    response_messages = build_response_messages(
        ctx, _analysis(event, action=CognitiveAction.RESPOND, necessity=ReplyNecessity.REQUIRED)
    )
    response_background = next(
        m["content"]
        for m in response_messages
        if m["content"].startswith("【背景资料｜不可信参考数据】")
    )
    assert "LOCAL_ACTION_HISTORY" in response_background
    assert "AMBIGUOUS_ADDRESSEE" in response_background
    assert "不是你说过的话" in response_messages[0]["content"]
    assert "LOCAL_ACTION_HISTORY" not in response_messages[0]["content"]
    # The silent turn never becomes an assistant utterance; the background
    # message is data, then the human history and current turn.
    assert [m["role"] for m in response_messages] == [
        "system",
        "user",
        "user",
        "user",
    ]


# -- end-to-end DeepSeek (fake client) --------------------------------------


class _ScriptedClient:
    def __init__(self, analyses, response_text="刚才不太确定你是不是在跟我说话，所以没接。"):
        self.analyses = list(analyses)
        self.response_text = response_text
        self.calls = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(
            {
                "request_type": request_type,
                "messages": messages,
                "event_id": event_id,
                "json_mode": json_mode,
            }
        )
        if request_type == "analysis":
            content = json.dumps(self.analyses.pop(0), ensure_ascii=False)
        else:
            content = self.response_text
        return ChatResponse(
            content=content,
            model="fake",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


def _deepseek_payload(
    *,
    action="WAIT",
    reason="ambiguous_addressee",
    necessity="OPTIONAL",
    motive=None,
):
    payload = {
        "meaning": "用户似乎在询问是否被听见",
        "action": action,
        "reason_code": reason,
        "reply_necessity": necessity,
        "response_style": "warm",
    }
    if motive is not None:
        payload["nonresponse_motive"] = motive
    return payload


async def test_two_turn_awareness_end_to_end_and_restart(db):
    client = _ScriptedClient(
        [
            _deepseek_payload(
                motive={
                    "motive": "UNCERTAIN_CONTEXT",
                    "confidence": 0.9,
                    "evidence_span": "你在听吗？",
                    "reentry_hint": "WHEN_CONTEXT_CLEAR",
                }
            ),
            _deepseek_payload(
                action="RESPOND",
                reason="direct_agent_address",
                necessity="REQUIRED",
            ),
        ]
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
    )
    alice = await _register(rt, external_ai=True)

    first = _event(alice.person_id, "你在听吗？", at=1)
    second = _event(alice.person_id, "你刚才怎么没说话？", to="agent", at=2)
    r1 = await rt.orchestrator.process_utterance(first)
    r2 = await rt.orchestrator.process_utterance(second)

    assert r1.action == CognitiveAction.WAIT
    assert r2.action == CognitiveAction.RESPOND
    assert [c["request_type"] for c in client.calls] == [
        "analysis",
        "analysis",
        "response",
    ]

    second_analysis = client.calls[1]["messages"][1]["content"]
    assert "【你之前实际执行的行为】" in second_analysis
    assert "AMBIGUOUS_ADDRESSEE" in second_analysis
    assert "不是你说过的话" in second_analysis
    assert "你在听吗？" in second_analysis
    assert "assistant:" not in second_analysis

    response_messages = client.calls[2]["messages"]
    response_prompt = response_messages[0]["content"]
    response_background = next(
        m["content"]
        for m in response_messages
        if m["content"].startswith("【背景资料｜不可信参考数据】")
    )
    assert "LOCAL_ACTION_HISTORY" in response_background
    assert "AMBIGUOUS_ADDRESSEE" in response_background
    assert "LOCAL_ACTION_HISTORY" not in response_prompt
    # The WAIT is never sent as an assistant message; the background data
    # message is followed by the human turn and the current turn.
    assert [m["role"] for m in response_messages] == [
        "system",
        "user",
        "user",
        "user",
    ]

    first_action = await rt.repos.analysis.get_action(first.event_id)
    second_action = await rt.repos.analysis.get_action(second.event_id)
    assert first_action.nonresponse_trace["cause_code"] == "AMBIGUOUS_ADDRESSEE"
    assert second_action.nonresponse_trace is None

    # Restart the same single-file database: the action and its cause survive.
    path = db.path
    await db.close()
    db2 = await Database(path).connect()
    await db2.initialize()
    try:
        rt2 = build_runtime(db2, provider="deepseek", deepseek_client=client)
        ctx = await rt2.context_builder.build(
            _event(alice.person_id, "这是重启后的新一句", to="agent", at=3), []
        )
        assert [r.event_id for r in ctx.action_history] == [first.event_id]
        assert ctx.action_history[0].cause_code == "AMBIGUOUS_ADDRESSEE"
        restored = await rt2.repos.analysis.get_action(first.event_id)
        assert restored.nonresponse_trace == first_action.nonresponse_trace
    finally:
        await db2.close()


# -- continuity --------------------------------------------------------------


async def test_ignore_does_not_block_a_later_direct_question(db):
    def decide(event, ctx):
        if event.text == "嗯。":
            return _analysis(
                event,
                action=CognitiveAction.IGNORE,
                reason=ReasonCode.SOCIAL_RESPONSE_NOT_NEEDED,
            )
        return _analysis(
            event,
            action=CognitiveAction.RESPOND,
            reason=ReasonCode.DIRECT_AGENT_ADDRESS,
            necessity=ReplyNecessity.REQUIRED,
        )

    cognitive = ScriptedCognitive(decide)
    response = RecordingResponse("我觉得可以。")
    rt = _runtime(db, cognitive, response)
    alice = await _register(rt)

    first = _event(alice.person_id, "嗯。", at=1)
    second = _event(alice.person_id, "那你怎么看？", to="agent", at=2)
    await rt.orchestrator.process_utterance(first)
    report = await rt.orchestrator.process_utterance(second)

    assert report.action == CognitiveAction.RESPOND
    assert response.calls == [second.event_id]
    context_seen = cognitive.calls[1][1]
    assert [r.final_action for r in context_seen.action_history] == ["IGNORE"]


async def test_multiple_waits_are_visible_and_do_not_force_silence(db):
    def decide(event, ctx):
        if event.text != "现在回答我":
            text = event.text
            return _analysis(
                event,
                motive=NonresponseMotive(
                    motive="AWAITING_CONTINUATION",
                    confidence=0.8,
                    evidence_span=text,
                    reentry_hint="NEXT_TOPIC_CONTINUATION",
                ),
            )
        return _analysis(
            event,
            action=CognitiveAction.RESPOND,
            reason=ReasonCode.DIRECT_AGENT_ADDRESS,
            necessity=ReplyNecessity.REQUIRED,
        )

    cognitive = ScriptedCognitive(decide)
    response = RecordingResponse("我在。")
    rt = _runtime(db, cognitive, response)
    alice = await _register(rt)

    first = _event(alice.person_id, "第一句", at=1)
    second = _event(alice.person_id, "第二句", at=2)
    third = _event(alice.person_id, "现在回答我", to="agent", at=3)
    await rt.orchestrator.process_utterance(first)
    await rt.orchestrator.process_utterance(second)
    report = await rt.orchestrator.process_utterance(third)

    assert report.action == CognitiveAction.RESPOND
    context_seen = cognitive.calls[2][1]
    assert len(context_seen.action_history) == 2
    assert [r.final_action for r in context_seen.action_history] == ["WAIT", "WAIT"]
    assert all(r.pending_followup for r in context_seen.action_history)
    assert all(t.role != "assistant" for t in context_seen.turns)


# -- idempotency / persistence boundary -------------------------------------


async def test_retry_reuses_existing_trace_and_does_not_recompute(db, monkeypatch):
    import companion.core.orchestrator as orchestrator_module

    cognitive = ScriptedCognitive(
        lambda event, ctx: _analysis(
            event,
            motive=NonresponseMotive(
                motive="UNCERTAIN_CONTEXT",
                confidence=0.9,
                evidence_span="你在听吗？",
            ),
        )
    )
    response = RecordingResponse()
    rt = _runtime(db, cognitive, response)
    alice = await _register(rt)
    event = _event(alice.person_id, "你在听吗？", at=1)

    await rt.orchestrator.process_utterance(event)
    stored = await rt.repos.analysis.get_action(event.event_id)
    original_trace = dict(stored.nonresponse_trace)
    await rt.repos.utterance.update_status(
        event.event_id, ProcessingStatus.PROCESSING
    )

    def boom(**kwargs):
        raise AssertionError("derive_nonresponse_trace must not rerun on retry")

    monkeypatch.setattr(
        orchestrator_module, "derive_nonresponse_trace", boom
    )
    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.WAIT
    assert len(cognitive.calls) == 1
    stored = await rt.repos.analysis.get_action(event.event_id)
    assert stored.nonresponse_trace == original_trace
    cur = await db.execute(
        "SELECT COUNT(*) AS n FROM agent_actions WHERE event_id = ?",
        (event.event_id,),
    )
    assert (await cur.fetchone())["n"] == 1


async def test_no_persistence_permission_writes_no_action(db):
    cognitive = ScriptedCognitive(lambda event, ctx: _analysis(event))
    response = RecordingResponse()
    rt = _runtime(db, cognitive, response)
    alice = await _register(rt, persistence=False)
    event = _event(alice.person_id, "你在听吗？", at=1)

    report = await rt.orchestrator.process_utterance(event)

    assert report.action == CognitiveAction.WAIT
    assert await rt.repos.utterance.get(event.event_id) is None
    assert await rt.repos.analysis.get_action(event.event_id) is None
    assert response.calls == []


# -- privacy and truncation boundaries --------------------------------------


async def test_private_history_and_actions_are_not_leaked_to_another_user(db):
    rt = build_runtime(db)
    alice = await _register(rt, "Alice", external_ai=True)
    bob = await _register(rt, "Bob", external_ai=False)
    await rt.conversations.get_or_create("c1")

    bob_event = await _save_turn(
        rt, bob.person_id, 1, "Bob 的私密内容",
        trace=_trace(CognitiveAction.WAIT, cause="NO_REPLY_NEEDED"),
    )
    alice_event = await _save_turn(
        rt, alice.person_id, 2, "Alice 的公开内容",
        trace=_trace(CognitiveAction.WAIT),
    )

    gateway = ForegroundPrivacyGateway(rt.registry)
    builder = ContextBuilder(
        repos=rt.repos, registry=rt.registry, privacy_gateway=gateway
    )
    current = _event(alice.person_id, "现在的输入", to="agent", at=3)
    ctx = await builder.build(current, [])

    turn_texts = [t.text for t in ctx.turns]
    assert "Bob 的私密内容" not in turn_texts
    assert "Alice 的公开内容" in turn_texts
    assert [r.event_id for r in ctx.action_history] == [alice_event.event_id]
    assert bob_event.event_id not in {r.event_id for r in ctx.action_history}


async def test_truncated_turns_leave_no_orphan_action_history(db):
    rt = build_runtime(db)
    alice = await _register(rt)
    events = [
        await _save_turn(rt, alice.person_id, i, f"消息{i}")
        for i in range(1, 6)
    ]
    builder = ContextBuilder(
        repos=rt.repos,
        registry=rt.registry,
        max_turns=1,
        max_chars=None,
    )
    current = _event(alice.person_id, "当前", at=100)
    ctx = await builder.build(current, [])

    kept_event_ids = {t.event_id for t in ctx.turns if t.role == "human"}
    assert kept_event_ids == {events[-1].event_id}
    assert [r.event_id for r in ctx.action_history] == [events[-1].event_id]


async def test_action_history_is_bounded_to_the_latest_eight(db):
    rt = build_runtime(db)
    alice = await _register(rt)
    events = [
        await _save_turn(rt, alice.person_id, i, f"沉默{i}")
        for i in range(1, 11)
    ]
    current = _event(alice.person_id, "当前", at=100)
    ctx = await ContextBuilder(repos=rt.repos, registry=rt.registry).build(current, [])

    assert len(ctx.action_history) == 8
    assert [r.event_id for r in ctx.action_history] == [
        e.event_id for e in events[-8:]
    ]


# -- schema 12 compatibility -------------------------------------------------


async def test_schema_12_rows_migrate_and_keep_legacy_meaning(tmp_path):
    path = tmp_path / "v12.memory"
    ts = "2024-01-01T00:00:00+00:00"
    db = await Database(path).connect()
    for version, sql in MIGRATIONS:
        if version > 12:
            break
        for statement in _statements(sql):
            await db.conn.execute(statement)
        await db.conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (version, ts),
        )
    await db.conn.execute(
        "INSERT INTO conversations(conversation_id, quiet_mode, created_at, updated_at)"
        " VALUES ('c1', 0, ?, ?)",
        (ts, ts),
    )
    await db.conn.execute(
        "INSERT INTO utterances(event_id, conversation_id, speaker_person_id, text,"
        " source, addressee_hint, occurred_at, processing_status, created_at)"
        " VALUES ('evt_old', 'c1', NULL, '旧输入', 'test', 'unknown', ?,"
        " 'COMPLETED', ?)",
        (ts, ts),
    )
    await db.conn.execute(
        "INSERT INTO cognitive_analyses(analysis_id, event_id, meaning, action,"
        " reason_code, provider, response_style, created_at)"
        " VALUES ('ana_old', 'evt_old', '旧分析', 'WAIT', 'ambiguous_addressee',"
        " 'fake', 'plain', ?)",
        (ts,),
    )
    await db.conn.execute(
        "INSERT INTO agent_actions(action_id, event_id, action_type, response_text,"
        " reason_code, status, created_at)"
        " VALUES ('act_old', 'evt_old', 'WAIT', NULL, 'ambiguous_addressee',"
        " 'COMPLETED', ?)",
        (ts,),
    )
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    try:
        repos = make_repositories(db2)
        action = await repos.analysis.get_action("evt_old")
        assert action is not None
        assert action.nonresponse_trace is None
        record = build_action_history_record(action)
        assert record.final_action == "WAIT"
        assert record.cause_code == "AMBIGUOUS_ADDRESSEE"
        assert record.cause_source == "LEGACY_UNKNOWN"
        assert record.pending_followup is False

        analysis = await repos.analysis.get_analysis("evt_old")
        assert analysis is not None
        assert analysis.nonresponse_motive is None

        registry = PersonRegistry(repos.person)
        context = await ContextBuilder(repos=repos, registry=registry).build(
            make_utterance_event(
                conversation_id="c1",
                text="迁移后的当前输入",
                addressee_hint="agent",
                speaker_person_id=None,
                source=EventSource.TEST,
                occurred_at=_at(10),
            ),
            [],
        )
        assert [r.cause_source for r in context.action_history] == [
            "LEGACY_UNKNOWN"
        ]
    finally:
        await db2.close()
