"""V0.6.1 offline regression tests.

Covers the repair contract without any real network call:

- the REPL consumes the real streaming API and never prints a reply twice;
- failure / cancellation / WAIT are reported truthfully;
- ``time`` is read at generation while ``event_time`` is the stored event time;
- a per-speaker time zone wins over the application default;
- only RESPOND counts as a last interaction;
- the model can tell "no memories", "no retrieval hit" and "history missing"
  apart, and an explicit sample never crosses another person's privacy;
- prompt budgets are bounded and the diagnostic is explicitly an estimate.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from companion.cli import _run_streaming_turn, cmd_runtime_inspect
from companion.cognition.context_builder import (
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TURNS,
)
from companion.core.response_events import ResponseEvent
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ReasonCode,
)
from companion.domain.models import make_utterance_event
from companion.memory.models import AccessContext
from companion.prompts.response import (
    MAX_BACKGROUND_MEMORIES,
    estimate_prompt_budget,
)
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime
from companion.runtime_context.assembler import compile_runtime_context
from companion.runtime_context.snapshot import RuntimeSnapshot


# -- REPL streaming ----------------------------------------------------------


class _Stream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        self._iter = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        return None


class _Orchestrator:
    def __init__(self, events):
        self._events = events

    def process_utterance_stream(self, event):
        return _Stream(self._events)


class _AnalysisRepo:
    def __init__(self, action=None):
        self._action = action

    async def get_action(self, event_id):
        return self._action


def _event(text="你好", *, person_id="person_1", event_id="evt_1"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        event_id=event_id,
    )


def _stream_event(kind, *, text=None, delta=None, status="STREAMING", **extra):
    return ResponseEvent(
        type=kind,
        event_id="evt_1",
        conversation_id="c1",
        response_id="resp_1",
        sequence=0,
        status=status,
        text=text,
        delta=delta,
        **extra,
    )


def _stub_runtime(events, *, action=None):
    return SimpleNamespace(
        orchestrator=_Orchestrator(events),
        repos=SimpleNamespace(analysis=_AnalysisRepo(action)),
    )


async def test_repl_streams_deltas_without_printing_the_text_twice(capsys):
    rt = _stub_runtime(
        [
            _stream_event("response_started"),
            _stream_event("response_delta", delta="你"),
            _stream_event("response_delta", delta="好"),
            _stream_event(
                "response_completed", text="你好", status="COMPLETED"
            ),
        ]
    )
    await _run_streaming_turn(rt, _event())
    out = capsys.readouterr().out
    assert out == "assistant> 你好\n"
    assert out.count("你好") == 1


async def test_repl_prints_a_non_streaming_reply_once(capsys):
    rt = _stub_runtime(
        [
            _stream_event("response_started"),
            _stream_event(
                "response_completed", text="完整回复", status="COMPLETED"
            ),
        ]
    )
    await _run_streaming_turn(rt, _event())
    out = capsys.readouterr().out
    assert out.count("完整回复") == 1


async def test_repl_reports_failure_truncation_and_cancellation(capsys):
    failed = _stub_runtime(
        [
            _stream_event("response_started"),
            _stream_event("response_delta", delta="半"),
            _stream_event(
                "response_failed", text="半", status="FAILED", error="boom"
            ),
        ]
    )
    await _run_streaming_turn(failed, _event())
    out = capsys.readouterr().out
    assert "[failed] boom" in out
    assert "半" in out

    truncated = _stub_runtime(
        [
            _stream_event("response_started"),
            _stream_event("response_delta", delta="半句"),
            _stream_event(
                "response_completed",
                text="半句",
                status="TRUNCATED",
                finish_reason="length",
            ),
        ]
    )
    await _run_streaming_turn(truncated, _event())
    assert "[truncated: token limit]" in capsys.readouterr().out

    cancelled = _stub_runtime([_stream_event("response_cancelled")])
    await _run_streaming_turn(cancelled, _event())
    assert "[cancelled]" in capsys.readouterr().out


async def test_repl_is_silent_for_wait_but_debug_shows_the_action(capsys):
    action = SimpleNamespace(
        action_type=CognitiveAction.WAIT,
        reason_code=ReasonCode.AMBIGUOUS_ADDRESSEE,
    )
    rt = _stub_runtime([], action=action)

    # Ordinary chat: WAIT is intentional silence, with no assistant line and
    # no internal action label.
    await _run_streaming_turn(rt, _event())
    assert capsys.readouterr().out == ""

    # Diagnostics remain available explicitly.
    await _run_streaming_turn(rt, _event(), debug=True)
    assert capsys.readouterr().out.strip() == "[WAIT (ambiguous_addressee)]"


# -- time and continuity -----------------------------------------------------


def _snapshot(**updates):
    payload = dict(
        snapshot_id="rts_e1",
        event_id="evt_1",
        conversation_id="c1",
        speaker_person_id="person_1",
        speaker_label="Alice",
        received_at_utc=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc),
        occurred_at_utc=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc),
        timezone="Asia/Shanghai",
        timezone_source="user_config",
        utc_offset_seconds=8 * 3600,
        local_time="2026-11-04 18:00",
    )
    payload.update(updates)
    return RuntimeSnapshot(**payload)


def test_time_is_read_at_generation_but_event_time_is_stored():
    snapshot = _snapshot()
    block = compile_runtime_context(
        snapshot, now=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    )
    assert "time=2026-11-04 20:00" in block
    assert "event_time=2026-11-04 18:00" in block
    assert "Asia/Shanghai" in block


class _FakeClient:
    def __init__(self, analysis_action="RESPOND"):
        self.analysis_action = analysis_action
        self.response_text = "好。"
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append({"request_type": request_type, "messages": messages})
        if request_type == "analysis":
            content = json.dumps(
                {
                    "meaning": "对方在说话",
                    "action": self.analysis_action,
                    "reason_code": "direct_agent_address",
                    "referenced_memory_ids": [],
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
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

    def response_system(self) -> str:
        calls = [
            call for call in self.calls if call["request_type"] == "response"
        ]
        return calls[-1]["messages"][0]["content"]

    def response_messages(self) -> list[dict]:
        calls = [
            call for call in self.calls if call["request_type"] == "response"
        ]
        return calls[-1]["messages"]


async def _person(rt, name="Alice"):
    return await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


async def _remember(rt, person, content):
    await rt.memory_service.create(
        kind="EPISODE",
        content=content,
        access=AccessContext("admin", is_admin=True),
        visibility="PRIVATE",
        subjects=[("PERSON", person.person_id)],
    )


async def test_per_speaker_timezone_wins_over_the_application_default(db):
    client = _FakeClient()
    rt = build_runtime(
        db, provider="deepseek", deepseek_client=client, fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    await rt.repos.settings.set("runtime.timezone", "America/New_York")
    await rt.repos.settings.set(
        f"runtime.timezone.person.{person.person_id}", "Asia/Shanghai"
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="现在几点？",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
        occurred_at=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc),
    )

    snapshot = await rt.orchestrator.runtime_snapshot_builder.build(
        event, persist=False
    )
    assert snapshot.timezone == "Asia/Shanghai"
    assert snapshot.timezone_source == "user_config"


async def test_wait_and_ignore_do_not_count_as_last_interaction(db):
    client = _FakeClient(analysis_action="WAIT")
    rt = build_runtime(
        db, provider="deepseek", deepseek_client=client, fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    start = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)

    await rt.orchestrator.process_utterance(
        make_utterance_event(
            conversation_id="c1",
            text="嗯。",
            addressee_hint="unknown",
            speaker_person_id=person.person_id,
            source=EventSource.TEST,
            occurred_at=start,
        )
    )

    client.analysis_action = "RESPOND"
    second = make_utterance_event(
        conversation_id="c1",
        text="在吗？",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
        occurred_at=start + timedelta(minutes=5),
    )
    await rt.orchestrator.process_utterance(second)
    snapshot = await rt.repos.runtime_snapshot.get(second.event_id)
    assert snapshot is not None
    # The previous WAIT is not an interaction.
    assert snapshot.last_interaction_at_utc is None

    third = make_utterance_event(
        conversation_id="c1",
        text="还在吗？",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
        occurred_at=start + timedelta(minutes=10),
    )
    await rt.orchestrator.process_utterance(third)
    snapshot = await rt.repos.runtime_snapshot.get(third.event_id)
    assert snapshot is not None
    assert snapshot.last_interaction_at_utc == second.occurred_at


# -- memory truthfulness -----------------------------------------------------


async def test_memory_status_distinguishes_no_db_empty_and_hits(db):
    client = _FakeClient()
    rt = build_runtime(
        db, provider="deepseek", deepseek_client=client, fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)

    await rt.orchestrator.process_utterance(_event(person_id=person.person_id))
    assert "此人目前没有任何长期记忆记录" in client.response_system()

    await _remember(rt, person, "Alice 喜欢猫。")
    await rt.orchestrator.process_utterance(
        _event("今天天气怎么样", person_id=person.person_id, event_id="evt_2")
    )
    assert "本次没有检索到相关记忆" in client.response_system()
    assert "不代表长期记忆数据库为空" in client.response_system()

    await rt.orchestrator.process_utterance(
        _event("我是不是喜欢猫？", person_id=person.person_id, event_id="evt_3")
    )
    system = client.response_system()
    assert "找到相关记忆" in system
    assert "Alice 喜欢猫。" in json.dumps(
        client.response_messages(), ensure_ascii=False
    )


async def test_explicit_memory_sample_and_earliest_history(db):
    client = _FakeClient()
    rt = build_runtime(
        db, provider="deepseek", deepseek_client=client, fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    await _remember(rt, person, "Alice 喜欢猫。")

    await rt.orchestrator.process_utterance(
        _event("举一个你的记忆", person_id=person.person_id, event_id="evt_s")
    )
    system = client.response_system()
    assert "【你实际的长期记忆样本】" in system
    assert "Alice 喜欢猫。" in system

    first = _event(
        "这是我们最早说的话", person_id=person.person_id, event_id="evt_first"
    )
    await rt.orchestrator.process_utterance(first)
    await rt.orchestrator.process_utterance(
        _event(
            "我们第一次聊的是什么？",
            person_id=person.person_id,
            event_id="evt_e",
        )
    )
    system = client.response_system()
    assert "【最早的真实对话】" in system
    assert "这是我们最早说的话" in system


async def test_memory_sample_never_leaks_another_person(db):
    client = _FakeClient()
    rt = build_runtime(
        db, provider="deepseek", deepseek_client=client, fast_path_enabled=False,
        router_mode="legacy",
)
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")
    await _remember(rt, bob, "Bob 的私密事实：他的银行卡密码。")

    await rt.orchestrator.process_utterance(
        _event("举一个你的记忆", person_id=alice.person_id, event_id="evt_alice")
    )
    rendered = json.dumps(client.response_messages(), ensure_ascii=False)
    assert "银行卡密码" not in rendered


# -- token budgets -----------------------------------------------------------


def test_prompt_budget_is_bounded_and_labelled_as_an_estimate():
    from companion.cognition.models import CognitiveContext, CognitiveTurn

    turns = [
        CognitiveTurn(
            event_id=f"e{i}",
            role="human",
            speaker_person_id="person_1",
            speaker_label="Alice",
            text="短历史",
        )
        for i in range(8)
    ]
    memories = [
        {"kind": "EPISODE", "content": "较长的记忆内容" * 20} for _ in range(8)
    ]
    context = CognitiveContext(
        conversation_id="c1",
        current_event_id="evt_now",
        current_speaker_id="person_1",
        current_speaker_label="Alice",
        current_text="你好",
        turns=turns,
        memories=memories,
        scene_state={
            "topic": "旅行",
            "open_loops": [
                {"topic": f"t{i}", "summary": "未完成" * 30} for i in range(10)
            ],
        },
    )

    budget = estimate_prompt_budget(context)
    assert budget["total_tokens_estimate"] > 0
    assert "not DeepSeek billing usage" in budget["note"]
    assert budget["chars"]["memory"] > 0
    # Hard caps keep one turn bounded: three memories and at most five loops.
    assert budget["chars"]["memory"] < sum(
        len(m["content"]) for m in memories
    )
    assert MAX_BACKGROUND_MEMORIES == 3
    assert (DEFAULT_HISTORY_LIMIT, DEFAULT_MAX_TURNS, DEFAULT_MAX_CHARS) == (
        8,
        16,
        1800,
    )
    # V0.9.6's base prompt is deliberately longer (continuity + information
    # digestion rules), and V0.9.9.8 adds the shared understanding/priority
    # section. The ceiling moves with the base prompt; the per-section caps are
    # unchanged and still bound the turn.
    assert budget["total_tokens_estimate"] < 3800


async def test_runtime_inspect_shows_metadata_without_conversation_text(
    db, capsys
):
    client = _FakeClient()
    rt = build_runtime(
        db, provider="deepseek", deepseek_client=client, fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    event = _event("这是不能打印的私密正文", person_id=person.person_id)
    await rt.orchestrator.process_utterance(event)

    rc = await cmd_runtime_inspect(
        SimpleNamespace(db=Path(db.path), event_id=event.event_id)
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert f"event: {event.event_id}" in out
    assert "occurred_at_utc:" in out
    assert "timezone:" in out
    assert "time=" in out
    assert "event_time=" in out
    assert "这是不能打印的私密正文" not in out
    assert "192.168" not in out


async def test_dropped_memory_does_not_claim_a_retrieval_hit(db):
    client = _FakeClient()
    rt = build_runtime(
        db, provider="deepseek", deepseek_client=client, fast_path_enabled=False,
        router_mode="legacy",
)
    alice = await _person(rt, "Alice")
    bob = await rt.registry.register(
        "Bob",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=False,
    )
    await rt.memory_service.create(
        kind="EPISODE",
        content="Alice 和 Bob 一起喝咖啡。",
        access=AccessContext("admin", is_admin=True),
        visibility="PRIVATE",
        subjects=[("PERSON", alice.person_id), ("PERSON", bob.person_id)],
    )

    await rt.orchestrator.process_utterance(
        _event("我们喝咖啡那件事", person_id=alice.person_id, event_id="evt_shared")
    )
    system = client.response_system()
    # Bob's memory is dropped by the gateway, so there is no visible hit.
    assert "本次没有检索到相关记忆" in system
    assert "找到相关记忆" not in system
    assert "Bob" not in json.dumps(
        client.response_messages(), ensure_ascii=False
    )
