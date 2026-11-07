"""V0.9.9.7: time as context and memory as background knowledge.

The runtime block carries facts the trusted local code knows: local date,
weekday, daytime phase, the gap since the last real reply, and a return signal
only when the person's previous turn was an explicit departure. The Speaker
prompt integrates time and background knowledge into the existing continuity
and digestion sections instead of appending a second instruction layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.prompts.response import BASE_RESPONSE_PROMPT
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime
from companion.runtime_context.assembler import compile_runtime_context
from companion.runtime_context.snapshot import RuntimeSnapshot
from companion.runtime_context.temporal import (
    build_temporal_signals,
    day_phase,
    weekday_label,
)


class CapturingClient:
    def __init__(self):
        self.messages: list[list[dict]] = []

    async def complete(self, messages, *, request_type, **kwargs):
        self.messages.append(messages)
        return ChatResponse(
            content="嗯。",
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _runtime(db):
    client = CapturingClient()
    rt = build_runtime(
        db, provider="deepseek", deepseek_client=client, router_mode="legacy"
    )
    person = await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("c1")
    return rt, person, client


def _event(person_id, text, *, at):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        occurred_at=at,
    )


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


def _system(client) -> str:
    return client.messages[-1][0]["content"]


# -- local daytime phase -----------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (0, "凌晨"),
        (4, "凌晨"),
        (5, "早晨"),
        (10, "早晨"),
        (11, "中午"),
        (12, "中午"),
        (13, "下午"),
        (17, "下午"),
        (18, "晚上"),
        (22, "晚上"),
        (23, "深夜"),
    ],
)
def test_day_phase_is_computed_from_the_local_hour(hour, expected):
    assert day_phase(datetime(2026, 11, 4, hour, 30)) == expected


def test_weekday_label_uses_the_local_date():
    assert weekday_label(datetime(2026, 11, 4, 20, 0)) == "周三"
    assert weekday_label(datetime(2026, 11, 8, 9, 0)) == "周日"


def test_local_phase_follows_the_trusted_zone_not_the_server_clock():
    snapshot = _snapshot()
    block = compile_runtime_context(
        snapshot, now=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    )
    # 12:00 UTC is 20:00 in Asia/Shanghai: the block must say 晚上, not 中午.
    assert "time=2026-11-04 20:00 周三" in block
    assert "晚上" in block
    assert "Asia/Shanghai" in block


def test_runtime_block_reports_the_gap_since_the_last_real_reply():
    snapshot = _snapshot(
        last_interaction_at_utc=datetime(2026, 11, 4, 7, 30, tzinfo=timezone.utc),
        last_interaction_gap_seconds=2 * 3600 + 30 * 60,
    )
    block = compile_runtime_context(
        snapshot, now=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    )
    assert "last_interaction=2026-11-04 15:30" in block
    assert "2 小时 30 分钟前" in block


def test_unknown_zone_is_not_presented_as_the_users_local_time():
    snapshot = _snapshot(
        timezone="UTC", timezone_source="utc", utc_offset_seconds=0
    )
    block = compile_runtime_context(
        snapshot, now=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    )
    assert "time=2026-11-04 10:00" in block
    assert "时区未确定，非用户当地时间" in block


def test_host_fallback_zone_is_labelled_as_possibly_not_the_users():
    snapshot = _snapshot(
        timezone="America/Chicago",
        timezone_source="system_default",
        utc_offset_seconds=-6 * 3600,
    )
    block = compile_runtime_context(
        snapshot, now=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    )
    assert "系统时区，可能非用户当地时间" in block
    assert "America/Chicago" in block


# -- return evidence ---------------------------------------------------------


def test_a_return_is_emitted_only_after_an_explicit_departure():
    snapshot = _snapshot(previous_turn_departure=True)
    block = compile_runtime_context(
        snapshot, now=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    )
    assert "return=上次发言为明确离开，现在是再次出现" in block

    quiet = compile_runtime_context(
        _snapshot(previous_turn_departure=False),
        now=datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc),
    )
    assert "return=" not in quiet


@pytest.mark.asyncio
async def test_builder_detects_departure_structurally_not_from_silence(db):
    rt, person, _client = await _runtime(db)
    start = datetime(2026, 11, 4, 20, 0, tzinfo=timezone.utc)

    first = _event(person.person_id, "我先去睡了，晚安。", at=start)
    await rt.repos.utterance.save(first)

    later = _event(
        person.person_id, "在吗", at=start + timedelta(hours=3)
    )
    snapshot = await rt.orchestrator.runtime_snapshot_builder.build(
        later, persist=False
    )
    assert snapshot.previous_turn_departure is True
    assert snapshot.reentry_state == "CONTINUING"
    block = compile_runtime_context(snapshot, now=later.occurred_at)
    assert "return=" in block

    # A long gap with a neutral previous turn is not a departure.
    rt2, person2, _ = await _runtime(db)
    neutral = _event(person2.person_id, "今天挺忙的。", at=start)
    await rt2.repos.utterance.save(neutral)
    after = _event(
        person2.person_id, "在吗", at=start + timedelta(days=1)
    )
    plain = await rt2.orchestrator.runtime_snapshot_builder.build(
        after, persist=False
    )
    assert plain.previous_turn_departure is False
    assert "return=" not in compile_runtime_context(plain, now=after.occurred_at)


@pytest.mark.asyncio
async def test_hypothetical_question_and_negated_departures_are_not_returns(db):
    rt, person, _client = await _runtime(db)
    start = datetime(2026, 11, 4, 20, 0, tzinfo=timezone.utc)

    for text in ("如果我先走了呢", "我先走了吗？", "他说我先走了", "我不走"):
        event = _event(person.person_id, text, at=start)
        await rt.repos.utterance.save(event)
        following = _event(
            person.person_id,
            "继续刚才的话题",
            at=start + timedelta(minutes=30),
        )
        snapshot = await rt.orchestrator.runtime_snapshot_builder.build(
            following, persist=False
        )
        assert snapshot.previous_turn_departure is False, text
        start += timedelta(hours=1)


@pytest.mark.asyncio
async def test_speaker_receives_the_return_fact_without_a_greeting_script(db):
    rt, person, client = await _runtime(db)
    start = datetime(2026, 11, 4, 20, 0, tzinfo=timezone.utc)

    first = _event(person.person_id, "我先去睡了，晚安。", at=start)
    result = await rt.orchestrator.process_utterance(first)
    assert result.status == "COMPLETED"
    assert "return=" not in _system(client)

    later = _event(
        person.person_id, "在吗", at=start + timedelta(hours=3)
    )
    result = await rt.orchestrator.process_utterance(later)
    assert result.status == "COMPLETED"
    system = _system(client)
    assert "return=上次发言为明确离开，现在是再次出现" in system
    # The runtime block states a fact; the prompt, not the block, decides how
    # to react. No fixed greeting or "welcome back" line is prescribed.
    assert "欢迎回来" not in system
    assert "必须问候" not in system


def test_temporal_signals_carry_departure_as_plain_data():
    now = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    signals = build_temporal_signals(
        now, last_heard_at=now - timedelta(hours=1), previous_turn_departure=True
    )
    assert signals.previous_turn_departure is True
    assert build_temporal_signals(now).previous_turn_departure is False


# -- Speaker prompt ----------------------------------------------------------


def test_prompt_integrates_time_into_continuity_instead_of_a_new_section():
    assert "【时间意识】" not in BASE_RESPONSE_PROMPT
    continuity = BASE_RESPONSE_PROMPT.split("【交流连续性】", 1)[1]
    assert "你对交流具有连续的时间意识" in continuity
    assert "不要把沉默等同于离开" in continuity
    assert "不要根据时间猜测对方的现实活动" in continuity
    assert "而不是成为你需要展示的一项能力" in continuity
    # The old continuity text is still there, not duplicated in a new block.
    assert "你正在经历一段连续的交流，而不是每轮重新开始。" in continuity
    assert BASE_RESPONSE_PROMPT.count("你正在经历一段连续的交流") == 1


def test_prompt_uses_memory_as_background_not_as_a_process_report():
    digestion = BASE_RESPONSE_PROMPT.split("【信息消化】", 1)[1]
    assert "知识背景" in digestion
    assert "而不是路由、查询和检索过程" in digestion
    assert "无须展示自己记住了多少" in digestion
    assert "当对方明确询问信息来源、记忆状态、系统机制或操作结果时" in digestion


def test_prompt_keeps_honesty_and_does_not_script_time_talk():
    for banned in ("我记得我们刚才讨论过", "根据我们的上下文", "你刚刚提到了"):
        assert banned not in BASE_RESPONSE_PROMPT
    assert "始终区分已知、推断和未知" in BASE_RESPONSE_PROMPT
    assert "没有可靠证据时，不编造事实、记忆、工具结果或现实环境。" in (
        BASE_RESPONSE_PROMPT
    )
