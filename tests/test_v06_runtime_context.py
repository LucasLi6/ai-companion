"""V0.6 runtime context: trusted time, temporal gaps and prompt injection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from companion.domain.enums import CognitiveAction, EventSource, ProcessingStatus
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime
from companion.runtime_context.temporal import (
    REENTRY_ACTIVE,
    REENTRY_CONTINUING,
    REENTRY_NEW,
    REENTRY_RETURNING,
    build_temporal_signals,
    humanize_gap,
)
from companion.runtime_context.timezone import TimeZoneService


class CapturingClient:
    def __init__(self):
        self.messages: list[list[dict]] = []

    async def complete(self, messages, *, request_type, **kwargs):
        self.messages.append(messages)
        return ChatResponse(
            content="我在。",
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _runtime(db, client):
    rt = build_runtime(db, provider="deepseek", deepseek_client=client)
    person = await rt.registry.register(
        "Alice",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    return rt, person


def _event(person_id, text, *, at=None):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
        occurred_at=at,
    )


# -- time zone resolution ----------------------------------------------------


def test_user_config_outranks_trusted_device_and_geo():
    service = TimeZoneService(system_zone="UTC")
    zone = service.resolve(
        user_timezone="Asia/Shanghai",
        device_timezone="America/Los_Angeles",
        device_trusted=True,
        geo_timezone="Europe/Paris",
        at=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc),
    )
    assert (zone.name, zone.source) == ("Asia/Shanghai", "user_config")
    assert zone.offset_seconds == 8 * 3600


def test_untrusted_device_zone_is_ignored():
    # An invalid system zone forces the documented UTC fallback, so the test
    # does not depend on the host's own configuration.
    service = TimeZoneService(system_zone="Not/AZone")
    zone = service.resolve(
        device_timezone="America/New_York",
        device_trusted=False,
        at=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc),
    )
    assert zone.source == "utc"
    assert zone.name == "UTC"


def test_geo_zone_is_used_only_when_present_and_valid():
    service = TimeZoneService(system_zone="UTC")
    zone = service.resolve(
        geo_timezone="Europe/Paris",
        at=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc),
    )
    assert (zone.name, zone.source) == ("Europe/Paris", "ip_geo")

    fallback = service.resolve(
        geo_timezone="Not/AZone",
        at=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc),
    )
    assert (fallback.name, fallback.source) == ("UTC", "system_default")


def test_daylight_saving_is_derived_not_hardcoded():
    service = TimeZoneService(system_zone="America/New_York")
    before = service.resolve(at=datetime(2026, 3, 8, 6, 30, tzinfo=timezone.utc))
    after = service.resolve(at=datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc))
    assert before.offset_seconds == -5 * 3600
    assert after.offset_seconds == -4 * 3600


# -- temporal continuity -----------------------------------------------------


def test_reentry_states_are_labels_only():
    now = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    signals = build_temporal_signals(now)
    assert signals.reentry_state == REENTRY_NEW
    assert signals.last_heard_gap_seconds is None

    ten_minutes = build_temporal_signals(
        now, last_heard_at=now - timedelta(minutes=10)
    )
    assert ten_minutes.reentry_state == REENTRY_ACTIVE

    two_hours = build_temporal_signals(
        now, last_heard_at=now - timedelta(hours=2)
    )
    assert two_hours.reentry_state == REENTRY_CONTINUING

    two_days = build_temporal_signals(
        now, last_heard_at=now - timedelta(days=2)
    )
    assert two_days.reentry_state == REENTRY_RETURNING
    assert two_days.reentry is True


def test_three_gaps_are_computed_independently():
    now = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    signals = build_temporal_signals(
        now,
        last_heard_at=now - timedelta(hours=3),
        last_interaction_at=now - timedelta(minutes=30),
        last_meaningful_context_at=now - timedelta(hours=3),
    )
    assert signals.last_heard_gap_seconds == 3 * 3600
    assert signals.last_interaction_gap_seconds == 30 * 60
    assert signals.last_meaningful_context_gap_seconds == 3 * 3600


def test_gap_rendering_is_compact_and_never_invents_precision():
    assert humanize_gap(None) is None
    assert humanize_gap(5) == "刚刚"
    assert humanize_gap(90) == "1 分钟"
    assert humanize_gap(2 * 3600 + 5 * 60) == "2 小时 5 分钟"
    assert humanize_gap(25 * 3600) == "1 天 1 小时"


# -- per-event snapshot ------------------------------------------------------


async def test_snapshot_is_built_once_and_stays_stable_across_a_retry(db):
    client = CapturingClient()
    rt, person = await _runtime(db, client)
    event = _event(person.person_id, "你好")

    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "COMPLETED"
    stored = await rt.repos.runtime_snapshot.get(event.event_id)
    assert stored is not None
    original = (
        stored.received_at_utc,
        stored.occurred_at_utc,
        stored.last_heard_gap_seconds,
    )

    # Make the turn fail only after the snapshot exists, then retry.
    await rt.repos.utterance.update_status(
        event.event_id, ProcessingStatus.FAILED
    )
    second = await rt.orchestrator.process_utterance(event)
    assert second.status == "COMPLETED"
    again = await rt.repos.runtime_snapshot.get(event.event_id)
    assert again is not None
    assert (again.received_at_utc, again.occurred_at_utc, again.last_heard_gap_seconds) == original


async def test_runtime_block_is_trusted_data_after_the_persona_prefix(db):
    client = CapturingClient()
    rt, person = await _runtime(db, client)
    await rt.orchestrator.process_utterance(_event(person.person_id, "你好"))

    system = client.messages[0][0]["content"]
    assert "【当前时间与状态】（本地可信数据）" in system
    identity = system.index("【角色身份】")
    runtime_block = system.index("【当前时间与状态】")
    assert identity < runtime_block
    # The prompt says the gap is a fact, not a greeting script.
    assert "不是固定话术" in system


async def test_two_turns_produce_real_gaps_in_the_prompt(db):
    client = CapturingClient()
    rt, person = await _runtime(db, client)
    start = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    await rt.orchestrator.process_utterance(
        _event(person.person_id, "早上好", at=start)
    )
    second = _event(
        person.person_id, "中午好", at=start + timedelta(hours=2)
    )
    await rt.orchestrator.process_utterance(second)

    snapshot = await rt.repos.runtime_snapshot.get(second.event_id)
    assert snapshot is not None
    assert snapshot.last_heard_gap_seconds == 2 * 3600
    assert snapshot.last_meaningful_context_gap_seconds == 2 * 3600
    assert snapshot.reentry_state == REENTRY_CONTINUING


async def test_emotion_state_is_attached_to_the_snapshot(db):
    client = CapturingClient()
    rt, person = await _runtime(db, client)
    event = _event(person.person_id, "今天有点累")

    await rt.orchestrator.process_utterance(event)

    snapshot = await rt.repos.runtime_snapshot.get(event.event_id)
    assert snapshot is not None
    assert snapshot.ava_emotion  # a compact label, not raw model output


async def test_builder_does_not_store_a_location_without_geo(db):
    client = CapturingClient()
    rt, person = await _runtime(db, client)
    from companion.runtime_context.connection import ConnectionInfo

    event = _event(person.person_id, "你好")
    snapshot = await rt.orchestrator.runtime_snapshot_builder.build(
        event,
        connection=ConnectionInfo(remote_ip="8.8.8.8"),
        allow_external_ai=True,
        persist=False,
    )
    assert snapshot.location is None
