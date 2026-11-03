"""V0.3.1 transactional emotion turns: receipts, rollback, concurrency.

The whole read-modify-write of one turn must behave like a single durable
operation: one instant, one transaction, one receipt, at most one event.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from companion.emotion.service import EmotionProfileConflict


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


def _freeze(rt, moment):
    rt.emotion_service.clock = lambda: moment


async def _count(rt, table):
    cur = await rt.db.execute(f"SELECT COUNT(*) AS n FROM {table}")
    return int((await cur.fetchone())["n"])


async def test_process_turn_writes_event_snapshot_receipt_at_one_instant(rt):
    agent_id = await _agent_id(rt)
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    _freeze(rt, now)

    state = await rt.emotion_service.process_turn(
        agent_id,
        "person_tx",
        text="我成功了",
        conversation_id=None,
        source_event_id="evt_tx",
        persist=True,
    )
    assert state.emotion == "JOY"

    event = await rt.emotion_service.repo.get_event_for_source("evt_tx")
    snapshot = await rt.emotion_service.repo.get_snapshot(agent_id, "person_tx")
    receipt = await rt.emotion_service.repo.get_receipt("evt_tx")
    assert event is not None and snapshot is not None and receipt is not None
    # Every timestamp comes from the one injected clock value.
    assert event["created_at"] == now.isoformat()
    assert snapshot["updated_at"] == now.isoformat()
    assert receipt["processed_at"] == now.isoformat()
    assert receipt["outcome"] == "APPLIED"


async def test_process_turn_idempotent_by_receipt(rt):
    agent_id = await _agent_id(rt)
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    _freeze(rt, now)

    first = await rt.emotion_service.process_turn(
        agent_id, "p", text="我成功了", source_event_id="evt_same"
    )
    assert first.emotion == "JOY"

    later = now + timedelta(seconds=30)
    _freeze(rt, later)
    second = await rt.emotion_service.process_turn(
        agent_id, "p", text="我成功了", source_event_id="evt_same"
    )
    assert await _count(rt, "emotion_events") == 1
    assert await _count(rt, "emotion_turn_receipts") == 1
    # The second call returns the decayed state; it did not add the delta again.
    assert second.intensity < first.intensity


async def test_process_turn_no_match_writes_receipt_only(rt):
    agent_id = await _agent_id(rt)
    state = await rt.emotion_service.process_turn(
        agent_id, "p2", text="随便聊聊天气", source_event_id="evt_none"
    )
    assert state.emotion == "NEUTRAL"
    receipt = await rt.emotion_service.repo.get_receipt("evt_none")
    assert receipt is not None and receipt["outcome"] == "NO_CHANGE"
    assert await _count(rt, "emotion_events") == 0
    assert await rt.emotion_service.repo.get_snapshot(agent_id, "p2") is None


async def test_process_turn_persist_false_writes_nothing(rt):
    agent_id = await _agent_id(rt)
    for _ in range(2):
        state = await rt.emotion_service.process_turn(
            agent_id,
            "p3",
            text="我成功了",
            source_event_id="evt_no_persist",
            persist=False,
        )
        assert state.emotion == "JOY"
    assert await _count(rt, "emotion_events") == 0
    assert await _count(rt, "emotion_turn_receipts") == 0
    assert await rt.emotion_service.repo.get_snapshot(agent_id, "p3") is None


async def test_crash_mid_transaction_rolls_back_everything(rt):
    agent_id = await _agent_id(rt)
    repo = rt.emotion_service.repo
    original = repo.insert_receipt

    async def boom(**kwargs):
        raise RuntimeError("crash before receipt")

    repo.insert_receipt = boom
    try:
        with pytest.raises(RuntimeError):
            await rt.emotion_service.process_turn(
                agent_id, "p4", text="我成功了", source_event_id="evt_crash"
            )
    finally:
        repo.insert_receipt = original

    # Event and snapshot were inserted before the failure inside the same
    # transaction, so the rollback must remove them too.
    assert await _count(rt, "emotion_events") == 0
    assert await _count(rt, "emotion_turn_receipts") == 0
    assert await rt.emotion_service.repo.get_snapshot(agent_id, "p4") is None


async def test_concurrent_same_turn_applies_once(rt):
    agent_id = await _agent_id(rt)

    results = await asyncio.gather(
        *[
            rt.emotion_service.process_turn(
                agent_id, "p5", text="我成功了", source_event_id="evt_race"
            )
            for _ in range(4)
        ]
    )
    assert all(state.emotion == "JOY" for state in results)
    assert await _count(rt, "emotion_events") == 1
    assert await _count(rt, "emotion_turn_receipts") == 1


async def test_concurrent_ensure_defaults_seeds_once(rt):
    agent_id = await _agent_id(rt)
    await asyncio.gather(
        *[rt.emotion_service.ensure_defaults(agent_id) for _ in range(5)]
    )
    rules = await rt.emotion_service.repo.list_rules(agent_id)
    from companion.emotion.appraisal import DEFAULT_RULES

    assert len(rules) == len(DEFAULT_RULES)
    assert len({r.name for r in rules}) == len(DEFAULT_RULES)


async def test_profile_update_uses_optimistic_revision(rt):
    agent_id = await _agent_id(rt)
    await rt.emotion_service.ensure_defaults(agent_id)
    profile = await rt.emotion_service.repo.get_profile(agent_id)

    assert await rt.emotion_service.repo.update_profile_optimistic(
        profile, expected_revision=profile.revision
    )
    # The same stale revision can no longer win.
    assert not await rt.emotion_service.repo.update_profile_optimistic(
        profile, expected_revision=profile.revision
    )


async def test_update_profile_reports_conflict_not_overwrite(rt):
    agent_id = await _agent_id(rt)
    await rt.emotion_service.ensure_defaults(agent_id)
    fresh = await rt.emotion_service.repo.get_profile(agent_id)
    stale = fresh.model_copy(update={"revision": fresh.revision - 1})
    repo = rt.emotion_service.repo
    original = repo.get_profile

    async def stale_get(agent_id_):
        return stale

    repo.get_profile = stale_get
    try:
        with pytest.raises(EmotionProfileConflict):
            await rt.emotion_service.update_profile(
                agent_id, responsiveness=0.9
            )
    finally:
        repo.get_profile = original
    # No partial write happened.
    current = await repo.get_profile(agent_id)
    assert current.responsiveness == fresh.responsiveness
