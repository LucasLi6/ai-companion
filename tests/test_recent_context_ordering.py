"""Phase 1 — ``recent_context`` must return the *latest* N events, oldest first.

Regression test for the bug where the query ordered ``occurred_at ASC … LIMIT``,
returning the *earliest* N events instead of the most recent. Events are
inserted directly with controlled timestamps so ordering is deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from companion.domain.enums import AddresseeHint, EventSource
from companion.domain.models import make_utterance_event
from companion.storage.database import Database
from companion.runtime import build_runtime

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


async def _insert(rt, conversation, i, base=BASE):
    event = make_utterance_event(
        conversation_id=conversation,
        text=f"event-{i}",
        addressee_hint=AddresseeHint.AGENT,
        source=EventSource.TEST,
        occurred_at=base + timedelta(minutes=i),
    )
    await rt.repos.utterance.save(event)
    return event


async def test_recent_context_returns_latest_n_oldest_first(rt):
    await rt.repos.conversation.get_or_create("c1")
    events = [await _insert(rt, "c1", i) for i in range(50)]

    ctx = await rt.conversations.get_recent_context("c1", exclude_event_id=None, limit=10)
    ids = [e.event_id for e in ctx]

    # The last 10 (by occurred_at), not the first 10.
    expected = [e.event_id for e in events[-10:]]
    assert ids == expected, "must return the latest 10, in ascending order"


async def test_recent_context_excludes_current_event(rt):
    await rt.repos.conversation.get_or_create("c1")
    events = [await _insert(rt, "c1", i) for i in range(20)]

    ctx = await rt.conversations.get_recent_context(
        "c1", exclude_event_id=events[-1].event_id, limit=10
    )
    ids = [e.event_id for e in ctx]
    assert events[-1].event_id not in ids
    # Without the newest event, the window is events[-11:-1].
    assert ids == [e.event_id for e in events[-11:-1]]


async def test_recent_context_isolated_to_conversation(rt):
    await rt.repos.conversation.get_or_create("c1")
    await rt.repos.conversation.get_or_create("c2")
    c1_events = [await _insert(rt, "c1", i) for i in range(10)]
    other = [await _insert(rt, "c2", i + 100) for i in range(5)]  # newer timestamps

    ctx = await rt.conversations.get_recent_context("c1", exclude_event_id=None, limit=10)
    ids = {e.event_id for e in ctx}
    assert ids == {e.event_id for e in c1_events}
    assert all(o.event_id not in ids for o in other)


async def test_recent_context_deterministic_for_equal_timestamps(rt):
    await rt.repos.conversation.get_or_create("c1")
    # A tie group of 5 events all sharing the newest timestamp.
    newest = BASE + timedelta(minutes=100)
    events = [await _insert(rt, "c1", i) for i in range(5)]
    for i in range(5, 10):
        events.append(await _insert(rt, "c1", i, base=newest))

    first = await rt.conversations.get_recent_context("c1", None, limit=10)
    second = await rt.conversations.get_recent_context("c1", None, limit=10)
    assert [e.event_id for e in first] == [e.event_id for e in second]
    # All 5 tied events are present (no misses/duplicates across the boundary).
    ids = {e.event_id for e in first}
    for e in events[5:]:
        assert e.event_id in ids


async def test_recent_context_stable_across_reopen(tmp_path):
    path = tmp_path / "reopen.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db)
    await rt.repos.conversation.get_or_create("c1")
    events = [await _insert(rt, "c1", i) for i in range(50)]
    before = await rt.conversations.get_recent_context("c1", None, limit=10)
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        after = await rt2.conversations.get_recent_context("c1", None, limit=10)
        assert [e.event_id for e in before] == [e.event_id for e in after]
    finally:
        await db2.close()
