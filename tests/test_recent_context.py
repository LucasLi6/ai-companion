"""A5 — the current event must never also appear in its own history context.

The current utterance is persisted before context is assembled, so a future
cognitive engine assembling a chat history must exclude the current ``event_id``.
"""

from __future__ import annotations


async def test_recent_context_excludes_current_event(register, say, rt):
    p = await register("Speaker")
    r1 = await say(p.person_id, "agent", "first")
    r2 = await say(p.person_id, "agent", "second")

    ctx = await rt.conversations.get_recent_context(
        "default", exclude_event_id=r2.event_id
    )
    ids = [e.event_id for e in ctx]
    assert r2.event_id not in ids
    assert r1.event_id in ids
    assert ids == [r1.event_id]  # oldest first


async def test_recent_context_without_exclusion_includes_all(register, say, rt):
    p = await register("Speaker")
    r1 = await say(p.person_id, "agent", "first")
    r2 = await say(p.person_id, "agent", "second")

    ctx = await rt.conversations.get_recent_context("default", exclude_event_id=None)
    ids = [e.event_id for e in ctx]
    assert ids == [r1.event_id, r2.event_id]
