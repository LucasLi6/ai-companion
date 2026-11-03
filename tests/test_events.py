"""Test B: every allowed utterance yields a decision.

Also Test D: idempotent re-submission and conflict detection.
"""

from __future__ import annotations

from companion.domain.enums import ProcessingStatus, ReportStatus


async def test_each_utterance_has_a_decision(register, say, rt):
    p = await register("Speaker")
    r1 = await say(p.person_id, "agent", "hello agent")
    r2 = await say(p.person_id, "human", "hello human")
    r3 = await say(p.person_id, "unknown", "hello nobody")

    for r in (r1, r2, r3):
        assert r.status == "COMPLETED"
        assert r.action is not None
        # every completed event has an analysis and a final action
        analysis = await rt.repos.analysis.get_analysis(r.event_id)
        action = await rt.repos.analysis.get_action(r.event_id)
        assert analysis is not None
        assert action is not None

    assert r1.action.value == "RESPOND"
    assert r2.action.value == "IGNORE"
    assert r3.action.value == "WAIT"


async def test_same_text_different_context_gives_different_decision(register, say):
    p = await register("Speaker")
    text = "我在做一个新机器人。"
    to_agent = await say(p.person_id, "agent", text)
    to_human = await say(p.person_id, "human", text)
    assert to_agent.action.value == "RESPOND"
    assert to_human.action.value == "IGNORE"


async def test_resubmission_is_idempotent(register, say, rt):
    p = await register("Speaker")
    first = await say(p.person_id, "agent", "unique sentence")
    # Re-submit the exact same event (same id and content).
    from companion.domain.models import make_utterance_event
    from companion.domain.enums import EventSource

    event = make_utterance_event(
        conversation_id="default",
        text="unique sentence",
        addressee_hint="agent",
        speaker_person_id=p.person_id,
        source=EventSource.TEST,
        event_id=first.event_id,
    )
    second = await rt.orchestrator.process_utterance(event)
    assert second.status == ReportStatus.DUPLICATE.value

    # No duplicate rows.
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS c FROM utterances WHERE event_id = ?", (first.event_id,)
    )
    assert (await cur.fetchone())["c"] == 1
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS c FROM cognitive_analyses WHERE event_id = ?", (first.event_id,)
    )
    assert (await cur.fetchone())["c"] == 1
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS c FROM agent_actions WHERE event_id = ?", (first.event_id,)
    )
    assert (await cur.fetchone())["c"] == 1


async def test_conflicting_content_is_rejected(register, say, rt):
    p = await register("Speaker")
    first = await say(p.person_id, "agent", "original content")

    from companion.domain.models import make_utterance_event
    from companion.domain.enums import EventSource

    event = make_utterance_event(
        conversation_id="default",
        text="different content",
        addressee_hint="agent",
        speaker_person_id=p.person_id,
        source=EventSource.TEST,
        event_id=first.event_id,
    )
    result = await rt.orchestrator.process_utterance(event)
    assert result.status == ReportStatus.CONFLICT.value

    # The stored content was not silently overwritten.
    stored = await rt.repos.utterance.get(first.event_id)
    assert stored.text == "original content"
