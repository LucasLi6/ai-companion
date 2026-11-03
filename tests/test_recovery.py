"""Test E (restart persistence), G (provider exception), I (quiet mode)."""

from __future__ import annotations

from companion.domain.enums import CognitiveAction, EventSource, ProcessingStatus
from companion.domain.models import make_utterance_event
from companion.runtime import build_runtime
from companion.storage.database import Database


class FailingCognitive:
    async def analyze(self, event, context, memories):
        raise RuntimeError("cognitive boom")


async def test_db_restart_preserves_state(tmp_path):
    path = tmp_path / "e.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db)
    p = await rt.registry.register(
        "Owner", "owner",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    event = make_utterance_event(
        conversation_id="default",
        text="persist me",
        addressee_hint="agent",
        speaker_person_id=p.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    await rt.conversations.set_quiet_mode("default", True)
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        # Person, utterance, decision, and quiet mode all survive the restart.
        assert await rt2.registry.get(p.person_id) is not None
        assert await rt2.repos.utterance.get(event.event_id) is not None
        assert await rt2.repos.analysis.get_action(event.event_id) is not None
        ctx = await rt2.repos.conversation.get("default")
        assert ctx.quiet_mode is True
    finally:
        await db2.close()


async def test_provider_exception_is_retryable(db):
    rt = build_runtime(db)
    p = await rt.registry.register(
        "Speaker", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    event = make_utterance_event(
        conversation_id="default",
        text="trigger failure",
        addressee_hint="agent",
        speaker_person_id=p.person_id,
        source=EventSource.TEST,
    )

    failing = build_runtime(db, cognitive=FailingCognitive())
    report = await failing.orchestrator.process_utterance(event)
    assert report.status == "FAILED"
    assert report.action is None

    # Event is kept in a retryable state, not marked complete, with no
    # analysis/action written.
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status == ProcessingStatus.FAILED
    assert await rt.repos.analysis.get_analysis(event.event_id) is None
    assert await rt.repos.analysis.get_action(event.event_id) is None

    # Retrying the same event with a working provider succeeds.
    retry_report = await rt.orchestrator.process_utterance(event)
    assert retry_report.status == "COMPLETED"
    assert retry_report.action == CognitiveAction.RESPOND
    assert await rt.repos.analysis.get_analysis(event.event_id) is not None


async def test_quiet_mode_suppresses_response(register, say, rt):
    p = await register("Speaker")
    await rt.conversations.set_quiet_mode("default", True)
    report = await say(p.person_id, "agent", "speak in quiet mode")

    # Event is still fully processed per its permissions...
    assert report.status == "COMPLETED"
    assert await rt.repos.utterance.get(report.event_id) is not None
    # ...but no normal response is produced.
    assert report.action == CognitiveAction.WAIT
    assert report.response_text is None

    # Turning quiet mode off restores normal responses.
    await rt.conversations.set_quiet_mode("default", False)
    report2 = await say(p.person_id, "agent", "speak normally now")
    assert report2.action == CognitiveAction.RESPOND
    assert report2.response_text is not None


async def test_quiet_mode_persists_across_restart(tmp_path):
    path = tmp_path / "q.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db)
    await rt.conversations.set_quiet_mode("default", True)
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        ctx = await rt2.repos.conversation.get("default")
        assert ctx.quiet_mode is True
    finally:
        await db2.close()
