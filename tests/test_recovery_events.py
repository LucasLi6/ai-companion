"""P1 — event recovery after a crash at each durable-stage boundary.

Recovery re-submits a leftover ``PROCESSING`` event through the orchestrator,
which resumes from whatever durable stages already exist — it never blindly
resets and re-runs the whole pipeline.
"""

from __future__ import annotations

from companion.domain.enums import EventSource, ProcessingStatus
from companion.domain.models import make_utterance_event
from companion.recovery import RecoveryManager
from companion.runtime import build_runtime
from companion.storage.database import Database


class FailingCognitive:
    async def analyze(self, event, context, memories):
        raise RuntimeError("cognitive boom")


class FailingOnceResponse:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, event, analysis, context) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("response boom")
        return "ok"


class CountingResponse:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, event, analysis, context) -> str:
        self.calls += 1
        return "recovered"


async def _count(db, table) -> int:
    cur = await db.execute(f"SELECT COUNT(*) AS c FROM {table}")
    return (await cur.fetchone())["c"]


async def _setup_person(rt):
    return await rt.registry.register(
        "Speaker", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )


async def _event(person_id, event_id="evt_crash"):
    return make_utterance_event(
        conversation_id="c1", text="hello", addressee_hint="agent",
        speaker_person_id=person_id, source=EventSource.TEST, event_id=event_id,
    )


async def test_recovery_after_crash_before_analysis(tmp_path):
    path = tmp_path / "p1.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db, cognitive=FailingCognitive())
    p = await _setup_person(rt)
    event = await _event(p.person_id)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "FAILED"
    # Crash happens right after persistence, before analysis: leave PROCESSING.
    await rt.repos.utterance.update_status(event.event_id, ProcessingStatus.PROCESSING)
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        recovered = await RecoveryManager(rt2.repos, rt2.orchestrator).retry(event.event_id)
        assert recovered.status == "COMPLETED"
        assert await _count(db2, "cognitive_analyses") == 1
        assert await _count(db2, "agent_actions") == 1
        stored = await rt2.repos.utterance.get(event.event_id)
        assert stored.processing_status == ProcessingStatus.COMPLETED
    finally:
        await db2.close()


async def test_recovery_after_analysis_and_job_before_response(tmp_path):
    path = tmp_path / "p2.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db, response=FailingOnceResponse())
    p = await _setup_person(rt)
    event = await _event(p.person_id)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "FAILED"
    # Analysis + candidate + job are already durable; crash before response.
    assert await _count(db, "cognitive_analyses") == 1
    assert await _count(db, "memory_candidates") == 1
    assert await _count(db, "processing_jobs") == 1
    assert await _count(db, "agent_actions") == 0
    await rt.repos.utterance.update_status(event.event_id, ProcessingStatus.PROCESSING)
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        recovered = await RecoveryManager(rt2.repos, rt2.orchestrator).retry(event.event_id)
        assert recovered.status == "COMPLETED"
        # Analysis was reused, not recomputed; exactly one of each durable record.
        assert await _count(db2, "cognitive_analyses") == 1
        assert await _count(db2, "memory_candidates") == 1
        assert await _count(db2, "processing_jobs") == 1
        assert await _count(db2, "agent_actions") == 1
        stored = await rt2.repos.utterance.get(event.event_id)
        assert stored.processing_status == ProcessingStatus.COMPLETED
    finally:
        await db2.close()


async def test_recovery_after_action_saved_before_completed(tmp_path):
    path = tmp_path / "p3.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db)
    p = await _setup_person(rt)
    event = await _event(p.person_id)
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    # Crash right after the action was saved, before the status commit.
    await rt.repos.utterance.update_status(event.event_id, ProcessingStatus.PROCESSING)
    await db.close()

    db2 = await Database(path).connect()
    await db2.initialize()
    response = CountingResponse()
    rt2 = build_runtime(db2, response=response)
    try:
        recovered = await RecoveryManager(rt2.repos, rt2.orchestrator).retry(event.event_id)
        assert recovered.status == "COMPLETED"
        # The already-persisted action is reused: no duplicate, no re-generated
        # response, no duplicate memory side effects.
        assert await _count(db2, "agent_actions") == 1
        assert await _count(db2, "cognitive_analyses") == 1
        assert response.calls == 0
        stored = await rt2.repos.utterance.get(event.event_id)
        assert stored.processing_status == ProcessingStatus.COMPLETED
    finally:
        await db2.close()


async def test_recovery_list_finds_unfinished(tmp_path):
    path = tmp_path / "l.db"
    db = await Database(path).connect()
    await db.initialize()
    rt = build_runtime(db, cognitive=FailingCognitive())
    p = await _setup_person(rt)
    event = await _event(p.person_id, event_id="evt_unfinished")
    await rt.orchestrator.process_utterance(event)  # FAILED
    await rt.repos.utterance.update_status(event.event_id, ProcessingStatus.PROCESSING)

    unfinished = await RecoveryManager(rt.repos, rt.orchestrator).list_unfinished()
    ids = {e.event_id for e in unfinished}
    assert "evt_unfinished" in ids
    await db.close()
