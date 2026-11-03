"""P0 — response failure must not lose the analysis or strand the memory job.

Regression tests for the retry/resume semantics of
``Orchestrator.process_utterance``: a failed ``ResponseProvider`` keeps the
already-persisted analysis, candidates and memory job, and a retry resumes
rather than clears and re-runs them.
"""

from __future__ import annotations

from companion.domain.enums import (
    EventSource,
    JobStatus,
    MemoryCandidateStatus,
    ProcessingStatus,
)
from companion.domain.models import make_utterance_event
from companion.runtime import build_runtime
from companion.storage.database import Database


class FailingOnceResponseProvider:
    """Fails the first ``generate`` call, then succeeds."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, event, analysis, context) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("response boom")
        return "ok response"


async def _count(db, table, where="") -> int:
    cur = await db.execute(f"SELECT COUNT(*) AS c FROM {table} {where}")
    return (await cur.fetchone())["c"]


async def test_response_failure_retry_preserves_analysis_and_job(tmp_path):
    db = await Database(tmp_path / "r.db").connect()
    await db.initialize()
    provider = FailingOnceResponseProvider()
    rt = build_runtime(db, response=provider)
    p = await rt.registry.register(
        "Speaker", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="hello",
        addressee_hint="agent",
        speaker_person_id=p.person_id,
        source=EventSource.TEST,
    )

    # 1st attempt: analysis + candidate + job persist, response fails.
    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "FAILED"

    assert await rt.repos.analysis.get_analysis(event.event_id) is not None
    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert len(candidates) == 1
    job = await rt.repos.job.get_by_key(f"memory:{event.event_id}")
    assert job is not None and job.status == JobStatus.PENDING

    # The worker runs the already-queued job (candidate -> CONSOLIDATED).
    await rt.worker.run_once()
    job = await rt.repos.job.get(job.job_id)
    assert job.status == JobStatus.COMPLETED
    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert all(c.status == MemoryCandidateStatus.CONSOLIDATED for c in candidates)

    # Retry the same event id + content.
    retry = make_utterance_event(
        conversation_id="c1",
        text="hello",
        addressee_hint="agent",
        speaker_person_id=p.person_id,
        source=EventSource.TEST,
        event_id=event.event_id,
    )
    second = await rt.orchestrator.process_utterance(retry)
    assert second.status == "COMPLETED"
    assert provider.calls == 2  # failed once, succeeded once

    # Exactly one analysis, one action, one candidate, one job. No strand.
    assert await _count(db, "cognitive_analyses") == 1
    assert await _count(db, "agent_actions") == 1
    assert await _count(db, "memory_candidates") == 1
    assert await _count(db, "processing_jobs") == 1

    stored = await rt.repos.utterance.get(event.event_id)
    assert stored.processing_status == ProcessingStatus.COMPLETED
    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert all(c.status == MemoryCandidateStatus.CONSOLIDATED for c in candidates)

    await db.close()


async def test_retry_after_completed_job_does_not_duplicate_candidates(tmp_path):
    """A completed memory job must not cause new candidates to go unprocessed."""
    db = await Database(tmp_path / "s.db").connect()
    await db.initialize()
    provider = FailingOnceResponseProvider()
    rt = build_runtime(db, response=provider)
    p = await rt.registry.register(
        "Speaker", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    event = make_utterance_event(
        conversation_id="c1", text="hi", addressee_hint="agent",
        speaker_person_id=p.person_id, source=EventSource.TEST,
    )
    await rt.orchestrator.process_utterance(event)  # FAILED at response
    await rt.worker.run_once()  # job COMPLETED, candidate CONSOLIDATED

    retry = event.model_copy(update={"processing_status": ProcessingStatus.PENDING})
    report = await rt.orchestrator.process_utterance(retry)
    assert report.status == "COMPLETED"

    assert await _count(db, "memory_candidates") == 1
    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert all(c.status == MemoryCandidateStatus.CONSOLIDATED for c in candidates)
    job = await rt.repos.job.get_by_key(f"memory:{event.event_id}")
    assert job.status == JobStatus.COMPLETED
    await db.close()


async def test_failure_during_cognitive_is_retryable_without_duplicates(tmp_path):
    """Cognitive failure leaves no analysis; retry recomputes exactly once."""

    class FailingOnceCognitive:
        def __init__(self) -> None:
            self.calls = 0

        async def analyze(self, event, context, memories):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("cognitive boom")
            from companion.domain.enums import CognitiveAction, ReasonCode
            from companion.domain.models import CognitiveResult

            return CognitiveResult(
                event_id=event.event_id,
                meaning="ok",
                action=CognitiveAction.RESPOND,
                reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
            )

    db = await Database(tmp_path / "c.db").connect()
    await db.initialize()
    cognitive = FailingOnceCognitive()
    rt = build_runtime(db, cognitive=cognitive)
    p = await rt.registry.register(
        "Speaker", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    event = make_utterance_event(
        conversation_id="c1", text="hi", addressee_hint="agent",
        speaker_person_id=p.person_id, source=EventSource.TEST,
    )
    first = await rt.orchestrator.process_utterance(event)
    assert first.status == "FAILED"
    assert cognitive.calls == 1
    assert await _count(db, "cognitive_analyses") == 0

    second = await rt.orchestrator.process_utterance(
        event.model_copy(update={"processing_status": ProcessingStatus.PENDING})
    )
    assert second.status == "COMPLETED"
    assert cognitive.calls == 2
    assert await _count(db, "cognitive_analyses") == 1
    assert await _count(db, "agent_actions") == 1
    await db.close()
