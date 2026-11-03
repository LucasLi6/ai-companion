"""A4 — multi-person learning authorisation.

A candidate describes a *subject*, but its *source* is the speaker. The
consolidation job must check both the source's authorisation and the subject's
privacy, and must never turn a third-party statement (speaker != subject) into
a confirmed fact about the subject.
"""

from __future__ import annotations

from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    JobStatus,
    MemoryCandidateStatus,
    MemoryOperation,
    MemoryStatus,
    ReasonCode,
)
from companion.domain.models import (
    CognitiveResult,
    Consent,
    MemoryCandidate,
    make_utterance_event,
    new_id,
    utcnow,
)
from companion.runtime import build_runtime
from companion.storage.database import Database


class ThirdPartyCognitive:
    """Emits one candidate describing ``subject_id`` (a third party)."""

    def __init__(self, subject_id: str) -> None:
        self.subject_id = subject_id

    async def analyze(self, event, context, memories) -> CognitiveResult:
        return CognitiveResult(
            event_id=event.event_id,
            meaning="third-party statement about another person",
            action=CognitiveAction.IGNORE,
            reason_code=ReasonCode.DIRECT_HUMAN_ADDRESS,
            memory_candidates=[
                MemoryCandidate(
                    candidate_id=new_id("memc"),
                    source_event_id=event.event_id,
                    subject_person_id=self.subject_id,
                    content="B is working on a robotics project",
                    operation=MemoryOperation.ADD,
                )
            ],
        )


async def _setup(tmp_path, subject_id):
    db = await Database(tmp_path / "t.db").connect()
    await db.initialize()
    rt = build_runtime(db, cognitive=ThirdPartyCognitive(subject_id))
    return db, rt


async def _run(rt, speaker_id, conversation="c1"):
    event = make_utterance_event(
        conversation_id=conversation,
        text="B is working on a robotics project",
        addressee_hint="human",
        speaker_person_id=speaker_id,
        source=EventSource.TEST,
    )
    return await rt.orchestrator.process_utterance(event)


async def test_third_party_report_is_not_confirmed(tmp_path):
    db, rt = await _setup(tmp_path, "PLACEHOLDER")
    # register A (speaker) and B (subject) with full learning consent
    a = await rt.registry.register(
        "A", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    b = await rt.registry.register(
        "B", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    rt.cognitive.subject_id = b.person_id

    report = await _run(rt, a.person_id)
    assert report.status == "COMPLETED"
    job = await rt.repos.job.get_by_key(f"memory:{report.event_id}")
    assert job.status == JobStatus.PENDING

    await rt.worker.run_once()
    candidates = await rt.repos.analysis.get_candidates(report.event_id)
    assert len(candidates) == 1
    # The report is preserved with attribution, but never becomes a confirmed
    # (ACTIVE) fact about B.
    assert candidates[0].status == MemoryCandidateStatus.CONSOLIDATED
    write = await rt.repos.analysis.get_candidate_write(candidates[0].candidate_id)
    assert write["outcome"] == "SAVED"

    item = await rt.memory_service.repo.get(write["memory_id"])
    assert item is not None
    assert item.status != MemoryStatus.ACTIVE
    assert item.third_party_reported is True
    assert item.human_confirmed is False
    # B is not silently enrolled: no subject link, no assignment.
    assert write and await rt.memory_service.repo.list_for_subject(
        b.person_id, statuses=[MemoryStatus.ACTIVE.value]
    ) == []
    await db.close()


async def test_source_revocation_skips_queued_job(tmp_path):
    db, rt = await _setup(tmp_path, "PLACEHOLDER")
    a = await rt.registry.register(
        "A", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    b = await rt.registry.register(
        "B", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    rt.cognitive.subject_id = b.person_id

    report = await _run(rt, a.person_id)
    # Revoke the *source* speaker's learning consent after queuing.
    await rt.registry.set_consent(
        Consent(
            person_id=a.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            updated_at=utcnow(),
        )
    )
    await rt.worker.run_once()
    candidates = await rt.repos.analysis.get_candidates(report.event_id)
    assert candidates[0].status == MemoryCandidateStatus.SKIPPED
    await db.close()


async def test_subject_privacy_denial_skips_queued_job(tmp_path):
    db, rt = await _setup(tmp_path, "PLACEHOLDER")
    a = await rt.registry.register(
        "A", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    b = await rt.registry.register(
        "B", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    rt.cognitive.subject_id = b.person_id

    report = await _run(rt, a.person_id)
    # Revoke the *subject's* learning consent after queuing.
    await rt.registry.set_consent(
        Consent(
            person_id=b.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            updated_at=utcnow(),
        )
    )
    await rt.worker.run_once()
    candidates = await rt.repos.analysis.get_candidates(report.event_id)
    assert candidates[0].status == MemoryCandidateStatus.SKIPPED
    await db.close()
