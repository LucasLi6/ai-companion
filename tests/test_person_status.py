"""P1 — person registration status and consent govern processing.

``ACTIVE`` + consent is required; ``SUSPENDED`` and ``REVOKED`` deny new input
regardless of prior consent; ``owner`` role does not bypass consent; and a
pending memory job re-checks authorisation before consolidating.
"""

from __future__ import annotations

from companion.domain.enums import (
    EventSource,
    JobStatus,
    MemoryCandidateStatus,
    RegistrationStatus,
    ReportStatus,
)
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.runtime import build_runtime
from companion.storage.database import Database


async def test_suspended_denies_processing(register, say, rt):
    p = await register("Suspended")  # all consent True
    await rt.registry.set_status(p.person_id, RegistrationStatus.SUSPENDED)
    report = await say(p.person_id, "agent", "hello")
    assert report.status == ReportStatus.PERMISSION_DENIED.value
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 0


async def test_revoked_denies_processing(register, say, rt):
    p = await register("Revoked")
    await rt.registry.set_status(p.person_id, RegistrationStatus.REVOKED)
    report = await say(p.person_id, "agent", "hello")
    assert report.status == ReportStatus.PERMISSION_DENIED.value
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 0


async def test_owner_role_does_not_bypass_consent(register, say, rt):
    p = await register(
        "Owner", "owner", processing=False, persistence=False, learning=False
    )
    report = await say(p.person_id, "agent", "hello")
    assert report.status == ReportStatus.PERMISSION_DENIED.value
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 0


async def test_active_allowed_after_re_activation(register, say, rt):
    p = await register("ReActivate")
    await rt.registry.set_status(p.person_id, RegistrationStatus.SUSPENDED)
    denied = await say(p.person_id, "agent", "hello")
    assert denied.status == ReportStatus.PERMISSION_DENIED.value
    await rt.registry.set_status(p.person_id, RegistrationStatus.ACTIVE)
    allowed = await say(p.person_id, "agent", "hello again")
    assert allowed.status == ReportStatus.COMPLETED.value


async def test_revoke_learning_after_job_queued_skips_consolidation(tmp_path):
    db = await Database(tmp_path / "l.db").connect()
    await db.initialize()
    rt = build_runtime(db)
    p = await rt.registry.register(
        "Learner", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    event = make_utterance_event(
        conversation_id="c1", text="learn about me", addressee_hint="agent",
        speaker_person_id=p.person_id, source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    job = await rt.repos.job.get_by_key(f"memory:{event.event_id}")
    assert job.status == JobStatus.PENDING

    # Revoke learning consent *after* the job was queued.
    await rt.registry.set_consent(
        Consent(
            person_id=p.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            updated_at=utcnow(),
        )
    )

    await rt.worker.run_once()
    job = await rt.repos.job.get(job.job_id)
    assert job.status == JobStatus.COMPLETED  # job finishes, but skips learning

    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert candidates
    assert all(c.status == MemoryCandidateStatus.SKIPPED for c in candidates)

    # And no new candidate may be produced for new content.
    event2 = make_utterance_event(
        conversation_id="c1", text="still learning?", addressee_hint="agent",
        speaker_person_id=p.person_id, source=EventSource.TEST,
    )
    report2 = await rt.orchestrator.process_utterance(event2)
    assert report2.status == "COMPLETED"
    assert await rt.repos.analysis.get_candidates(event2.event_id) == []
    assert await rt.repos.job.get_by_key(f"memory:{event2.event_id}") is None
    await db.close()


async def test_revoked_person_pending_job_is_skipped(tmp_path):
    db = await Database(tmp_path / "r.db").connect()
    await db.initialize()
    rt = build_runtime(db)
    p = await rt.registry.register(
        "RevokedLearner", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    event = make_utterance_event(
        conversation_id="c1", text="learn", addressee_hint="agent",
        speaker_person_id=p.person_id, source=EventSource.TEST,
    )
    await rt.orchestrator.process_utterance(event)
    job = await rt.repos.job.get_by_key(f"memory:{event.event_id}")
    assert job.status == JobStatus.PENDING

    await rt.registry.set_status(p.person_id, RegistrationStatus.REVOKED)
    await rt.worker.run_once()

    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert all(c.status == MemoryCandidateStatus.SKIPPED for c in candidates)
    await db.close()
