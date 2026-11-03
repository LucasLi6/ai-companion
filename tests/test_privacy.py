"""Test H: unauthorised persons are denied; consent flags are honoured."""

from __future__ import annotations

from companion.domain.enums import ReportStatus


async def test_unauthorized_person_is_denied(register, say, rt):
    # allow_processing=False → the default for this person.
    denied = await register("Stranger", "guest", processing=False)
    report = await say(denied.person_id, "agent", "please record me")

    assert report.status == ReportStatus.PERMISSION_DENIED.value

    # No persistent content, no analysis, no action, no candidates.
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 0
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM cognitive_analyses")
    assert (await cur.fetchone())["c"] == 0
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM agent_actions")
    assert (await cur.fetchone())["c"] == 0
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM memory_candidates")
    assert (await cur.fetchone())["c"] == 0
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM processing_jobs")
    assert (await cur.fetchone())["c"] == 0


async def test_unregistered_speaker_is_denied(say, rt):
    report = await say(None, "agent", "I have no identity")
    assert report.status == ReportStatus.PERMISSION_DENIED.value
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 0


async def test_unknown_person_id_is_denied(say, rt):
    report = await say("person_does_not_exist", "agent", "hello")
    assert report.status == ReportStatus.PERMISSION_DENIED.value
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 0


async def test_processing_without_persistence_runs_but_does_not_store(register, say, rt):
    # allow_processing=True but allow_persistence=False: cognition runs and a
    # response is produced, but nothing is written to disk.
    p = await register("Ephemeral", processing=True, persistence=False, learning=False)
    report = await say(p.person_id, "agent", "process but forget")
    assert report.status == "COMPLETED"
    assert report.response_text is not None

    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 0
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM cognitive_analyses")
    assert (await cur.fetchone())["c"] == 0


async def test_learning_disabled_creates_no_candidates(register, say, rt):
    p = await register("NoLearn", processing=True, persistence=True, learning=False)
    report = await say(p.person_id, "agent", "do not learn from me")
    assert report.status == "COMPLETED"
    candidates = await rt.repos.analysis.get_candidates(report.event_id)
    assert candidates == []
    job = await rt.repos.job.get_by_key(f"memory:{report.event_id}")
    assert job is None
