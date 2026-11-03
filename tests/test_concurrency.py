"""P0 — real async concurrency: shared connection, per-conversation
serialization, atomic job claim, and transaction isolation.

These tests use ``asyncio.gather`` for genuine interleaving rather than
sequential ``await``.
"""

from __future__ import annotations

import asyncio

from companion.domain.enums import EventSource, JobStatus, JobType
from companion.domain.models import make_utterance_event
from companion.jobs.worker import JobWorker
from companion.runtime import build_runtime
from companion.storage.database import Database
from companion.storage.repositories import make_repositories


async def _count(db, table) -> int:
    cur = await db.execute(f"SELECT COUNT(*) AS c FROM {table}")
    return (await cur.fetchone())["c"]


async def test_concurrent_events_across_conversations(register, rt):
    people = [await register(f"P{i}") for i in range(3)]
    to = ["agent", "human", "unknown"]

    async def submit(i):
        p = people[i % 3]
        event = make_utterance_event(
            conversation_id=f"conv-{i % 3}",
            text=f"message {i}",
            addressee_hint=to[i % 3],
            speaker_person_id=p.person_id,
            source=EventSource.TEST,
        )
        return await rt.orchestrator.process_utterance(event)

    reports = await asyncio.gather(*[submit(i) for i in range(40)])
    assert all(r.status == "COMPLETED" for r in reports)

    assert await _count(rt.db, "utterances") == 40
    assert await _count(rt.db, "cognitive_analyses") == 40
    assert await _count(rt.db, "agent_actions") == 40
    assert await _count(rt.db, "memory_candidates") == 40
    assert await _count(rt.db, "processing_jobs") == 40

    # Foreign keys and uniqueness hold after heavy concurrency.
    cur = await rt.db.execute("PRAGMA foreign_key_check")
    assert await cur.fetchall() == []
    cur = await rt.db.execute("PRAGMA integrity_check")
    assert (await cur.fetchone())[0] == "ok"


async def test_same_conversation_serialized(register, rt):
    p = await register("Speaker")

    async def submit(text):
        event = make_utterance_event(
            conversation_id="shared",
            text=text,
            addressee_hint="agent",
            speaker_person_id=p.person_id,
            source=EventSource.TEST,
        )
        return await rt.orchestrator.process_utterance(event)

    r1, r2 = await asyncio.gather(submit("first"), submit("second"))
    assert r1.status == "COMPLETED" and r2.status == "COMPLETED"
    assert await _count(rt.db, "utterances") == 2
    cur = await rt.db.execute(
        "SELECT COUNT(*) AS c FROM utterances WHERE conversation_id = 'shared'"
    )
    assert (await cur.fetchone())["c"] == 2


async def test_different_conversations_run_concurrently(register, rt):
    people = [await register(f"Q{i}") for i in range(2)]

    async def submit(conv, person, text):
        event = make_utterance_event(
            conversation_id=conv, text=text, addressee_hint="agent",
            speaker_person_id=person, source=EventSource.TEST,
        )
        return await rt.orchestrator.process_utterance(event)

    a, b = await asyncio.gather(
        submit("conv-a", people[0].person_id, "a"),
        submit("conv-b", people[1].person_id, "b"),
    )
    assert a.status == "COMPLETED" and b.status == "COMPLETED"
    assert await _count(rt.db, "utterances") == 2


async def test_two_workers_compete_for_one_job(tmp_path):
    db = await Database(tmp_path / "w.db").connect()
    await db.initialize()
    repos = make_repositories(db)
    calls = {"n": 0}

    async def handler(payload, repos, logger):
        calls["n"] += 1
        await asyncio.sleep(0.02)  # widen the race window

    handlers = {"MEMORY_CONSOLIDATION": handler}
    worker1 = JobWorker(repos, handlers)
    worker2 = JobWorker(repos, handlers)
    job_id = await repos.job.enqueue(JobType.MEMORY_CONSOLIDATION, "k", {})

    await asyncio.gather(worker1.run_once(), worker2.run_once())

    assert calls["n"] == 1
    job = await repos.job.get(job_id)
    assert job.status == JobStatus.COMPLETED
    await db.close()


async def test_two_connections_compete_for_one_job(tmp_path):
    path = tmp_path / "wc.db"
    db1 = await Database(path).connect()
    await db1.initialize()
    db2 = await Database(path).connect()
    await db2.initialize()

    repos1 = make_repositories(db1)
    repos2 = make_repositories(db2)
    job_id = await repos1.job.enqueue(JobType.MEMORY_CONSOLIDATION, "k", {})

    calls = {"n": 0}

    async def handler(payload, repos, logger):
        calls["n"] += 1
        await asyncio.sleep(0.02)

    handlers = {"MEMORY_CONSOLIDATION": handler}
    worker1 = JobWorker(repos1, handlers)
    worker2 = JobWorker(repos2, handlers)

    await asyncio.gather(worker1.run_once(), worker2.run_once())

    assert calls["n"] == 1
    job = await repos1.job.get(job_id)
    assert job.status == JobStatus.COMPLETED
    await db1.close()
    await db2.close()


async def test_rollback_does_not_affect_other_task(tmp_path):
    db = await Database(tmp_path / "tx.db").connect()
    await db.initialize()

    async def failing():
        async with db.transaction():
            await db.execute(
                "INSERT INTO conversations(conversation_id, quiet_mode, created_at,"
                " updated_at) VALUES ('tx-a', 0, '2024-01-01T00:00:00+00:00',"
                " '2024-01-01T00:00:00+00:00')"
            )
            raise RuntimeError("boom")

    async def other():
        await db.execute(
            "INSERT INTO conversations(conversation_id, quiet_mode, created_at,"
            " updated_at) VALUES ('tx-b', 0, '2024-01-01T00:00:00+00:00',"
            " '2024-01-01T00:00:00+00:00')"
        )

    results = await asyncio.gather(failing(), other(), return_exceptions=True)
    assert isinstance(results[0], RuntimeError)

    cur = await db.execute(
        "SELECT COUNT(*) AS c FROM conversations WHERE conversation_id = 'tx-a'"
    )
    assert (await cur.fetchone())["c"] == 0  # rolled back
    cur = await db.execute(
        "SELECT COUNT(*) AS c FROM conversations WHERE conversation_id = 'tx-b'"
    )
    assert (await cur.fetchone())["c"] == 1  # unaffected
    await db.close()
