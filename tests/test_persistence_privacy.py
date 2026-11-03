"""P0 — strict no-persistence: no conversation metadata may be written for an
``allow_persistence=False`` interaction, even inside an existing conversation.
"""

from __future__ import annotations

from companion.domain.enums import EventSource, ReportStatus
from companion.domain.models import make_utterance_event


async def test_no_persistence_writes_nothing(register, say, rt):
    p = await register("Ephemeral", processing=True, persistence=False, learning=True)
    report = await say(p.person_id, "agent", "process but forget")

    assert report.status == "COMPLETED"
    assert report.response_text is not None

    for table in (
        "utterances",
        "cognitive_analyses",
        "agent_actions",
        "memory_candidates",
        "processing_jobs",
        "conversations",
    ):
        cur = await rt.db.execute(f"SELECT COUNT(*) AS c FROM {table}")
        assert (await cur.fetchone())["c"] == 0, f"{table} must be empty"


async def test_no_persistence_does_not_touch_existing_conversation(register, say, rt):
    persistent = await register("Persistent")
    await say(persistent.person_id, "agent", "hello")  # creates "default"
    before = await rt.repos.conversation.get("default")
    assert before is not None
    assert before.last_action == "RESPOND"

    ephemeral = await register(
        "Ephemeral", processing=True, persistence=False, learning=False
    )
    report = await say(
        ephemeral.person_id, "agent", "ephemeral in existing conv",
        conversation="default",
    )
    assert report.status == "COMPLETED"
    assert report.response_text is not None

    after = await rt.repos.conversation.get("default")
    assert after.last_action == before.last_action
    assert after.updated_at == before.updated_at


async def test_no_persistence_still_responds_in_memory(register, say, rt):
    p = await register("Ephemeral", processing=True, persistence=False, learning=False)
    report = await say(p.person_id, "agent", "talk to me")
    assert report.status == ReportStatus.COMPLETED.value
    assert report.response_text is not None
    assert report.action is not None
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 0
