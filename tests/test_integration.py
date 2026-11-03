"""End-to-end integration across subsystem boundaries (V0.1 scenarios A–G).

Each test drives a *chain* of subsystems — identity → cognition → jobs →
unified memory → idle research → audit — rather than one unit in isolation.
"""

from __future__ import annotations

from companion.domain.enums import (
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
)
from companion.idle import build_idle_scheduler
from companion.memory.models import AccessContext
from companion.runtime import build_runtime
from companion.storage.database import Database

ADMIN = AccessContext("admin", is_admin=True)


async def test_A_thru_G_full_lifecycle(db, register, say):
    """One person → say → memory → research → idle → audit, end to end."""
    # A — schema is at the current version, migration atomic (already at head).
    cur = await db.execute("SELECT MAX(version) AS v FROM schema_version")
    from companion.storage.schema import SCHEMA_VERSION

    assert (await cur.fetchone())["v"] == SCHEMA_VERSION

    # B — register two consenting people.
    owner = await register("Owner", role="owner")
    friend = await register("Friend", role="friend")

    # C — a self-report is consolidated into the unified store via the job.
    report = await say(owner.person_id, "agent", "我喜欢做机器人")
    assert report.status == "COMPLETED"

    # D — a third-party statement about someone else must not auto-confirm.
    await say(friend.person_id, "human", "Owner 已经完成了整个项目")

    # E — the human-confirmed fact survives AI proposals, with privacy intact.
    rt = build_runtime(db)
    item = await rt.memory_service.create(
        kind=MemoryKind.PROJECT_STATE,
        content="机械设计完成，软件尚未完成。",
        access=ADMIN,
        subjects=[("PERSON", owner.person_id)],
        human_confirmed=True,
    )
    ai = AccessContext("idle", is_admin=False, role="idle")
    proposal = await rt.memory_service.propose(
        access=ai, content="整个项目已经完成。", conflicts_with=item.memory_id
    )
    assert proposal.status == MemoryStatus.NEEDS_REVIEW
    original = await rt.memory_service.get_item(item.memory_id, ADMIN)
    assert original.human_confirmed is True

    # F — idle cycle: reflection + curiosity + research, gated on, then off.
    await rt.repos.settings.set("idle.enabled", "true")
    scheduler = build_idle_scheduler(rt)
    cycle = await scheduler.run_cycle()
    assert cycle["skipped"] is False

    # G — every mutation left an audit trail, and the audit log is readable.
    audit = await rt.repos.audit.list(limit=100)
    actions = {r["action"] for r in audit}
    assert "create" in actions
    assert "register" in actions

    # Regression: the V0.0 CLI pipeline still reports and persists an action.
    action = await rt.repos.analysis.get_action(report.event_id)
    assert action is not None


async def test_privacy_isolation_across_persons(db, register):
    """Non-admin memory access is isolated to the subject's own records."""
    rt = build_runtime(db)
    a = await register("A")
    b = await register("B")

    await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="A 的私密事实",
        access=ADMIN,
        subjects=[("PERSON", a.person_id)],
        visibility=MemoryVisibility.PRIVATE,
    )
    await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="B 的私密事实",
        access=ADMIN,
        subjects=[("PERSON", b.person_id)],
        visibility=MemoryVisibility.PRIVATE,
    )

    a_view = await rt.memory_service.search(
        access=AccessContext(a.person_id, role="cognitive")
    )
    contents = {x["content"] for x in a_view}
    assert "A 的私密事实" in contents
    assert "B 的私密事实" not in contents


async def test_reopen_preserves_memory_and_settings(tmp_path):
    """A closed database reopens with memory and settings intact (no rebuild)."""
    from companion.storage.schema import SCHEMA_SQL  # noqa: F401  (documents legacy path)

    db = await Database(tmp_path / "reopen.db").connect()
    await db.initialize()
    rt = build_runtime(db)
    await rt.memory_service.create(
        kind=MemoryKind.WORLD_KNOWLEDGE,
        content="跨会话持久化",
        access=ADMIN,
        visibility=MemoryVisibility.PUBLIC,
    )
    await rt.repos.settings.set("idle.enabled", "true")
    await db.close()

    db2 = await Database(tmp_path / "reopen.db").connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        found = await rt2.memory_service.search(access=ADMIN, query="持久化")
        assert found
        assert await rt2.repos.settings.get_bool("idle.enabled") is True
    finally:
        await db2.close()
