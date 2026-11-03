"""B — unified memory foundation: kinds, subjects, sources, provenance,
privacy isolation, optimistic locking, human-confirmed priority, retract vs
delete, FTS search and legacy migration.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import MemoryKind, MemoryStatus, MemoryVisibility
from companion.memory.models import (
    AccessContext,
    MemoryEvidence,
    MemorySource,
    MemoryVersionConflictError,
)
from companion.storage.database import Database
from companion.storage.schema import SCHEMA_SQL

ADMIN = AccessContext("admin", is_admin=True)


def as_person(pid: str) -> AccessContext:
    return AccessContext(pid, is_admin=False, role="cognitive")


async def _search(service, access, **kw):
    return await service.search(access=access, **kw)


async def test_create_personal_fact(rt):
    svc = rt.memory_service
    p = await rt.registry.register(
        "Owner", "owner",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    item = await svc.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="正在开发一个机器人",
        access=ADMIN,
        subjects=[("PERSON", p.person_id)],
        human_confirmed=True,
        self_reported=True,
    )
    assert item.memory_id.startswith("mem_")
    assert item.version == 1

    view = await svc.get_view(item.memory_id, ADMIN)
    assert view["item"]["content"] == "正在开发一个机器人"
    assert [s["subject_id"] for s in view["subjects"]] == [p.person_id]
    assert len(view["revisions"]) == 1


async def test_create_world_knowledge_and_reflection(rt):
    svc = rt.memory_service
    k = await svc.create(
        kind=MemoryKind.WORLD_KNOWLEDGE,
        content="STM32 OTA uses a dual-bank scheme",
        access=ADMIN,
        visibility=MemoryVisibility.PUBLIC,
        source_supported=True,
    )
    r = await svc.create(
        kind=MemoryKind.REFLECTION,
        content="OTA reliability still has open questions",
        access=ADMIN,
        status=MemoryStatus.CANDIDATE,
        model_inferred=True,
    )
    results = await _search(svc, ADMIN, kinds=["WORLD_KNOWLEDGE", "REFLECTION"])
    ids = {x["memory_id"] for x in results}
    assert k.memory_id in ids
    assert r.memory_id in ids


async def test_one_memory_with_two_subjects(register, rt):
    svc = rt.memory_service
    a = await register("A")
    b = await register("B")
    item = await svc.create(
        kind=MemoryKind.RELATIONSHIP,
        content="A and B work together",
        access=ADMIN,
        subjects=[("PERSON", a.person_id), ("PERSON", b.person_id)],
    )
    view = await svc.get_view(item.memory_id, ADMIN)
    assert {s["subject_id"] for s in view["subjects"]} == {a.person_id, b.person_id}


async def test_knowledge_with_multiple_sources(rt):
    svc = rt.memory_service
    s1 = MemorySource(
        source_id="src_1", source_type="web", title="Doc A", url="https://a.example/x"
    )
    s2 = MemorySource(
        source_id="src_2", source_type="web", title="Doc B", url="https://b.example/y"
    )
    item = await svc.create(
        kind=MemoryKind.WORLD_KNOWLEDGE,
        content="verified across two sources",
        access=ADMIN,
        visibility=MemoryVisibility.PUBLIC,
        source_supported=True,
        sources=[s1, s2],
        evidence=[
            MemoryEvidence(source_id="src_1", verification_level="SUPPORTED"),
            MemoryEvidence(source_id="src_2", verification_level="SUPPORTED"),
        ],
    )
    view = await svc.get_view(item.memory_id, ADMIN)
    assert len(view["sources"]) == 2


async def test_ai_proposal_does_not_overwrite_human_confirmed(rt):
    svc = rt.memory_service
    p = await rt.registry.register(
        "Owner", "owner",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    item = await svc.create(
        kind=MemoryKind.PROJECT_STATE,
        content="机械设计完成，软件尚未完成。",
        access=ADMIN,
        subjects=[("PERSON", p.person_id)],
        human_confirmed=True,
    )

    # An AI reflection proposes a conflicting claim: it must NOT overwrite.
    ai = AccessContext("idle", is_admin=False, role="idle")
    proposal = await svc.propose(
        access=ai,
        content="整个项目已经完成。",
        conflicts_with=item.memory_id,
        kind=MemoryKind.REFLECTION,
    )
    assert proposal.status == MemoryStatus.NEEDS_REVIEW

    original = await svc.get_item(item.memory_id, ADMIN)
    assert original.content == "机械设计完成，软件尚未完成。"
    assert original.human_confirmed is True
    assert original.status == MemoryStatus.ACTIVE

    links = await svc.repo.get_links(proposal.memory_id)
    assert any(l["link_type"] == "conflicts_with" and l["linked_memory_id"] == item.memory_id for l in links)


async def test_concurrent_update_version_conflict(rt):
    svc = rt.memory_service
    item = await svc.create(kind=MemoryKind.EPISODE, content="v1", access=ADMIN)

    await svc.update(item.memory_id, access=ADMIN, expected_version=1, content="v2")
    # A second, stale update using the same expected_version is rejected.
    with pytest.raises(MemoryVersionConflictError):
        await svc.update(item.memory_id, access=ADMIN, expected_version=1, content="v3")

    current = await svc.get_item(item.memory_id, ADMIN)
    assert current.content == "v2"
    assert current.version == 2


async def test_private_memories_do_not_leak(register, rt):
    svc = rt.memory_service
    a = await register("A")
    b = await register("B")
    await svc.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="A 的私人信息",
        access=ADMIN,
        subjects=[("PERSON", a.person_id)],
        visibility=MemoryVisibility.PRIVATE,
    )
    await svc.create(
        kind=MemoryKind.PERSONAL_FACT,
        content="B 的私人信息",
        access=ADMIN,
        subjects=[("PERSON", b.person_id)],
        visibility=MemoryVisibility.PRIVATE,
    )

    a_sees = await _search(svc, as_person(a.person_id))
    a_contents = {x["content"] for x in a_sees}
    assert "A 的私人信息" in a_contents
    assert "B 的私人信息" not in a_contents

    b_sees = await _search(svc, as_person(b.person_id))
    b_contents = {x["content"] for x in b_sees}
    assert "B 的私人信息" in b_contents
    assert "A 的私人信息" not in b_contents


async def test_update_immediately_affects_search(rt):
    svc = rt.memory_service
    item = await svc.create(kind=MemoryKind.WORLD_KNOWLEDGE, content="alpha", access=ADMIN)
    assert any(x["memory_id"] == item.memory_id for x in await _search(svc, ADMIN, query="alpha"))

    await svc.update(item.memory_id, access=ADMIN, expected_version=1, content="beta")
    assert not any(x["memory_id"] == item.memory_id for x in await _search(svc, ADMIN, query="alpha"))
    assert any(x["memory_id"] == item.memory_id for x in await _search(svc, ADMIN, query="beta"))


async def test_retract_excluded_from_default_search(rt):
    svc = rt.memory_service
    item = await svc.create(kind=MemoryKind.PERSONAL_FACT, content="soon obsolete", access=ADMIN)
    await svc.retract(item.memory_id, access=ADMIN)

    assert not any(x["memory_id"] == item.memory_id for x in await _search(svc, ADMIN))
    # Still present when retracted items are explicitly included.
    assert any(
        x["memory_id"] == item.memory_id
        for x in await _search(svc, ADMIN, include_retracted=True)
    )


async def test_delete_removes_from_index(rt):
    svc = rt.memory_service
    item = await svc.create(
        kind=MemoryKind.PERSONAL_FACT, content="private secret", access=ADMIN
    )
    mid = item.memory_id
    await svc.delete(mid, access=ADMIN)

    assert await svc.get_item(mid, ADMIN) is None
    assert not any(x["memory_id"] == mid for x in await _search(svc, ADMIN))
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM memory_items_fts WHERE memory_id = ?", (mid,))
    assert (await cur.fetchone())["c"] == 0


async def test_legacy_memory_facts_migrate_into_unified_store(tmp_path):
    """An old V0.0 database with hand-authored facts is searchable after open."""
    path = tmp_path / "old.db"
    db = await Database(path).connect()
    await db.conn.executescript(SCHEMA_SQL)
    await db.conn.execute(
        "INSERT INTO schema_version(version, applied_at) VALUES (1, '2024-01-01T00:00:00+00:00')"
    )
    await db.conn.execute(
        "INSERT INTO persons(person_id, display_name, role, registration_status, created_at, updated_at)"
        " VALUES ('person_old', 'Old', 'friend', 'active', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    await db.conn.execute(
        "INSERT INTO memory_facts(fact_id, subject_person_id, content, status, created_at, updated_at)"
        " VALUES ('fact_1', 'person_old', 'legacy fact content', 'manual',"
        " '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')"
    )
    await db.close()

    from companion.runtime import build_runtime

    db2 = await Database(path).connect()
    await db2.initialize()
    rt2 = build_runtime(db2)
    try:
        results = await rt2.memory_service.search(access=ADMIN, query="legacy")
        assert any(x["memory_id"] == "fact_1" for x in results)
        # Old event/person ids preserved; subject migrated.
        view = await rt2.memory_service.get_view("fact_1", ADMIN)
        assert [s["subject_id"] for s in view["subjects"]] == ["person_old"]
    finally:
        await db2.close()
