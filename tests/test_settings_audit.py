"""Settings, audit log and research-task repositories (Phase C/D support)."""

from __future__ import annotations

from companion.domain.enums import MemoryKind, MemoryStatus, MemoryVisibility
from companion.memory.models import AccessContext

ADMIN = AccessContext("admin", is_admin=True)


async def test_settings_set_get_and_all(rt):
    assert await rt.repos.settings.get("missing") is None
    assert await rt.repos.settings.get("missing", "fallback") == "fallback"

    await rt.repos.settings.set("web_search.enabled", "false")
    await rt.repos.settings.set("cognitive.provider", "mock")
    assert await rt.repos.settings.get("web_search.enabled") == "false"
    assert await rt.repos.settings.get_bool("web_search.enabled") is False

    await rt.repos.settings.set("web_search.enabled", "true")
    assert await rt.repos.settings.get_bool("web_search.enabled") is True

    all_settings = await rt.repos.settings.all()
    assert all_settings["cognitive.provider"] == "mock"
    assert "web_search.enabled" in all_settings


async def test_memory_service_writes_audit_entries(rt):
    item = await rt.memory_service.create(
        kind=MemoryKind.WORLD_KNOWLEDGE,
        content="audit trail test",
        access=ADMIN,
        visibility=MemoryVisibility.PUBLIC,
    )
    await rt.memory_service.update(
        item.memory_id, access=ADMIN, expected_version=1, content="audit trail edited"
    )
    await rt.memory_service.retract(item.memory_id, access=ADMIN)

    rows = await rt.repos.audit.list(limit=50)
    actions = [r["action"] for r in rows]
    assert "create" in actions
    assert "update" in actions
    assert "retract" in actions
    for r in rows:
        assert r["entity_type"] == "memory"
        assert r["actor"] == "admin"


async def test_person_registry_writes_audit_entries(rt):
    await rt.registry.register(
        "Audited", "friend",
        allow_processing=True, allow_persistence=True, allow_memory_learning=True,
    )
    rows = await rt.repos.audit.list(entity_type="person")
    assert any(r["action"] == "register" and r["detail"].startswith("role=") for r in rows)


async def test_audit_filter_by_entity_and_actor(rt):
    await rt.repos.audit.record("alice", "poke", "memory", "mem_1", "hello")
    await rt.repos.audit.record("bob", "poke", "person", "person_1", "world")

    mem = await rt.repos.audit.list(entity_type="memory")
    assert [r["entity_id"] for r in mem] == ["mem_1"]

    bob = await rt.repos.audit.list(actor="bob")
    assert [r["entity_id"] for r in bob] == ["person_1"]


async def test_research_task_lifecycle(rt):
    task_id = await rt.repos.research.create(
        "why is OTA flaky?", source_kind="curiosity", priority=3
    )
    assert task_id.startswith("research_")

    tasks = await rt.repos.research.list()
    assert any(t["task_id"] == task_id and t["status"] == "PENDING" for t in tasks)

    await rt.repos.research.set_status(task_id, "RUNNING")
    await rt.repos.research.complete(task_id, "because of dual-bank writes")

    task = await rt.repos.research.get(task_id)
    assert task["status"] == "COMPLETED"
    assert task["result"] == "because of dual-bank writes"
    assert task["started_at"] is not None


async def test_research_filter_by_status(rt):
    await rt.repos.research.create("pending question", source_kind="manual")
    done = await rt.repos.research.create("done question", source_kind="manual")
    await rt.repos.research.complete(done, "answer")

    pending = await rt.repos.research.list(status="PENDING")
    completed = await rt.repos.research.list(status="COMPLETED")
    assert all(t["status"] == "PENDING" for t in pending)
    assert all(t["status"] == "COMPLETED" for t in completed)
