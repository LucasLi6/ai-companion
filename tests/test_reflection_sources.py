"""Phase 5: reflection source tracking.

``propose(source_ids=...)`` must link real sources via ``derived_from``,
reject fabricated or inaccessible ones, never auto-classify a proposal as a
PERSONAL_FACT, and invalidate derived reflections when a source is deleted or
retracted.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import MemoryKind, MemoryStatus, MemoryVisibility
from companion.idle import IDLE_AGENT, build_idle_scheduler
from companion.memory.models import (
    AccessContext,
    MemoryNotFoundError,
    MemoryPermissionDeniedError,
)

ADMIN = AccessContext("admin", is_admin=True)


async def _source(rt, visibility=MemoryVisibility.PUBLIC, subjects=None, content="源记忆"):
    item = await rt.memory_service.create(
        kind="PROJECT_STATE",
        content=content,
        access=ADMIN,
        visibility=visibility,
        subjects=subjects or [],
    )
    return item.memory_id


def _derived_links(view, source_id):
    return [
        l
        for l in view["links"]
        if l["link_type"] == "derived_from" and l["linked_memory_id"] == source_id
    ]


async def test_propose_links_derived_sources(rt):
    src = await _source(rt)
    proposal = await rt.memory_service.propose(
        access=IDLE_AGENT, content="反思", source_ids=[src], source_access=ADMIN
    )
    view = await rt.memory_service.get_view(proposal.memory_id, ADMIN)
    assert _derived_links(view, src), "derived_from link missing"


async def test_propose_rejects_fabricated_source(rt):
    with pytest.raises(MemoryNotFoundError):
        await rt.memory_service.propose(
            access=IDLE_AGENT, content="反思", source_ids=["mem_不存在"], source_access=ADMIN
        )


async def test_propose_rejects_inaccessible_source(rt):
    person = await rt.registry.register("X", "friend", allow_memory_learning=True)
    src = await _source(
        rt, MemoryVisibility.PRIVATE, subjects=[("PERSON", person.person_id)]
    )
    other = AccessContext("other", is_admin=False)
    with pytest.raises(MemoryPermissionDeniedError):
        await rt.memory_service.propose(
            access=other, content="反思", source_ids=[src]
        )


async def test_propose_never_auto_personalfact(rt):
    item = await rt.memory_service.propose(
        access=IDLE_AGENT, content="关于某人的事实", kind="PERSONAL_FACT"
    )
    assert item.kind == MemoryKind.REFLECTION


async def test_reflection_engine_links_its_sources(rt):
    src = await _source(rt)
    scheduler = build_idle_scheduler(rt)
    results = await scheduler.reflection.run()
    assert len(results) == 1
    view = await rt.memory_service.get_view(results[0].memory_id, ADMIN)
    assert _derived_links(view, src), "engine did not record derived_from"


async def test_source_delete_invalidates_derived(rt):
    src = await _source(rt)
    proposal = await rt.memory_service.propose(
        access=IDLE_AGENT, content="反思", source_ids=[src], source_access=ADMIN
    )
    await rt.memory_service.delete(src, access=ADMIN)
    derived = await rt.memory_service.get_item(proposal.memory_id, ADMIN)
    assert derived.status == MemoryStatus.RETRACTED


async def test_source_retract_invalidates_derived(rt):
    src = await _source(rt)
    proposal = await rt.memory_service.propose(
        access=IDLE_AGENT, content="反思", source_ids=[src], source_access=ADMIN
    )
    await rt.memory_service.retract(src, access=ADMIN)
    derived = await rt.memory_service.get_item(proposal.memory_id, ADMIN)
    assert derived.status == MemoryStatus.RETRACTED
