"""Phase 3: idle-agent privacy — local read vs external AI upload.

The idle engines must not send every PRIVATE memory to a (mock or future real)
provider. Only PUBLIC memories and PRIVATE/SHARED memories whose every PERSON
subject holds the independent ``allow_external_ai`` grant may leave the store.
``allow_memory_learning`` alone is never sufficient, and a revoked person's
memories are re-checked at read time.
"""

from __future__ import annotations

import json

from companion.domain.enums import MemoryVisibility, RegistrationStatus
from companion.idle import build_idle_scheduler
from companion.idle.context import IdleMemoryContextBuilder
from companion.memory.models import AccessContext

ADMIN = AccessContext("admin", is_admin=True)


async def _memory(rt, visibility, subjects, content="私密内容"):
    item = await rt.memory_service.create(
        kind="EPISODE",
        content=content,
        access=ADMIN,
        visibility=visibility,
        subjects=subjects,
    )
    return item.memory_id


async def _three_people(rt):
    a = await rt.registry.register(
        "A", "friend", allow_memory_learning=True, allow_external_ai=False
    )
    b = await rt.registry.register(
        "B", "friend", allow_memory_learning=True, allow_external_ai=True
    )
    c = await rt.registry.register(
        "C", "friend", allow_memory_learning=True, allow_external_ai=True
    )
    # C is later revoked: their private context must stop being exportable even
    # though their stored consent flag is still True.
    await rt.registry.set_status(c.person_id, RegistrationStatus.REVOKED)
    return a, b, c


async def test_external_context_privacy_matrix(rt):
    a, b, c = await _three_people(rt)

    public_id = await _memory(rt, MemoryVisibility.PUBLIC, [], "公开")
    a_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", a.person_id)])
    b_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", b.person_id)])
    c_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", c.person_id)])
    ab_shared = await _memory(
        rt, MemoryVisibility.SHARED, [("PERSON", a.person_id), ("PERSON", b.person_id)]
    )
    no_person = await _memory(rt, MemoryVisibility.PRIVATE, [("PROJECT", "p1")])

    builder = IdleMemoryContextBuilder(rt.memory_service, rt.registry, audit=rt.repos.audit)
    released = await builder.external_context(limit=50)
    released_ids = {m["memory_id"] for m in released}

    assert public_id in released_ids
    assert b_private in released_ids
    # Local-only learning is not upload permission.
    assert a_private not in released_ids
    # A revoked person is re-checked and excluded.
    assert c_private not in released_ids
    # Multi-subject: all subjects must authorise.
    assert ab_shared not in released_ids
    # Non-person-scoped private content is never auto-exported.
    assert no_person not in released_ids

    # Audit records exactly which ids were released and why.
    rows = await rt.repos.audit.list(actor="idle", limit=10)
    release_audits = [r for r in rows if r["action"] == "idle_external_context"]
    assert release_audits, "no idle external-context audit record written"
    reasons = json.loads(release_audits[0]["detail"])
    assert reasons[public_id] == "public"
    assert reasons[b_private] == "external_ai_consent"
    assert a_private not in reasons


class RecordingProvider:
    name = "recording-deepseek"

    def __init__(self):
        self.reflect_inputs: list[list[str]] = []
        self.question_inputs: list[list[str]] = []

    async def reflect(self, memories):
        self.reflect_inputs.append([m["memory_id"] for m in memories])
        if not memories:
            return []
        return [{"content": "[mock] reflection", "kind": "REFLECTION", "conflicts_with": None}]

    async def ask_questions(self, memories):
        self.question_inputs.append([m["memory_id"] for m in memories])
        return []


async def test_reflection_engine_sends_only_authorized_context(rt):
    a, b, c = await _three_people(rt)
    public_id = await _memory(rt, MemoryVisibility.PUBLIC, [], "公开")
    b_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", b.person_id)])
    a_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", a.person_id)])
    c_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", c.person_id)])

    provider = RecordingProvider()
    scheduler = build_idle_scheduler(rt, deepseek=provider)
    await scheduler.reflection.run()

    sent = set(provider.reflect_inputs[0])
    assert public_id in sent
    assert b_private in sent
    assert a_private not in sent
    assert c_private not in sent


async def test_curiosity_engine_sends_only_authorized_context(rt):
    a, b, c = await _three_people(rt)
    public_id = await _memory(rt, MemoryVisibility.PUBLIC, [], "公开")
    b_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", b.person_id)])
    a_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", a.person_id)])
    c_private = await _memory(rt, MemoryVisibility.PRIVATE, [("PERSON", c.person_id)])

    provider = RecordingProvider()
    scheduler = build_idle_scheduler(rt, deepseek=provider)
    await scheduler.curiosity.run()

    sent = set(provider.question_inputs[0])
    assert public_id in sent
    assert b_private in sent
    assert a_private not in sent
    assert c_private not in sent
