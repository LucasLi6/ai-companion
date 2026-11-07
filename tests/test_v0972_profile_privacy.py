"""V0.9.7.2 Section B: multi-source privacy and retraction rules.

One fact may be supported by several memories. Retracting one of them must
keep what the survivors independently prove, suspend merged text that can no
longer be fully proven, and never let an explicit privacy delete leave a
source-only fragment behind — for this person or any other.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.memory.models import AccessContext, MemoryVisibility
from companion.people.profile_compiler import compile_profile_context
from companion.routing.models import DataControlRequest


async def _person(rt, name="Privacy"):
    person = await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.profile_service.ensure_profile(person.person_id, actor="test")
    return person


async def _event(rt, person, text, conversation="c1"):
    await rt.conversations.get_or_create(conversation)
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)
    return event


async def _memory(rt, person, content):
    item = await rt.memory_service.create(
        kind="PROJECT_STATE",
        content=content,
        access=AccessContext("admin", is_admin=True),
        visibility=MemoryVisibility.PRIVATE,
        subjects=[("PERSON", person.person_id)],
    )
    return item.memory_id


async def _entry(rt, person, *, content, memories=(), merged_from=(), status="ACTIVE"):
    return await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="interests.topics",
        content=content,
        status=status,
        source="INTERACTION_INFERENCE",
        confidence=0.9,
        evidence_event_ids=[],
        evidence_memory_ids=list(memories),
        merged_from=list(merged_from),
    )


async def _delete_memory(rt, person, memory_id, event):
    pending = await rt.data_control.resolve_request(
        DataControlRequest(
            operation="DELETE",
            target_type="MEMORY",
            target_id=memory_id,
            scope="SELF",
        ),
        event,
    )
    return await rt.data_control.execute_confirmed(pending.operation_id)


async def test_retracting_one_of_two_memories_keeps_independent_support(rt):
    person = await _person(rt)
    m1 = await _memory(rt, person, "摄影记忆一")
    m2 = await _memory(rt, person, "摄影记忆二")
    both = await _entry(rt, person, content="摄影", memories=[m1, m2])
    only_m1 = await _entry(rt, person, content="钓鱼", memories=[m1])
    only_m2 = await _entry(rt, person, content="露营", memories=[m2])
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision

    result = await rt.profile_service.invalidate_profile_sources(
        person.person_id, memory_ids=(m1,), deletion_mode="RETRACTION"
    )
    assert result.revision == revision + 1  # one commit, one revision bump

    kept = await rt.repos.profile.get_entry(both)
    assert kept["status"] == "ACTIVE"
    assert kept["content"] == "摄影"
    assert kept["evidence_memory_ids"] == [m2]

    assert (await rt.repos.profile.get_entry(only_m1))["status"] == "ARCHIVED"
    assert (await rt.repos.profile.get_entry(only_m2))["status"] == "ACTIVE"

    effective = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE", "TENTATIVE")
    )
    assert {item["content"] for item in effective} == {"摄影", "露营"}


async def test_a_merged_text_is_suspended_when_one_source_is_retracted(rt):
    person = await _person(rt)
    m1 = await _memory(rt, person, "STM32 记忆")
    m2 = await _memory(rt, person, "嵌入式记忆")
    merged = await _entry(
        rt,
        person,
        content="STM32 嵌入式开发",
        memories=[m1, m2],
        merged_from=["pfe_a", "pfe_b"],
    )

    await rt.profile_service.invalidate_profile_sources(
        person.person_id, memory_ids=(m1,), deletion_mode="RETRACTION"
    )

    entry = await rt.repos.profile.get_entry(merged)
    # The survivor cannot be shown to support every specific in the merged
    # text, so it stops being effective until someone reviews it.
    assert entry["status"] == "PENDING_REVIEW"
    assert entry["evidence_memory_ids"] == [m2]
    assert entry["content"] == "STM32 嵌入式开发"


async def test_privacy_delete_removes_everything_a_memory_alone_supports(rt):
    person = await _person(rt)
    event = await _event(rt, person, "随便聊聊。")
    m1 = await _memory(rt, person, "代号蓝鲸")
    m2 = await _memory(rt, person, "喜欢露营")
    secret = await _entry(rt, person, content="蓝鲸项目", memories=[m1])
    shared = await _entry(rt, person, content="蓝鲸与露营", memories=[m1, m2])
    unrelated = await _entry(rt, person, content="露营", memories=[m2])
    result = await _delete_memory(rt, person, m1, event)
    assert result.status == "COMPLETED"
    assert await rt.memory_service.repo.get(m1) is None

    # Anything the deleted memory helped support is gone, not archived.
    assert await rt.repos.profile.get_entry(secret) is None
    assert await rt.repos.profile.get_entry(shared) is None
    # Content supported only by the surviving memory is untouched.
    kept = await rt.repos.profile.get_entry(unrelated)
    assert kept["status"] == "ACTIVE"
    assert kept["content"] == "露营"
    # V0.9.9.6: no candidate queue exists, so nothing can survive the delete.
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []

    profile = await rt.profile_service.get(person.person_id)
    assert "蓝鲸" not in profile.compact_summary
    assert profile.compact_summary == compile_profile_context(profile)


async def test_deleting_one_persons_source_never_touches_another_person(rt):
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")
    event = await _event(rt, alice, "随便聊聊。")
    alice_memory = await _memory(rt, alice, "Alice 的秘密")
    bob_memory = await _memory(rt, bob, "Bob 的露营")
    alice_entry = await _entry(rt, alice, content="蓝鲸", memories=[alice_memory])
    bob_entry = await _entry(rt, bob, content="露营", memories=[bob_memory])
    bob_revision = (await rt.profile_service.get(bob.person_id)).metadata.revision

    result = await _delete_memory(rt, alice, alice_memory, event)
    assert result.status == "COMPLETED"

    assert await rt.repos.profile.get_entry(alice_entry) is None
    kept = await rt.repos.profile.get_entry(bob_entry)
    assert kept["status"] == "ACTIVE"
    assert kept["content"] == "露营"
    bob_profile = await rt.profile_service.get(bob.person_id)
    assert bob_profile.metadata.revision == bob_revision
    assert "蓝鲸" not in bob_profile.compact_summary
    assert [
        item.content
        for items in bob_profile.entries.values()
        for item in items
        if item.is_effective()
    ] == ["露营"]


async def test_a_failed_privacy_delete_rolls_back_the_whole_source_graph(rt, monkeypatch):
    person = await _person(rt)
    event = await _event(rt, person, "随便聊聊。")
    m1 = await _memory(rt, person, "代号蓝鲸")
    entry = await _entry(rt, person, content="蓝鲸项目", memories=[m1])
    revision = (await rt.profile_service.get(person.person_id)).metadata.revision

    async def explode(**kwargs):
        raise RuntimeError("simulated failure after profile invalidation")

    # Fail at the final revision write, after entries and the summary have
    # already been touched inside the same profile transaction.
    monkeypatch.setattr(rt.repos.profile, "insert_revision", explode)

    result = await _delete_memory(rt, person, m1, event)
    assert result.status == "FAILED"

    # Memory, entry, summary and revision are all exactly as before.
    assert await rt.memory_service.repo.get(m1) is not None
    kept = await rt.repos.profile.get_entry(entry)
    assert kept["status"] == "ACTIVE"
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == revision
