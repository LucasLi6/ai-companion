"""V0.9.7: PROFILE_CORE / PROFILE_RELEVANT compiling and person isolation."""

from __future__ import annotations

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_compiler import (
    MAX_BLOCK_CHARS,
    MAX_CORE_CHARS,
    compile_profile_context,
    compile_profile_core,
    compile_profile_relevant,
)
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider
from companion.routing.context import RoutingContextBuilder


async def _person(rt, name, *, external_ai=True):
    person = await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
    )
    await rt.profile_service.ensure_profile(person.person_id, actor="test")
    return person


async def _entry(rt, person, field_key, content, status="ACTIVE"):
    return await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key=field_key,
        content=content,
        status=status,
        source="DIRECT_SELF_REPORT",
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )


def _builder(rt):
    return RoutingContextBuilder(
        repos=rt.repos,
        registry=rt.registry,
        profile_service=rt.profile_service,
    )


async def test_core_uses_confirmed_identity_not_the_registration_label(rt):
    person = await _person(rt, "RegistrationLabel")
    profile = await rt.profile_service.get(person.person_id)
    block = compile_profile_core(profile)
    assert "【当前交流对象】" in block
    assert f"person_id={person.person_id}" in block
    assert "relationship=stranger" in block
    assert "RegistrationLabel" not in block
    assert "name=" not in block

    await _entry(rt, person, "identity.name", "小明")
    await _entry(rt, person, "identity.preferred_name", "阿明")
    await rt.profile_service.set_field(
        person.person_id, "communication.formality", "casual"
    )
    profile = await rt.profile_service.get(person.person_id)
    block = compile_profile_core(profile)
    assert "name=小明" in block
    assert "preferred_name=阿明" in block
    assert "formality=casual" in block
    assert "RegistrationLabel" not in block


async def test_relevant_block_follows_the_question(rt):
    person = await _person(rt, "Relevant")
    await _entry(rt, person, "interests.topics", "机器人")
    await _entry(rt, person, "interests.topics", "爵士乐")
    await _entry(rt, person, "projects.active", "Ava")
    profile = await rt.profile_service.get(person.person_id)

    jazz = compile_profile_relevant(profile, query="最近爵士乐听得怎么样？")
    assert "爵士乐" in jazz
    assert "机器人" not in jazz
    assert "skills=" not in jazz

    robot = compile_profile_relevant(profile, query="机器人进展如何？")
    assert "机器人" in robot
    assert "爵士乐" not in robot


async def test_item_budget_never_slices_a_fact_in_half(rt):
    person = await _person(rt, "Budget")
    for index in range(6):
        await _entry(
            rt, person, "interests.topics", f"兴趣条目{index}" + "细节" * 30
        )
    profile = await rt.profile_service.get(person.person_id)
    block = compile_profile_relevant(profile, max_chars=90)
    assert block is not None
    assert len(block) <= 90
    # A shortened block ends at an explicit ellipsis; otherwise items are whole.
    for item in block.split("=", 1)[-1].split(", "):
        assert item.endswith("…") or len(item) <= 40

    combined = compile_profile_context(profile, max_chars=MAX_BLOCK_CHARS)
    assert len(combined) <= MAX_BLOCK_CHARS
    core = compile_profile_core(profile)
    assert len(core) <= MAX_CORE_CHARS


async def test_each_person_sees_only_their_own_understanding(rt):
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")
    await _entry(rt, alice, "projects.active", "A 的机器人项目")
    await _entry(rt, bob, "projects.active", "B 的秘密项目")
    builder = _builder(rt)

    alice_block, alice_revision = await builder._profile_context(alice.person_id)
    bob_block, bob_revision = await builder._profile_context(bob.person_id)
    assert "A 的机器人项目" in alice_block
    assert "B 的秘密项目" not in alice_block
    assert "B 的秘密项目" in bob_block
    assert "A 的机器人项目" not in bob_block
    assert alice_revision == bob_revision == 1

    outsider = await _person(rt, "Outsider", external_ai=False)
    await _entry(rt, outsider, "projects.active", "不应上传")
    assert await builder._profile_context(outsider.person_id) == (None, None)


async def test_routing_context_carries_one_revision_and_no_other_person(rt):
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")
    await _entry(rt, alice, "projects.active", "A 的机器人项目")
    await _entry(rt, bob, "projects.active", "B 的秘密项目")
    conversation = await rt.conversations.get_or_create("ctx")
    event = make_utterance_event(
        conversation_id="ctx",
        text="机器人做得怎么样了？",
        addressee_hint="agent",
        speaker_person_id=alice.person_id,
        source=EventSource.TEST,
    )
    builder = _builder(rt)
    routing = await builder.build(event, conversation)
    assert routing.current_speaker_id == alice.person_id
    assert routing.compact_person_profile_revision == 1
    assert "A 的机器人项目" in routing.compact_person_profile
    assert "B 的秘密项目" not in routing.compact_person_profile
    assert "B 的秘密项目" not in str(routing.as_prompt_dict())


async def test_next_turn_reads_the_committed_revision(rt):
    person = await _person(rt, "Next")
    await rt.conversations.get_or_create("n1")
    event = make_utterance_event(
        conversation_id="n1",
        text="我在做机器人。",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)
    builder = _builder(rt)
    before, before_revision = await builder._profile_context(person.person_id)
    assert before_revision == 1
    assert "机器人" not in (before or "")

    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(
            [
                ProfileReviewDecision(
                    person_id=person.person_id,
                    base_revision=1,
                    decision="PROPOSE_CHANGES",
                    changes=[
                        ProfileChangeProposal(
                            operation="ADD",
                            field_key="projects.active",
                            proposed_content="机器人",
                            evidence_event_ids=[event.event_id],
                            confidence=0.95,
                        )
                    ],
                )
            ]
        ),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    await engine.apply_proposals(await engine.review(person.person_id, event.event_id))

    after, after_revision = await builder._profile_context(person.person_id)
    assert after_revision == 2
    assert "机器人" in after
