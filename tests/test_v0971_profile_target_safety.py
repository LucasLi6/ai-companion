"""V0.9.7.1 Section A: profile target safety.

Every operation may touch only the entries it is allowed to touch. The checks
are repeated inside the final write transaction, so a caller that skips the
engine and hands the service a hand-built ``ValidatedChange(auto_apply=True)``
still cannot cross fields, write identity, or archive a strict field.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
    ValidatedChange,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider


async def _person(rt, name="Targets"):
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


async def _event(rt, person, text, conversation="c1", *, occurred_at=None):
    await rt.conversations.get_or_create(conversation)
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
        occurred_at=occurred_at,
    )
    await rt.repos.utterance.save(event)
    return event


async def _review_add(rt, person, field_key, content, *, evidence=None):
    """Apply one legitimate ADD and return its entry id."""
    event = evidence or await _event(rt, person, content)
    proposal = ProfileChangeProposal(
        operation="ADD",
        field_key=field_key,
        proposed_content=content,
        evidence_event_ids=[event.event_id],
        confidence=0.95,
    )
    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(
            ProfileReviewDecision(
                person_id="ignored",
                base_revision=0,
                decision="PROPOSE_CHANGES",
                reason="test",
                changes=[proposal],
            )
        ),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    applied = await engine.apply_proposals(review)
    assert applied.status in ("APPLIED", "APPLIED_WITH_PENDING")
    assert applied.applied
    return applied.applied[0]["entry_id"]


async def _apply(rt, person, proposal, event):
    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(
            ProfileReviewDecision(
                person_id="ignored",
                base_revision=0,
                decision="PROPOSE_CHANGES",
                reason="test",
                changes=[proposal],
            )
        ),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    return await engine.apply_proposals(review)


async def test_add_may_not_target_an_existing_entry(rt):
    person = await _person(rt)
    interest = await _event(rt, person, "我喜欢机器人。")
    entry_id = await _review_add(rt, person, "interests.topics", "机器人", evidence=interest)

    event = await _event(rt, person, "我还喜欢爵士。")
    applied = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="ADD",
            field_key="interests.topics",
            target_entry_ids=[entry_id],
            proposed_content="爵士乐",
            evidence_event_ids=[event.event_id],
            confidence=0.95,
        ),
        event,
    )
    assert applied.status == "REJECTED"
    assert applied.rejected[0]["reason"] == "add_must_not_target_entries"
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2


async def test_cross_field_update_is_rejected_unless_an_explicit_transition(rt):
    person = await _person(rt)
    project_event = await _event(rt, person, "我在做机器狗。")
    project_id = await _review_add(
        rt, person, "projects.active", "机器狗", evidence=project_event
    )

    event = await _event(rt, person, "我最近很喜欢爵士。")
    applied = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="UPDATE",
            field_key="interests.topics",
            target_entry_ids=[project_id],
            proposed_content="爵士乐",
            evidence_event_ids=[event.event_id],
            confidence=0.95,
        ),
        event,
    )
    assert applied.status == "REJECTED"
    assert applied.rejected[0]["reason"] == "field_transition_not_allowed"
    # The project row is untouched.
    assert (await rt.repos.profile.get_entry(project_id))["field_key"] == (
        "projects.active"
    )
    assert (await rt.repos.profile.get_entry(project_id))["status"] == "ACTIVE"


async def test_allowed_project_transition_moves_the_state(rt):
    person = await _person(rt)
    start = await _event(rt, person, "我在做机器狗。")
    project_id = await _review_add(
        rt, person, "projects.active", "机器狗", evidence=start
    )

    event = await _event(rt, person, "机器狗暂时先放一放。")
    applied = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="UPDATE",
            field_key="projects.paused",
            target_entry_ids=[project_id],
            proposed_content="机器狗",
            evidence_event_ids=[event.event_id],
            confidence=0.95,
        ),
        event,
    )
    assert applied.applied
    old = await rt.repos.profile.get_entry(project_id)
    assert old["status"] == "ARCHIVED"
    assert old["superseded_by"]
    new = await rt.repos.profile.get_entry(old["superseded_by"])
    assert new["field_key"] == "projects.paused"
    assert new["content"] == "机器狗"


async def test_completed_project_may_not_silently_restart(rt):
    person = await _person(rt)
    start = await _event(rt, person, "我在做机器狗。")
    project_id = await _review_add(
        rt, person, "projects.active", "机器狗", evidence=start
    )
    done_event = await _event(rt, person, "机器狗做完了。")
    done = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="UPDATE",
            field_key="projects.completed",
            target_entry_ids=[project_id],
            proposed_content="机器狗",
            evidence_event_ids=[done_event.event_id],
            confidence=0.95,
        ),
        done_event,
    )
    completed_id = done.applied[0]["entry_id"]

    event = await _event(rt, person, "机器狗又有新想法了。")
    applied = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="UPDATE",
            field_key="projects.active",
            target_entry_ids=[completed_id],
            proposed_content="机器狗",
            evidence_event_ids=[event.event_id],
            confidence=0.95,
        ),
        event,
    )
    assert applied.status == "REJECTED"
    assert applied.rejected[0]["reason"] == "field_transition_not_allowed"
    assert (await rt.repos.profile.get_entry(completed_id))["field_key"] == (
        "projects.completed"
    )


async def test_merge_needs_two_distinct_entries_of_the_same_field(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我会 STM32。")
    first_id = await _review_add(rt, person, "knowledge.skills", "STM32", evidence=first)
    second = await _event(rt, person, "我也会嵌入式开发。")
    second_id = await _review_add(
        rt, person, "knowledge.skills", "嵌入式开发", evidence=second
    )

    one = await _event(rt, person, "这两样是一体。")
    applied = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="MERGE",
            field_key="knowledge.skills",
            target_entry_ids=[first_id],
            proposed_content="STM32 嵌入式开发",
            evidence_event_ids=[one.event_id],
            confidence=0.95,
        ),
        one,
    )
    assert applied.rejected[0]["reason"] == "merge_needs_two_targets"

    cross = await _event(rt, person, "我也对嵌入式感兴趣。")
    skill_entry = await rt.repos.profile.get_entry(second_id)
    assert skill_entry is not None
    interest_id = await _review_add(
        rt,
        person,
        "interests.topics",
        "嵌入式",
        evidence=cross,
    )
    merge_event = await _event(rt, person, "合并这两条。")
    mixed = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="MERGE",
            field_key="knowledge.skills",
            target_entry_ids=[first_id, interest_id],
            proposed_content="STM32 嵌入式开发",
            evidence_event_ids=[merge_event.event_id],
            confidence=0.95,
        ),
        merge_event,
    )
    assert mixed.rejected[0]["reason"] == "merge_targets_differ_in_field"
    assert (await rt.repos.profile.get_entry(first_id))["status"] == "ACTIVE"


async def test_refine_takes_exactly_one_target(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我会 STM32。")
    first_id = await _review_add(rt, person, "knowledge.skills", "STM32", evidence=first)
    second = await _event(rt, person, "我也会嵌入式开发。")
    second_id = await _review_add(
        rt, person, "knowledge.skills", "嵌入式开发", evidence=second
    )

    event = await _event(rt, person, "精确一点。")
    applied = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="REFINE",
            field_key="knowledge.skills",
            target_entry_ids=[first_id, second_id],
            proposed_content="STM32 嵌入式开发",
            evidence_event_ids=[event.event_id],
            confidence=0.95,
        ),
        event,
    )
    assert applied.rejected[0]["reason"] == "refine_needs_one_target"


async def test_archive_cannot_touch_a_strict_field(rt):
    person = await _person(rt)
    # Identity rows are only ever created by the trusted confirmation path.
    entry_id = await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="identity.name",
        content="阿明",
        status="ACTIVE",
        source="USER_CORRECTION",
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )
    event = await _event(rt, person, "不要再记我的名字。")
    applied = await _apply(
        rt,
        person,
        ProfileChangeProposal(
            operation="ARCHIVE",
            field_key="identity.name",
            target_entry_ids=[entry_id],
            proposed_content=None,
            evidence_event_ids=[event.event_id],
            confidence=0.95,
        ),
        event,
    )
    assert applied.rejected[0]["reason"] == "strict_field_cannot_be_archived"
    assert (await rt.repos.profile.get_entry(entry_id))["status"] == "ACTIVE"


async def test_service_final_commit_revalidates_even_without_engine_checks(
    rt, monkeypatch
):
    """Bypassing the engine and the precheck cannot bypass the write boundary."""
    person = await _person(rt)
    project_event = await _event(rt, person, "我在做机器狗。")
    project_id = await _review_add(
        rt, person, "projects.active", "机器狗", evidence=project_event
    )
    event = await _event(rt, person, "我最近喜欢爵士。")
    revision_before = (await rt.profile_service.get(person.person_id)).metadata.revision

    async def passthrough(person_id, changes):
        return list(changes), []

    monkeypatch.setattr(
        rt.profile_service, "_precheck_changes", passthrough, raising=False
    )
    decision = SimpleNamespace(
        status="PROPOSE_CHANGES",
        base_revision=revision_before,
        reason="hand built",
        changes=[
            ValidatedChange(
                proposal=ProfileChangeProposal(
                    operation="UPDATE",
                    field_key="interests.topics",
                    target_entry_ids=[project_id],
                    proposed_content="爵士乐",
                    evidence_event_ids=[event.event_id],
                    confidence=1.0,
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="field_transition_not_allowed"):
        await rt.profile_service.apply_refinement(
            person.person_id, decision, source_event_id=event.event_id
        )

    # The whole transaction rolled back: no revision bump, no mutation.
    assert (
        await rt.profile_service.get(person.person_id)
    ).metadata.revision == revision_before
    entry = await rt.repos.profile.get_entry(project_id)
    assert entry["field_key"] == "projects.active"
    assert entry["content"] == "机器狗"
    assert entry["status"] == "ACTIVE"
