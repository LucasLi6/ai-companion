"""V0.9.9.3: profile auto-fill repair, eager field filling and dedupe.

Offline only: every model call is a fake/scripted provider and every database
is a temporary SQLite file.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource, JobType, RegistrationStatus
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.jobs.profile_review import (
    LEGACY_SKIPPED_REASON,
    ProfileReviewScheduler,
    build_profile_review_handler,
    profile_review_job_key,
    repair_skipped_reviews,
)
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileRefinementError,
    ProfileReviewContextSource,
    ProfileReviewDecision,
    entries_equivalent,
)
from companion.people.profile_service import MAX_REFINEMENT_CHANGES
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider

LOG = logging.getLogger("test.v0993")


# -- helpers ---------------------------------------------------------------


async def _person(rt, name="Auto", *, ensure=True, **consent):
    values = dict(
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    values.update(consent)
    person = await rt.registry.register(name, "owner", **values)
    if ensure:
        await rt.profile_service.ensure_profile(person.person_id, actor="test")
    else:
        # Registration normally ensures the row; simulate an older account.
        await rt.profile_service.purge(person.person_id, reason="test_missing_row")
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


def _source(rt):
    return ProfileReviewContextSource(
        rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
    )


def _engine(rt, provider):
    return ProfileRefinementEngine(
        rt.profile_service, rt.repos.profile, provider, _source(rt)
    )


def _scheduler(rt, delay_seconds=0.0):
    return ProfileReviewScheduler(
        rt.repos.profile,
        rt.repos.job,
        persons=rt.repos.person,
        delay_seconds=delay_seconds,
    )


def _add(field_key, content, event_id, *, quote=None, confidence=0.9, confirm=False):
    return ProfileChangeProposal(
        operation="ADD",
        field_key=field_key,
        proposed_content=content,
        evidence_event_ids=[event_id],
        evidence_quotes=[quote] if quote else [],
        confidence=confidence,
        requires_confirmation=confirm,
    )


def _decision(*changes, reason="test"):
    return ProfileReviewDecision(
        person_id="placeholder",
        base_revision=0,
        decision="PROPOSE_CHANGES" if changes else "NO_CHANGE",
        reason=reason,
        changes=list(changes),
    )


async def _review(rt, provider, person_id, event_id):
    return await _engine(rt, provider).review(person_id, event_id)


async def _active_entries(rt, person_id):
    return await rt.repos.profile.list_entries(
        person_id, statuses=("ACTIVE", "TENTATIVE")
    )


def _receipts_skipped(receipts):
    return [item for item in receipts if item["status"] == "SKIPPED"]


# -- A. Profile initialisation --------------------------------------------


async def test_t01_missing_profile_row_is_created_for_authorized_person(rt, caplog):
    person = await _person(rt, ensure=False)
    event = await _event(rt, person, "你好。")
    provider = ScriptedProfileRefinementProvider([_decision()])
    caplog.set_level(logging.INFO, logger="companion.profile_refinement")

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.status == "NO_CHANGE"
    snapshot = await rt.repos.profile.get(person.person_id)
    assert snapshot is not None
    assert snapshot["revision"] == 1
    assert "profile_review_profile_repaired" in caplog.text
    assert provider.payloads


async def test_t02_missing_external_ai_consent_returns_unauthorized(rt):
    person = await _person(rt, ensure=False, allow_external_ai=False)
    event = await _event(rt, person, "我在做机器人。")

    result = await _review(
        rt, ScriptedProfileRefinementProvider([_decision()]),
        person.person_id, event.event_id,
    )

    assert result.status == "UNAUTHORIZED"
    assert result.reason == "missing_consent"
    assert await rt.repos.profile.get(person.person_id) is None


async def test_t03_missing_persistence_consent_never_creates_profile(rt):
    person = await _person(rt, ensure=False, allow_persistence=False)
    event = await _event(rt, person, "我在做机器人。")

    result = await _review(
        rt, ScriptedProfileRefinementProvider([_decision()]),
        person.person_id, event.event_id,
    )

    assert result.status == "UNAUTHORIZED"
    assert await rt.repos.profile.get(person.person_id) is None


async def test_t04_missing_processing_consent_returns_unauthorized(rt):
    person = await _person(rt, ensure=False, allow_processing=False)
    event = await _event(rt, person, "我在做机器人。")

    result = await _review(
        rt, ScriptedProfileRefinementProvider([_decision()]),
        person.person_id, event.event_id,
    )

    assert result.status == "UNAUTHORIZED"
    assert await rt.repos.profile.get(person.person_id) is None


async def test_t05_revoked_person_is_never_recreated(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器人。")
    await rt.registry.set_status(person.person_id, RegistrationStatus.REVOKED)

    result = await _review(
        rt, ScriptedProfileRefinementProvider([_decision()]),
        person.person_id, event.event_id,
    )

    assert result.status == "UNAUTHORIZED"
    assert await rt.repos.profile.get(person.person_id) is None


async def test_t06_repair_is_idempotent_across_reviews(rt):
    person = await _person(rt, ensure=False)
    first = await _event(rt, person, "你好。")
    second = await _event(rt, person, "还在吗。", "c2")
    provider = ScriptedProfileRefinementProvider([_decision(), _decision()])

    await _review(rt, provider, person.person_id, first.event_id)
    await _review(rt, provider, person.person_id, second.event_id)

    row = await rt.repos.profile.get(person.person_id)
    assert row is not None
    assert (await rt.repos.profile.list_revisions(person.person_id)) == []


async def test_t07_unreadable_after_repair_raises_explicitly():
    class BrokenProfiles:
        async def can_review(self, person_id):
            return True

        async def get_full_snapshot(self, person_id):
            return None

        async def ensure_profile(self, person_id, *, actor="system"):
            return False

    class NeverContext:
        async def load(self, **kwargs):
            raise AssertionError("must not reach the model")

    engine = ProfileRefinementEngine(
        BrokenProfiles(),
        None,
        ScriptedProfileRefinementProvider(),
        NeverContext(),
    )
    with pytest.raises(ProfileRefinementError) as excinfo:
        await engine.review("person_x", "event_x")
    assert "authorized_profile_initialization_failed" in str(excinfo.value)


async def test_t08_existing_empty_profile_is_usable(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("interests.likes", "喜欢爵士乐", event.event_id,
                        quote="我喜欢爵士乐"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.status == "PROPOSED"
    assert result.base_revision == 1
    assert len(result.changes) == 1


# -- B. Automatic write without a confidence gate -------------------------


ACCEPTANCE_TEXT = (
    "我最近在学葡萄牙语，也在做一个机器人项目。"
    "我喜欢爵士乐，不太喜欢太正式的回答。"
)


def _acceptance_decision(event_id):
    return _decision(
        _add("knowledge.skills", "正在学习葡萄牙语", event_id,
             quote="我最近在学葡萄牙语", confidence=0.2),
        _add("projects.active", "正在做一个机器人项目", event_id,
             quote="也在做一个机器人项目", confidence=0.2),
        _add("interests.likes", "喜欢爵士乐", event_id,
             quote="我喜欢爵士乐", confidence=0.2),
        _add("communication.preferred_tone", "偏好不太正式的回答", event_id,
             quote="不太喜欢太正式的回答", confidence=0.2),
    )


async def test_t09_four_fields_auto_apply_without_confidence_gate(rt):
    person = await _person(rt)
    event = await _event(rt, person, ACCEPTANCE_TEXT)
    provider = ScriptedProfileRefinementProvider([_acceptance_decision(event.event_id)])

    result = await _review(rt, provider, person.person_id, event.event_id)
    assert result.status == "PROPOSED"
    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.status == "APPLIED"
    entries = await _active_entries(rt, person.person_id)
    assert {(item["field_key"], item["content"]) for item in entries} == {
        ("knowledge.skills", "正在学习葡萄牙语"),
        ("projects.active", "正在做一个机器人项目"),
        ("interests.likes", "喜欢爵士乐"),
        ("communication.preferred_tone", "偏好不太正式的回答"),
    }
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2
    assert await rt.repos.profile.list_candidates(person.person_id, status=None) == []
    evidence = await rt.repos.profile.list_evidence(person.person_id)
    assert len(evidence) == 4
    assert {item["source_event_id"] for item in evidence} == {event.event_id}


async def test_t10_very_low_confidence_is_metadata_not_a_block(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做一个机械臂。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("projects.active", "正在做一个机械臂", event.event_id,
                        quote="我在做一个机械臂。", confidence=0.05))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)
    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.applied[0]["status"] == "APPLIED"
    assert applied.pending == []


async def test_t11_proposed_content_is_written_verbatim(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我最近在学葡萄牙语。")
    content = "正在学习葡萄牙语（入门阶段）"
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("knowledge.skills", content, event.event_id,
                        quote="我最近在学葡萄牙语。"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)
    await _engine(rt, provider).apply_proposals(result)

    entries = await _active_entries(rt, person.person_id)
    assert [item["content"] for item in entries] == [content]


async def test_t12_current_project_and_shared_context_are_recorded(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我们在上周末一起调试了机器狗。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _add("projects.active", "正在调试机器狗项目", event.event_id,
                 quote="调试了机器狗"),
            _add("interaction.shared_context", "上周末一起调试机器狗", event.event_id,
                 quote="上周末一起调试了机器狗"),
        )]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)
    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.status == "APPLIED"
    fields = {item["field_key"] for item in await _active_entries(rt, person.person_id)}
    assert fields == {"projects.active", "interaction.shared_context"}


async def test_t13_trait_with_quote_auto_applies(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我做事比较谨慎，喜欢先想清楚再动手。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("personality.traits", "做事谨慎，先想清楚再动手", event.event_id,
                        quote="我做事比较谨慎"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)
    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.status == "APPLIED"
    assert [item["content"] for item in await _active_entries(rt, person.person_id)] == [
        "做事谨慎，先想清楚再动手"
    ]


async def test_t14_inferred_trait_without_quote_stays_pending(rt):
    person = await _person(rt)
    event = await _event(rt, person, "今天一下子就把问题解决了。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("personality.traits", "行动果断", event.event_id,
                        confidence=0.95))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.changes[0].auto_apply is False
    assert result.changes[0].pending_reason == "trait_inference_requires_review"
    applied = await _engine(rt, provider).apply_proposals(result)
    assert applied.applied == []
    assert await _active_entries(rt, person.person_id) == []


async def test_t14b_inferred_habit_without_quote_stays_pending(rt):
    person = await _person(rt)
    event = await _event(rt, person, "今天又熬夜了。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("habits.stable_habits", "经常熬夜", event.event_id,
                        confidence=0.9))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.changes[0].pending_reason == "trait_inference_requires_review"


async def test_t15_sensitive_content_stays_pending(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我最近在吃药。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("background.stable_facts", "最近在用药", event.event_id,
                        quote="我最近在吃药。"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.changes[0].pending_reason == "sensitive_field_requires_review"


async def test_t16_hypothetical_stays_pending(rt):
    person = await _person(rt)
    event = await _event(rt, person, "如果我以后去日本，可能会学日语。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("knowledge.skills", "以后可能学日语", event.event_id,
                        quote="如果我以后去日本"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.changes[0].pending_reason == "uncertain_evidence_requires_review"


async def test_t17_strict_identity_stays_pending(rt):
    person = await _person(rt)
    event = await _event(rt, person, "叫我小王就行。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("identity.preferred_name", "小王", event.event_id,
                        quote="叫我小王"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.changes[0].pending_reason == "confirmed_identity_requires_confirmation"


async def test_t18_no_evidence_stays_pending(rt):
    person = await _person(rt)
    event = await _event(rt, person, "今天天气不错。")
    proposal = ProfileChangeProposal(
        operation="ADD", field_key="interests.likes",
        proposed_content="喜欢晴天", evidence_event_ids=[], evidence_quotes=[],
        confidence=0.99,
    )
    provider = ScriptedProfileRefinementProvider([_decision(proposal)])

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.changes[0].pending_reason == "no_evidence"


async def test_t19_quote_not_in_text_is_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("interests.likes", "喜欢古典乐", event.event_id,
                        quote="我喜欢古典乐"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.status == "REJECTED"
    assert result.rejected[0]["reason"] == "evidence_quote_not_found"


async def test_t20_other_persons_event_is_rejected(rt):
    owner = await _person(rt, "Owner")
    other = await _person(rt, "Other")
    event = await _event(rt, other, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("interests.likes", "喜欢爵士乐", event.event_id,
                        quote="我喜欢爵士乐"))]
    )

    result = await _review(rt, provider, owner.person_id, event.event_id)

    assert result.status == "UNAUTHORIZED"
    assert result.reason == "source_event_not_owned_by_person"


async def test_t21_model_requested_confirmation_stays_pending(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我可能有点完美主义。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("personality.traits", "完美主义", event.event_id,
                        quote="完美主义", confirm=True))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.changes[0].pending_reason == "model_requested_confirmation"


async def test_t22_change_limit_is_ten(rt):
    assert MAX_REFINEMENT_CHANGES == 10
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢很多事。")
    changes = [
        _add("interests.likes", f"兴趣{i}", event.event_id, quote="我喜欢很多事")
        for i in range(12)
    ]
    provider = ScriptedProfileRefinementProvider([_decision(*changes)])

    result = await _review(rt, provider, person.person_id, event.event_id)

    accepted = [item for item in result.changes if item.accepted]
    assert len(accepted) == 10
    assert [item["reason"] for item in result.rejected] == [
        "change_limit_exceeded", "change_limit_exceeded"
    ]


async def test_t23_nine_changes_are_all_accepted(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢很多事。")
    changes = [
        _add("interests.likes", f"兴趣{i}", event.event_id, quote="我喜欢很多事")
        for i in range(9)
    ]
    provider = ScriptedProfileRefinementProvider([_decision(*changes)])

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert len([item for item in result.changes if item.accepted]) == 9


async def test_t24_human_confirmed_target_requires_review(rt):
    person = await _person(rt)
    entry_id = await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="interests.likes",
        content="喜欢爵士乐",
        status="ACTIVE",
        source="USER_CORRECTION",
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )
    event = await _event(rt, person, "其实我更喜欢古典乐。")
    proposal = ProfileChangeProposal(
        operation="CORRECT",
        field_key="interests.likes",
        target_entry_ids=[entry_id],
        proposed_content="喜欢古典乐",
        evidence_event_ids=[event.event_id],
        evidence_quotes=["其实我更喜欢古典乐"],
        confidence=0.99,
    )
    provider = ScriptedProfileRefinementProvider([_decision(proposal)])

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.changes[0].pending_reason == "human_confirmed_target_requires_review"
    applied = await _engine(rt, provider).apply_proposals(result)
    assert applied.applied == []
    stored = await rt.repos.profile.get_entry(entry_id)
    assert stored["content"] == "喜欢爵士乐"


async def test_t25_field_capacity_still_protected(rt):
    person = await _person(rt)
    for index in range(8):
        await rt.repos.profile.insert_entry(
            person_id=person.person_id,
            field_key="communication.preferred_tone",
            content=f"语气{index}",
            status="ACTIVE",
            source="DIRECT_SELF_REPORT",
            confidence=1.0,
            evidence_event_ids=[],
            evidence_memory_ids=[],
        )
    event = await _event(rt, person, "别太正式。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("communication.preferred_tone", "不太正式", event.event_id,
                        quote="别太正式"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)
    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.pending[0]["reason"] == "field_capacity_reached"
    assert len(await _active_entries(rt, person.person_id)) == 8


# -- C. Dedupe / update preference ---------------------------------------


def test_t26_equivalence_unit_cases():
    assert entries_equivalent("喜欢爵士乐", "喜欢爵士乐") is True
    assert entries_equivalent("用户喜欢爵士乐", "喜欢爵士乐") is True
    assert entries_equivalent("正在做一个机器人项目", "做一个机器人项目") is True
    assert entries_equivalent("开发机器狗项目", "开发奶茶机项目") is False
    assert entries_equivalent("机器狗", "机械臂") is False
    assert entries_equivalent("喜欢爵士乐", "喜欢古典乐") is False


async def test_t27_exact_duplicate_add_is_rejected(rt):
    person = await _person(rt)
    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="interests.likes",
        content="喜欢爵士乐",
        status="ACTIVE",
        source="DIRECT_SELF_REPORT",
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )
    event = await _event(rt, person, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("interests.likes", "用户喜欢爵士乐", event.event_id,
                        quote="我喜欢爵士乐"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.status == "REJECTED"
    assert result.rejected[0]["reason"] == "already_present"


async def test_t28_different_entities_are_not_merged(rt):
    person = await _person(rt)
    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="projects.active",
        content="机器狗项目",
        status="ACTIVE",
        source="DIRECT_SELF_REPORT",
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )
    event = await _event(rt, person, "我还在做一个奶茶机项目。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("projects.active", "奶茶机项目", event.event_id,
                        quote="奶茶机项目"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)
    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.status == "APPLIED"
    contents = {
        item["content"] for item in await _active_entries(rt, person.person_id)
    }
    assert contents == {"机器狗项目", "奶茶机项目"}


async def test_t29_same_entity_with_new_detail_is_not_a_duplicate(rt):
    person = await _person(rt)
    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="projects.active",
        content="机器狗项目",
        status="ACTIVE",
        source="DIRECT_SELF_REPORT",
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )
    event = await _event(rt, person, "机器狗项目现在开始做视觉识别了。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("projects.active",
                        "机器狗项目正在加入视觉识别",
                        event.event_id,
                        quote="机器狗项目现在开始做视觉识别了"))]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)

    assert result.status == "PROPOSED"
    assert result.rejected == []


async def test_t30_duplicate_twice_in_one_batch_is_rejected_once(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _add("interests.likes", "喜欢爵士乐", event.event_id,
                 quote="我喜欢爵士乐"),
            _add("interests.likes", "用户喜欢爵士乐", event.event_id,
                 quote="我喜欢爵士乐"),
        )]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)
    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.status == "APPLIED"
    assert len(await _active_entries(rt, person.person_id)) == 1
    assert [item["reason"] for item in applied.rejected] == ["already_present"]


# -- D. Historical recovery ------------------------------------------------


async def _legacy_skipped(rt, person, text, conversation="legacy"):
    event = await _event(rt, person, text, conversation)
    review_id = await rt.repos.profile.ensure_review_receipt(
        person_id=person.person_id,
        source_event_id=event.event_id,
        job_key=profile_review_job_key(person.person_id, event.event_id),
    )
    await rt.repos.profile.update_review_receipt(
        review_id,
        status="SKIPPED",
        decision="SKIPPED",
        reason=LEGACY_SKIPPED_REASON,
        completed=True,
    )
    return event, review_id


async def test_t31_dry_run_writes_nothing(rt):
    person = await _person(rt, ensure=False)
    event, review_id = await _legacy_skipped(rt, person, "我在做机器人。")

    summary = await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=True, limit=30
    )

    assert summary["matched"] == 1
    assert [item["review_id"] for item in summary["would_retry"]] == [review_id]
    assert summary["retried"] == []
    assert await rt.repos.profile.get(person.person_id) is None
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "SKIPPED"
    assert await rt.repos.job.pending() == []


async def test_t32_apply_repairs_profile_and_runs_new_review(rt):
    person = await _person(rt, ensure=False)
    event, review_id = await _legacy_skipped(rt, person, "我在做一个机器人项目。")

    summary = await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=False, limit=30
    )

    assert [item["review_id"] for item in summary["retried"]] == [review_id]
    assert await rt.repos.profile.get(person.person_id) is not None
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "PENDING"
    jobs = [
        job for job in await rt.repos.job.pending()
        if job.job_type == JobType.PROFILE_REVIEW
    ]
    assert len(jobs) == 1

    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("projects.active", "正在做一个机器人项目", event.event_id,
                        quote="我在做一个机器人项目"))]
    )
    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        _engine(rt, provider)
    )
    await rt.worker.run_until_idle()

    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "COMPLETED"
    assert receipt["result_revision"] == 2
    assert [item["content"] for item in await _active_entries(rt, person.person_id)] == [
        "正在做一个机器人项目"
    ]


async def test_t33_apply_is_idempotent(rt):
    person = await _person(rt, ensure=False)
    _, review_id = await _legacy_skipped(rt, person, "我在做机器人。")

    await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=False, limit=30
    )
    second = await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=False, limit=30
    )

    assert second["matched"] == 0
    assert second["retried"] == []
    jobs = [
        job for job in await rt.repos.job.pending()
        if job.job_type == JobType.PROFILE_REVIEW
    ]
    assert len(jobs) == 1


async def test_t34_apply_skips_a_person_without_consent(rt):
    person = await _person(rt, ensure=False, allow_external_ai=False)
    _, review_id = await _legacy_skipped(rt, person, "我在做机器人。")

    summary = await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=False, limit=30
    )

    assert summary["retried"] == []
    assert {item["reason"] for item in summary["skipped"]} == {"not_authorized"}
    assert await rt.repos.profile.get(person.person_id) is None
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "SKIPPED"


async def test_t35_apply_skips_a_deleted_source(rt):
    person = await _person(rt, ensure=False)
    event, review_id = await _legacy_skipped(rt, person, "我在做机器人。")
    await rt.repos.utterance.delete_owned_event(
        event.event_id, speaker_person_id=person.person_id
    )

    summary = await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=False, limit=30
    )

    assert summary["retried"] == []
    assert {item["reason"] for item in summary["skipped"]} == {"source_event_deleted"}
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "SKIPPED"


async def test_t36_apply_skips_an_event_with_an_applied_revision(rt):
    person = await _person(rt)
    event, review_id = await _legacy_skipped(rt, person, "我在做机器人。")
    await rt.repos.profile.insert_revision(
        person_id=person.person_id,
        revision=2,
        changes={"changes": []},
        changed_by="profile_refinement",
        source_event_id=event.event_id,
        reason="prior_apply",
    )

    summary = await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=False, limit=30
    )

    assert summary["retried"] == []
    assert {item["reason"] for item in summary["skipped"]} == {"already_applied"}


async def test_t37_limit_bounds_one_run(rt):
    person = await _person(rt, ensure=False)
    await _legacy_skipped(rt, person, "第一件事。", "c1")
    await _legacy_skipped(rt, person, "第二件事。", "c2")

    summary = await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=False, limit=1
    )

    assert len(summary["retried"]) == 1
    jobs = [
        job for job in await rt.repos.job.pending()
        if job.job_type == JobType.PROFILE_REVIEW
    ]
    assert len(jobs) == 1


async def test_t38_non_legacy_skips_are_never_touched(rt):
    person = await _person(rt, ensure=False)
    event = await _event(rt, person, "我在做机器人。")
    review_id = await rt.repos.profile.ensure_review_receipt(
        person_id=person.person_id,
        source_event_id=event.event_id,
        job_key="legacy",
    )
    await rt.repos.profile.update_review_receipt(
        review_id, status="SKIPPED", decision="SKIPPED",
        reason="source_event_deleted", completed=True,
    )

    summary = await repair_skipped_reviews(
        rt.repos, person_id=person.person_id, dry_run=False, limit=30
    )

    assert summary["matched"] == 0
    assert await rt.repos.profile.get(person.person_id) is None


async def test_t39_other_people_are_out_of_scope(rt):
    owner = await _person(rt, "Owner", ensure=False)
    other = await _person(rt, "Other", ensure=False)
    _, other_review = await _legacy_skipped(rt, other, "我在做机器人。")
    await _legacy_skipped(rt, owner, "我在学吉他。", "own")

    summary = await repair_skipped_reviews(
        rt.repos, person_id=owner.person_id, dry_run=False, limit=30
    )

    assert len(summary["retried"]) == 1
    untouched = await rt.repos.profile.get_review_receipt_by_id(other_review)
    assert untouched["status"] == "SKIPPED"
    assert await rt.repos.profile.get(other.person_id) is None


def test_t40_cli_parser_defaults_to_dry_run():
    from companion.cli import build_parser

    args = build_parser().parse_args(
        ["profile", "repair-reviews", "--person", "person_1"]
    )
    assert args.apply is False
    assert args.limit == 30
    applied = build_parser().parse_args(
        ["profile", "repair-reviews", "--person", "person_1", "--apply",
         "--limit", "5"]
    )
    assert applied.apply is True
    assert applied.limit == 5
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["profile", "repair-reviews", "--person", "p", "--dry-run", "--apply"]
        )


# -- E. End-to-end, concurrency and preserved protections -----------------


async def test_t41_full_turn_worker_apply_reads_back(rt, say):
    person = await _person(rt)

    def build_decision(payload):
        event_id = payload["new_evidence"]["source_event_id"]
        return _acceptance_decision(event_id)

    provider = ScriptedProfileRefinementProvider([build_decision])
    rt.worker.handlers[JobType.PROFILE_REVIEW.value] = build_profile_review_handler(
        _engine(rt, provider)
    )

    report = await say(person.person_id, "agent", ACCEPTANCE_TEXT, conversation="e2e")
    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, report.event_id
    )
    assert receipt["status"] == "PENDING"

    await rt.db.execute(
        "UPDATE processing_jobs SET next_run_at = NULL WHERE job_key = ?",
        (profile_review_job_key(person.person_id, report.event_id),),
    )
    await rt.worker.run_until_idle()

    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, report.event_id
    )
    assert receipt["status"] == "COMPLETED"
    assert receipt["decision"] == "PROPOSE_CHANGES"
    assert receipt["base_revision"] == 1
    assert receipt["result_revision"] == 2

    profile = await rt.profile_service.get(person.person_id)
    assert profile is not None
    assert profile.metadata.revision == 2
    stored = {
        field_key: [entry.content for entry in items]
        for field_key, items in profile.entries.items()
        if items
    }
    assert stored == {
        "knowledge.skills": ["正在学习葡萄牙语"],
        "projects.active": ["正在做一个机器人项目"],
        "interests.likes": ["喜欢爵士乐"],
        "communication.preferred_tone": ["偏好不太正式的回答"],
    }
    evidence = await rt.repos.profile.list_evidence(person.person_id)
    assert {item["source_event_id"] for item in evidence} == {report.event_id}
    assert await rt.repos.profile.list_candidates(person.person_id, status=None) == []


async def test_t42_profile_survives_restart(rt, say):
    from companion.runtime import build_runtime
    from companion.storage.database import Database

    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("interests.likes", "喜欢爵士乐", event.event_id,
                        quote="我喜欢爵士乐"))]
    )
    result = await _review(rt, provider, person.person_id, event.event_id)
    await _engine(rt, provider).apply_proposals(result)

    await rt.db.close()
    reopened = await Database(rt.db.path).connect()
    try:
        await reopened.initialize()
        restarted = build_runtime(reopened, with_worker=False)
        profile = await restarted.profile_service.get(person.person_id)
        assert profile is not None
        assert [entry.content for entry in profile.entries["interests.likes"]] == [
            "喜欢爵士乐"
        ]
    finally:
        await reopened.close()


async def test_t43_stale_revision_is_never_overwritten(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("interests.likes", "喜欢爵士乐", event.event_id,
                        quote="我喜欢爵士乐"))]
    )
    result = await _review(rt, provider, person.person_id, event.event_id)
    await rt.repos.profile.bump_revision(
        person.person_id, expected_revision=1, updated_by="other"
    )

    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.status == "REVISION_CONFLICT"
    assert await _active_entries(rt, person.person_id) == []


async def test_t44_deleted_evidence_invalidates_the_change(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_add("interests.likes", "喜欢爵士乐", event.event_id,
                        quote="我喜欢爵士乐"))]
    )
    result = await _review(rt, provider, person.person_id, event.event_id)
    await rt.repos.utterance.delete_owned_event(
        event.event_id, speaker_person_id=person.person_id
    )

    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.applied == []
    assert [item["reason"] for item in applied.rejected] == [
        "evidence_event_not_accessible"
    ]
    assert await _active_entries(rt, person.person_id) == []


async def test_t45_ordinary_changes_do_not_require_identity_confirmation(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐，也在学葡萄牙语。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _add("interests.likes", "喜欢爵士乐", event.event_id,
                 quote="我喜欢爵士乐"),
            _add("knowledge.skills", "正在学习葡萄牙语", event.event_id,
                 quote="在学葡萄牙语"),
        )]
    )

    result = await _review(rt, provider, person.person_id, event.event_id)
    applied = await _engine(rt, provider).apply_proposals(result)

    assert applied.status == "APPLIED"
    assert applied.pending == []
    assert await rt.repos.profile.list_candidates(person.person_id, status=None) == []
    profile = await rt.profile_service.get(person.person_id)
    assert profile.identity["name"] == "UNKNOWN"
    assert profile.identity["preferred_name"] == "UNKNOWN"
