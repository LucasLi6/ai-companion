"""V0.9.9.6: Person Profile writes directly, with no candidate queue.

Categories A-E from the release brief:

* A  normal write and dedupe;
* B  explicit identity self-report (the only way to write identity);
* C  data-safety rejections;
* D  transaction, idempotency and concurrency;
* E  one-time migration of the retired candidate table.

Everything is offline: models are scripted and databases are temporary.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event, new_id, utcnow
from companion.people.profile_candidate_migration import (
    migrate_profile_candidates,
)
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
    ProfileReviewExecution,
)
from companion.people.profile_service import MAX_REFINEMENT_CHANGES, ReviewExecutionLost
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider
from companion.storage.database import Database
from companion.storage.schema import MIGRATIONS, SCHEMA_VERSION


# -- helpers ---------------------------------------------------------------


async def _person(rt, name="Direct", **consent):
    values = dict(
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    values.update(consent)
    person = await rt.registry.register(name, "owner", **values)
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


def _source(rt):
    return ProfileReviewContextSource(
        rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
    )


def _engine(rt, decision):
    return ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider([decision]),
        _source(rt),
    )


def _add(
    field_key,
    content,
    event_id,
    *,
    quote=None,
    confidence=0.9,
    confirm=False,
):
    return ProfileChangeProposal(
        operation="ADD",
        field_key=field_key,
        proposed_content=content,
        evidence_event_ids=[event_id] if event_id else [],
        evidence_quotes=[quote] if quote else [],
        confidence=confidence,
        requires_confirmation=confirm,
    )


def _decision(*changes):
    return ProfileReviewDecision(
        person_id="placeholder",
        base_revision=0,
        decision="PROPOSE_CHANGES" if changes else "NO_CHANGE",
        reason="test",
        changes=list(changes),
    )


async def _review(rt, person, event, *proposals):
    engine = _engine(rt, _decision(*proposals))
    result = await engine.review(person.person_id, event.event_id)
    return result, engine


async def _apply(rt, person, event, *proposals):
    result, engine = await _review(rt, person, event, *proposals)
    return result, await engine.apply_proposals(result)


async def _active(rt, person_id, field_key=None):
    return await rt.repos.profile.list_entries(
        person_id, statuses=("ACTIVE", "TENTATIVE"), field_key=field_key
    )


async def _revision(rt, person_id):
    return (await rt.profile_service.get(person_id)).metadata.revision


async def _legacy_candidate(
    rt,
    person,
    field_key,
    value,
    event_ids,
    *,
    status="PENDING",
    operation="ADD",
    confidence=0.7,
):
    now = utcnow().isoformat()
    candidate_id = new_id("pfc")
    await rt.db.execute(
        "INSERT INTO person_profile_candidates_legacy("
        " candidate_id, person_id, field_key, candidate_value_json, source,"
        " confidence, status, evidence_event_ids_json, operation,"
        " target_entry_ids_json, requires_confirmation, review_status,"
        " created_at, updated_at)"
        " VALUES (?, ?, ?, ?, 'INTERACTION_INFERENCE', ?, ?, ?, ?, '[]', 0,"
        " 'PENDING', ?, ?)",
        (
            candidate_id,
            person.person_id,
            field_key,
            json.dumps(value, ensure_ascii=False),
            confidence,
            status,
            json.dumps(list(event_ids), ensure_ascii=False),
            operation,
            now,
            now,
        ),
    )
    return candidate_id


# -- A. normal write -------------------------------------------------------


async def test_a01_one_supported_change_is_written(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    review, applied = await _apply(
        rt,
        person,
        event,
        _add("interests.topics", "喜欢爵士乐", event.event_id, quote="我喜欢爵士乐"),
    )
    assert review.status == "ACCEPTED"
    assert applied.status == "APPLIED"
    entries = await _active(rt, person.person_id)
    assert [(item["field_key"], item["content"]) for item in entries] == [
        ("interests.topics", "喜欢爵士乐")
    ]
    assert await _revision(rt, person.person_id) == 2


async def test_a02_multiple_fields_commit_in_one_revision(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在学葡萄牙语，也在做机器人项目。")
    _, applied = await _apply(
        rt,
        person,
        event,
        _add(
            "knowledge.skills",
            "正在学习葡萄牙语",
            event.event_id,
            quote="我在学葡萄牙语",
        ),
        _add(
            "projects.active",
            "正在做机器人项目",
            event.event_id,
            quote="也在做机器人项目",
        ),
    )
    assert applied.status == "APPLIED"
    assert {item["field_key"] for item in await _active(rt, person.person_id)} == {
        "knowledge.skills",
        "projects.active",
    }
    assert await _revision(rt, person.person_id) == 2


async def test_a03_duplicate_is_no_change_without_a_revision(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。")
    await _apply(
        rt, person, first,
        _add("interests.topics", "喜欢爵士乐", first.event_id, quote="我喜欢爵士乐"),
    )
    revision = await _revision(rt, person.person_id)

    second = await _event(rt, person, "我还是喜欢爵士乐。", "c2")
    review, _ = await _review(
        rt, person, second,
        _add("interests.topics", "用户喜欢爵士乐", second.event_id,
             quote="喜欢爵士乐"),
    )
    assert review.status == "NO_CHANGE"
    assert review.rejected[0]["reason"] == "already_present"
    assert await _revision(rt, person.person_id) == revision
    assert len(await _active(rt, person.person_id)) == 1


async def test_a04_like_and_dislike_are_not_deduped(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢正式回答。")
    await _apply(
        rt, person, first,
        _add("communication.preferred_tone", "喜欢正式回答", first.event_id,
             quote="我喜欢正式回答"),
    )
    second = await _event(rt, person, "其实我不喜欢正式回答。", "c2")
    review, applied = await _apply(
        rt, person, second,
        _add("communication.preferred_tone", "不喜欢正式回答", second.event_id,
             quote="我不喜欢正式回答"),
    )
    assert review.status == "ACCEPTED"
    assert applied.status == "APPLIED"
    contents = {
        item["content"] for item in await _active(rt, person.person_id)
    }
    assert contents == {"喜欢正式回答", "不喜欢正式回答"}


async def test_a05_update_changes_the_same_entity_in_place(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我正在做一个机器人项目。")
    _, applied = await _apply(
        rt, person, first,
        _add("projects.active", "正在做一个机器人项目", first.event_id,
             quote="我正在做一个机器人项目"),
    )
    entry_id = applied.applied[0]["entry_id"]

    second = await _event(rt, person, "机器人项目已经可以走路了。", "c2")
    proposal = ProfileChangeProposal(
        operation="UPDATE",
        field_key="projects.active",
        target_entry_ids=[entry_id],
        proposed_content="机器人项目可以走路了",
        evidence_event_ids=[second.event_id],
        evidence_quotes=["机器人项目已经可以走路了"],
        confidence=0.9,
    )
    _, updated = await _apply(rt, person, second, proposal)
    assert updated.status == "APPLIED"
    entries = await _active(rt, person.person_id, "projects.active")
    assert len(entries) == 1
    assert entries[0]["content"] == "机器人项目可以走路了"


async def test_a06_zero_confidence_is_metadata_not_a_block(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢摄影。")
    _, applied = await _apply(
        rt, person, event,
        _add("interests.topics", "喜欢摄影", event.event_id, quote="我喜欢摄影",
             confidence=0.0),
    )
    assert applied.status == "APPLIED"
    entries = await _active(rt, person.person_id)
    assert entries[0]["confidence"] == 0.0


async def test_a07_model_confirmation_flag_never_blocks_a_write(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢摄影。")
    review, applied = await _apply(
        rt, person, event,
        _add("interests.topics", "喜欢摄影", event.event_id, quote="我喜欢摄影",
             confirm=True),
    )
    assert review.status == "ACCEPTED"
    assert applied.status == "APPLIED"
    assert applied.rejected == []


async def test_a08_receipt_counts_are_content_free(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢摄影，也喜欢露营。")
    _, applied = await _apply(
        rt, person, event,
        _add("interests.topics", "喜欢摄影", event.event_id, quote="喜欢摄影"),
        _add("interests.topics", "喜欢露营", event.event_id, quote="喜欢露营"),
    )
    receipt = applied.receipt()
    assert receipt["status"] == "APPLIED"
    assert receipt["applied_count"] == 2
    assert receipt["rejected_count"] == 0
    assert "摄影" not in json.dumps(receipt, ensure_ascii=False)


# -- B. identity -----------------------------------------------------------


async def test_b01_explicit_name_is_written_directly(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我是李约。")
    result = await rt.profile_service.apply_direct_self_report(
        person.person_id,
        "identity.name",
        "李约",
        source_event_id=event.event_id,
    )
    assert result.status == "APPLIED"
    entries = await _active(rt, person.person_id, "identity.name")
    assert [item["content"] for item in entries] == ["李约"]
    assert entries[0]["source"] == "DIRECT_SELF_REPORT"


async def test_b02_english_preferred_name_is_written_directly(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的英文名是 Lucas。")
    result = await rt.profile_service.apply_direct_self_report(
        person.person_id,
        "identity.preferred_name",
        "Lucas",
        source_event_id=event.event_id,
    )
    assert result.status == "APPLIED"
    assert [
        item["content"]
        for item in await _active(rt, person.person_id, "identity.preferred_name")
    ] == ["Lucas"]


async def test_b03_future_alias_is_written_directly(rt):
    person = await _person(rt)
    event = await _event(rt, person, "以后叫我 Lucas。")
    result = await rt.profile_service.apply_direct_self_report(
        person.person_id,
        "identity.preferred_name",
        "Lucas",
        source_event_id=event.event_id,
    )
    assert result.status == "APPLIED"


async def test_b04_hypothetical_name_writes_nothing(rt):
    person = await _person(rt)
    event = await _event(rt, person, "如果我是李约。")
    result = await rt.profile_service.apply_direct_self_report(
        person.person_id,
        "identity.name",
        "李约",
        source_event_id=event.event_id,
    )
    assert result.status == "REJECTED"
    assert result.reason == "not_an_explicit_self_report"
    assert await _active(rt, person.person_id, "identity.name") == []


async def test_b05_third_party_name_writes_nothing(rt):
    person = await _person(rt)
    event = await _event(rt, person, "他说他叫李约。")
    result = await rt.profile_service.apply_direct_self_report(
        person.person_id,
        "identity.name",
        "李约",
        source_event_id=event.event_id,
    )
    assert result.status == "REJECTED"
    assert await _active(rt, person.person_id, "identity.name") == []


async def test_b06_model_inference_cannot_touch_a_name(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我叫李约。")
    await rt.profile_service.apply_direct_self_report(
        person.person_id,
        "identity.name",
        "李约",
        source_event_id=first.event_id,
    )
    second = await _event(rt, person, "我叫李约。", "c2")
    review, _ = await _review(
        rt, person, second,
        _add("identity.name", "李明", second.event_id, quote="我叫李约"),
    )
    assert review.status == "REJECTED"
    assert (
        review.rejected[0]["reason"]
        == "strict_identity_requires_explicit_self_report"
    )
    assert [
        item["content"]
        for item in await _active(rt, person.person_id, "identity.name")
    ] == ["李约"]


async def test_b07_a_new_explicit_name_replaces_the_old_one(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我叫李约。")
    await rt.profile_service.apply_direct_self_report(
        person.person_id, "identity.name", "李约",
        source_event_id=first.event_id,
    )
    second = await _event(rt, person, "我叫李明。", "c2")
    result = await rt.profile_service.apply_direct_self_report(
        person.person_id, "identity.name", "李明",
        source_event_id=second.event_id,
    )
    assert result.status == "APPLIED"
    assert [
        item["content"]
        for item in await _active(rt, person.person_id, "identity.name")
    ] == ["李明"]
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert revisions[-1]["changes"]["old"] == ["李约"]
    assert revisions[-1]["changes"]["new"] == "李明"


async def test_b08_repeating_the_same_name_is_a_no_op(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我叫李约。")
    await rt.profile_service.apply_direct_self_report(
        person.person_id, "identity.name", "李约",
        source_event_id=first.event_id,
    )
    revision = await _revision(rt, person.person_id)
    second = await _event(rt, person, "我叫李约。", "c2")
    result = await rt.profile_service.apply_direct_self_report(
        person.person_id, "identity.name", "李约",
        source_event_id=second.event_id,
    )
    assert result.status == "NO_CHANGE"
    assert result.reason == "already_present"
    assert await _revision(rt, person.person_id) == revision
    assert len(await _active(rt, person.person_id, "identity.name")) == 1


# -- C. data safety --------------------------------------------------------


async def test_c01_no_evidence_is_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "今天天气不错。")
    proposal = ProfileChangeProposal(
        operation="ADD",
        field_key="interests.topics",
        proposed_content="喜欢晴天",
        confidence=1.0,
    )
    review, _ = await _review(rt, person, event, proposal)
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "no_evidence"


async def test_c02_another_persons_message_is_not_evidence(rt):
    owner = await _person(rt, "Owner")
    other = await _person(rt, "Other")
    event = await _event(rt, owner, "随便聊聊。")
    foreign = await _event(rt, other, "我喜欢摄影。", "c9")
    review, _ = await _review(
        rt, owner, event,
        _add("interests.topics", "喜欢摄影", foreign.event_id,
             quote="我喜欢摄影"),
    )
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "evidence_event_not_authorized"


async def test_c03_a_quote_must_exist_in_the_cited_message(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    review, _ = await _review(
        rt, person, event,
        _add("interests.topics", "喜欢古典乐", event.event_id,
             quote="我喜欢古典乐"),
    )
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "evidence_quote_not_found"


async def test_c04_hypothetical_and_joke_are_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "如果我有时间，我也想学冲浪。")
    review, _ = await _review(
        rt, person, event,
        _add("knowledge.skills", "想学冲浪", event.event_id,
             quote="如果我有时间"),
    )
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "hypothetical_or_reported_evidence"


async def test_c05_reported_third_party_is_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "他说我喜欢潜水。")
    review, _ = await _review(
        rt, person, event,
        _add("interests.topics", "喜欢潜水", event.event_id,
             quote="他说我喜欢潜水"),
    )
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "hypothetical_or_reported_evidence"


async def test_c06_a_single_act_is_not_a_stable_trait(rt):
    person = await _person(rt)
    event = await _event(rt, person, "今天一下子就把问题解决了。")
    review, _ = await _review(
        rt, person, event,
        _add("personality.traits", "行动果断", event.event_id,
             quote="今天一下子就把问题解决了"),
    )
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "trait_inferred_from_single_act"


async def test_c07_sensitive_health_and_financial_text_is_rejected(rt):
    person = await _person(rt)
    health = await _event(rt, person, "我最近在医院用药。")
    financial = await _event(rt, person, "我的工资是两万。", "c2")
    review, _ = await _review(
        rt, person, health,
        _add("background.stable_facts", "最近在医院用药", health.event_id,
             quote="我在医院用药"),
    )
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "sensitive_attribute_not_auto_saved"

    review, _ = await _review(
        rt, person, financial,
        _add("background.stable_facts", "工资两万", financial.event_id,
             quote="我的工资是两万"),
    )
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "sensitive_attribute_not_auto_saved"


async def test_c08_relationship_claims_need_explicit_self_report(rt):
    person = await _person(rt)
    event = await _event(rt, person, "你是我的好朋友。")
    review, _ = await _review(
        rt, person, event,
        _add("relationship.label", "好朋友", event.event_id,
             quote="你是我的好朋友"),
    )
    assert review.status == "REJECTED"
    assert (
        review.rejected[0]["reason"]
        == "strict_identity_requires_explicit_self_report"
    )


async def test_c09_single_value_conflict_is_rejected(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我在学葡萄牙语。")
    await _apply(
        rt, person, first,
        _add("knowledge.skills", "正在学习葡萄牙语", first.event_id,
             quote="我在学葡萄牙语"),
    )
    # An inference cannot silently replace an established single value.
    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="background.occupation",
        content="工程师",
        status="ACTIVE",
        source="DIRECT_SELF_REPORT",
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )
    second = await _event(rt, person, "我在做工程师的工作。", "c2")
    review, _ = await _review(
        rt, person, second,
        _add("background.occupation", "工程师", second.event_id,
             quote="我在做工程师的工作"),
    )
    assert review.status in ("NO_CHANGE", "REJECTED")
    assert (
        review.rejected[0]["reason"]
        in ("already_present", "unresolved_conflict_with_existing_value")
    )
    assert (
        await rt.repos.profile.list_entries(
            person.person_id, field_key="background.occupation"
        )
    )[0]["content"] == "工程师"


async def test_c10_deleted_evidence_cannot_be_written(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    review, engine = await _review(
        rt, person, event,
        _add("interests.topics", "喜欢爵士乐", event.event_id,
             quote="我喜欢爵士乐"),
    )
    await rt.repos.utterance.delete_owned_event(
        event.event_id, speaker_person_id=person.person_id
    )
    applied = await engine.apply_proposals(review)
    assert applied.status == "REJECTED"
    assert applied.rejected[0]["reason"] == "evidence_event_not_accessible"
    assert await _active(rt, person.person_id) == []


async def test_c11_unknown_field_is_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "随便聊聊。")
    review, _ = await _review(
        rt, person, event,
        _add("not.a.field", "随便聊聊", event.event_id),
    )
    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "unknown_field_key"


# -- D. transaction and concurrency ---------------------------------------


async def test_d01_retrying_the_same_event_is_already_applied(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    result, engine = await _review(
        rt, person, event,
        _add("interests.topics", "喜欢爵士乐", event.event_id,
             quote="我喜欢爵士乐"),
    )
    plain = await engine.apply_proposals(result)
    assert plain.status == "APPLIED"
    revision = await _revision(rt, person.person_id)
    again = await engine.apply_proposals(result)
    assert again.status == "ALREADY_APPLIED"
    assert await _revision(rt, person.person_id) == revision
    assert len(await _active(rt, person.person_id)) == 1


async def test_d02_concurrent_writers_never_duplicate(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。")
    second = await _event(rt, person, "我喜欢摄影。", "c2")
    review_a, engine_a = await _review(
        rt, person, first,
        _add("interests.topics", "喜欢爵士乐", first.event_id,
             quote="我喜欢爵士乐"),
    )
    review_b, engine_b = await _review(
        rt, person, second,
        _add("interests.topics", "喜欢摄影", second.event_id,
             quote="我喜欢摄影"),
    )
    a, b = await asyncio.gather(
        engine_a.apply_proposals(review_a),
        engine_b.apply_proposals(review_b),
    )
    assert {a.status, b.status} == {"APPLIED", "REVISION_CONFLICT"}
    assert len(await _active(rt, person.person_id)) == 1


async def test_d03_losing_the_lease_rolls_back_everything(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    review, engine = await _review(
        rt, person, event,
        _add("interests.topics", "喜欢爵士乐", event.event_id,
             quote="我喜欢爵士乐"),
    )
    review_id = await rt.repos.profile.ensure_review_receipt(
        person_id=person.person_id,
        source_event_id=event.event_id,
        job_key="lease-test",
    )
    assert await rt.repos.profile.acquire_review_lease(
        person.person_id, "worker-a", lease_seconds=30
    )
    assert await rt.repos.profile.claim_review_receipt(review_id, "worker-a")
    execution = ProfileReviewExecution(
        person_id=person.person_id,
        review_id=review_id,
        lease_owner_id="worker-a",
        source_event_id=event.event_id,
    )
    await rt.db.execute(
        "UPDATE profile_review_leases SET expires_at = ? WHERE person_id = ?",
        ("2000-01-01T00:00:00+00:00", person.person_id),
    )
    revision = await _revision(rt, person.person_id)
    with pytest.raises(ReviewExecutionLost, match="review_lease_lost"):
        await engine.apply_proposals(review, execution)
    assert await _active(rt, person.person_id) == []
    assert await _revision(rt, person.person_id) == revision


async def test_d04_a_database_failure_commits_nothing(rt, monkeypatch):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    review, engine = await _review(
        rt, person, event,
        _add("interests.topics", "喜欢爵士乐", event.event_id,
             quote="我喜欢爵士乐"),
    )
    revision = await _revision(rt, person.person_id)

    async def explode(**kwargs):
        raise RuntimeError("simulated sqlite failure")

    monkeypatch.setattr(rt.repos.profile, "insert_revision", explode)
    with pytest.raises(RuntimeError, match="simulated sqlite failure"):
        await engine.apply_proposals(review)
    monkeypatch.undo()
    assert await _active(rt, person.person_id) == []
    assert await _revision(rt, person.person_id) == revision
    assert await rt.repos.profile.list_revisions(person.person_id) == []


async def test_d05_revision_records_its_source_and_change(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    await _apply(
        rt, person, event,
        _add("interests.topics", "喜欢爵士乐", event.event_id,
             quote="我喜欢爵士乐"),
    )
    revision = (await rt.repos.profile.list_revisions(person.person_id))[-1]
    assert revision["revision"] == 2
    assert revision["source_event_id"] == event.event_id
    assert revision["changed_by"] == "profile_refinement"
    change = revision["changes"]["changes"][0]
    assert change["field_key"] == "interests.topics"
    assert change["after"] == "喜欢爵士乐"


async def test_d06_duplicate_in_one_batch_is_written_once(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    review, applied = await _apply(
        rt, person, event,
        _add("interests.topics", "喜欢爵士乐", event.event_id,
             quote="我喜欢爵士乐"),
        _add("interests.topics", "用户喜欢爵士乐", event.event_id,
             quote="我喜欢爵士乐"),
    )
    assert applied.status == "APPLIED"
    assert len(await _active(rt, person.person_id)) == 1
    assert [item["reason"] for item in applied.rejected] == ["already_present"]


async def test_d07_field_capacity_is_enforced(rt):
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
    _, applied = await _apply(
        rt, person, event,
        _add("communication.preferred_tone", "不太正式", event.event_id,
             quote="别太正式"),
    )
    assert applied.status == "REJECTED"
    assert applied.rejected[0]["reason"] == "field_capacity_reached"
    assert len(await _active(rt, person.person_id)) == 8


# -- E. migration ----------------------------------------------------------


async def test_e01_dry_run_writes_nothing(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我是李约。")
    await _legacy_candidate(
        rt, person, "identity.name", "李约", [event.event_id]
    )
    revision = await _revision(rt, person.person_id)

    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=True
    )
    assert summary["scanned"] == 1
    assert summary["results"][0]["status"] == "WOULD_APPLY"
    assert summary["applied"] == 0
    assert await _active(rt, person.person_id) == []
    assert await _revision(rt, person.person_id) == revision
    assert await rt.repos.profile.get_candidate_migration(
        summary["results"][0]["candidate_id"]
    ) is None


async def test_e02_apply_writes_an_explicit_name_with_audit(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我是李约。")
    candidate_id = await _legacy_candidate(
        rt, person, "identity.name", "李约", [event.event_id]
    )
    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert summary["applied"] == 1
    entries = await _active(rt, person.person_id, "identity.name")
    assert [item["content"] for item in entries] == ["李约"]
    assert entries[0]["source"] == "DIRECT_SELF_REPORT"
    audit = await rt.repos.profile.get_candidate_migration(candidate_id)
    assert audit["status"] == "APPLIED"
    assert audit["source_event_id"] == event.event_id
    assert audit["entry_id"] == entries[0]["entry_id"]
    assert audit["revision"] == 2


async def test_e03_apply_is_idempotent(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我叫李约。")
    await _legacy_candidate(
        rt, person, "identity.name", "李约", [event.event_id]
    )
    first = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    revision = await _revision(rt, person.person_id)
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    second = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert first["applied"] == 1
    assert second["applied"] == 0
    assert second["already_migrated"] == 1
    assert await _revision(rt, person.person_id) == revision
    assert len(await rt.repos.profile.list_revisions(person.person_id)) == len(
        revisions
    )
    assert len(await _active(rt, person.person_id, "identity.name")) == 1


async def test_e04_candidate_without_evidence_is_skipped(rt):
    person = await _person(rt)
    candidate_id = await _legacy_candidate(
        rt, person, "identity.name", "李约", []
    )
    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert summary["skipped"] == 1
    assert summary["results"][0]["reason"] == "no_evidence"
    assert await _active(rt, person.person_id, "identity.name") == []
    audit = await rt.repos.profile.get_candidate_migration(candidate_id)
    assert audit["status"] == "SKIPPED"


async def test_e05_explanatory_nickname_text_is_not_migrated(rt):
    person = await _person(rt)
    event = await _event(rt, person, "lucas 这个名字不错。")
    await _legacy_candidate(
        rt,
        person,
        "identity.nicknames",
        "用户喜欢被叫做 Lucas 的昵称",
        [event.event_id],
    )
    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert summary["skipped"] == 1
    assert summary["results"][0]["reason"] == "nickname_not_a_reliable_alias"
    assert await _active(rt, person.person_id, "identity.nicknames") == []


async def test_e06_inferred_tone_is_not_migrated(rt):
    person = await _person(rt)
    event = await _event(rt, person, "你可以采用列表式回答吗")
    await _legacy_candidate(
        rt,
        person,
        "communication.preferred_tone",
        "采用列表式回答",
        [event.event_id],
    )
    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert summary["skipped"] == 1
    assert summary["results"][0]["reason"] == "preference_not_explicitly_stated"
    assert await _active(rt, person.person_id) == []


async def test_e07_objective_recurring_pattern_is_migrated(rt):
    person = await _person(rt)
    first = await _event(rt, person, "你记住我的名字了吗", "c1")
    second = await _event(rt, person, "你有没有保存我的信息", "c2")
    await _legacy_candidate(
        rt,
        person,
        "interaction.recurring_patterns",
        "曾多次询问 Ava 是否记住其姓名，以及是否保存了相关信息",
        [first.event_id, second.event_id],
    )
    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert summary["applied"] == 1
    entries = await _active(
        rt, person.person_id, "interaction.recurring_patterns"
    )
    assert len(entries) == 1
    assert "记住" in entries[0]["content"]
    assert set(entries[0]["evidence_event_ids"]) == {
        first.event_id,
        second.event_id,
    }


async def test_e08_sensitive_candidate_is_skipped(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我最近在医院用药。")
    await _legacy_candidate(
        rt, person, "background.stable_facts", "最近在医院用药",
        [event.event_id],
    )
    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert summary["skipped"] == 1
    assert summary["results"][0]["reason"] == "sensitive_attribute_not_auto_saved"
    assert await _active(rt, person.person_id) == []


async def test_e09_runtime_has_no_candidate_write_api(rt):
    cur = await rt.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name IN ('person_profile_candidates',"
        " 'person_profile_candidates_legacy')"
    )
    names = {row["name"] for row in await cur.fetchall()}
    assert names == {"person_profile_candidates_legacy"}
    assert not hasattr(rt.repos.profile, "insert_candidate")
    assert not hasattr(rt.repos.profile, "list_candidates")
    assert not hasattr(rt.repos.profile, "get_candidate")


async def test_e10_an_old_database_upgrades_without_losing_rows(tmp_path):
    path = tmp_path / "old.memory"
    db = await Database(path).connect()
    await db.conn.execute(
        "CREATE TABLE schema_version ("
        " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    from companion.storage.database import _statements

    for number, sql in MIGRATIONS[:29]:
        for statement in _statements(sql):
            await db.conn.execute(statement)
        await db.conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (number, "2026-01-01T00:00:00+00:00"),
        )
    await db.conn.commit()
    now = "2026-01-01T00:00:00+00:00"
    await db.execute(
        "INSERT INTO persons(person_id, display_name, role, created_at,"
        " updated_at) VALUES ('person_old', 'Old', 'owner', ?, ?)",
        (now, now),
    )
    await db.execute(
        "INSERT INTO person_profile_candidates(candidate_id, person_id,"
        " field_key, candidate_value_json, source, confidence, status,"
        " evidence_event_ids_json, created_at, updated_at)"
        " VALUES ('pfc_old', 'person_old', 'identity.name', '\"李约\"',"
        " 'INTERACTION_INFERENCE', 0.6, 'PENDING', '[]', ?, ?)",
        (now, now),
    )
    await db.close()

    upgraded = await Database(path).connect()
    try:
        await upgraded.initialize()
        cur = await upgraded.execute("SELECT MAX(version) AS v FROM schema_version")
        assert (await cur.fetchone())["v"] == SCHEMA_VERSION == 32
        cur = await upgraded.execute(
            "SELECT candidate_id, field_key FROM"
            " person_profile_candidates_legacy WHERE person_id='person_old'"
        )
        row = await cur.fetchone()
        assert row["candidate_id"] == "pfc_old"
        assert row["field_key"] == "identity.name"
        cur = await upgraded.execute(
            "SELECT COUNT(*) AS n FROM person_profile_entries"
            " WHERE person_id='person_old'"
        )
        assert (await cur.fetchone())["n"] == 0
    finally:
        await upgraded.close()


async def test_e11_the_change_limit_still_applies(rt):
    person = await _person(rt)
    text = (
        "我喜欢爵士乐、摄影、露营、徒步、钓鱼、跑步、游泳、编程、写作、"
        "咖啡、电影、旅行。"
    )
    event = await _event(rt, person, text)
    labels = [
        "爵士乐", "摄影", "露营", "徒步", "钓鱼", "跑步",
        "游泳", "编程", "写作", "咖啡", "电影", "旅行",
    ]
    review, _ = await _review(
        rt,
        person,
        event,
        *[
            _add("interests.topics", f"喜欢{label}", event.event_id, quote=label)
            for label in labels
        ],
    )
    assert MAX_REFINEMENT_CHANGES == 10
    assert len(review.accepted_changes()) == 10
    assert [item["reason"] for item in review.rejected] == [
        "change_limit_exceeded",
        "change_limit_exceeded",
    ]


async def test_e12_name_candidate_is_verified_against_the_persons_own_words(rt):
    person = await _person(rt)
    cited = await _event(rt, person, "李约，lucas，记住了吗", "c1")
    explicit = await _event(rt, person, "我是李约", "c2")
    await _legacy_candidate(
        rt, person, "identity.name", "李约", [cited.event_id]
    )
    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert summary["applied"] == 1
    entries = await _active(rt, person.person_id, "identity.name")
    assert [item["content"] for item in entries] == ["李约"]
    # The write cites the sentence that actually states the name.
    assert entries[0]["evidence_event_ids"] == [explicit.event_id]


async def test_e13_nickname_migration_keeps_only_the_alias(rt):
    person = await _person(rt)
    event = await _event(rt, person, "李约，lucas，记住了吗")
    await _legacy_candidate(
        rt,
        person,
        "identity.nicknames",
        "Lucas（本人给出的英文名/别名）",
        [event.event_id],
    )
    summary = await migrate_profile_candidates(
        rt.profile_service, person.person_id, dry_run=False
    )
    assert summary["applied"] == 1
    entries = await _active(rt, person.person_id, "identity.nicknames")
    assert [item["content"] for item in entries] == ["Lucas"]
