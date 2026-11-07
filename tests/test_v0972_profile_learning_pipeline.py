"""V0.9.7.2 Section D: one learning pipeline, with a recoverable commit.

Ordinary conversation text may only register a review request. The regex
learner no longer writes the profile or creates candidates, explicit trusted
commands still work, and a failure anywhere in the apply/receipt/summary/retry
sequence leaves the profile consistent instead of half-applied.
"""

from __future__ import annotations

import logging

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.jobs.profile_review import (
    ProfileReviewScheduler,
    build_profile_review_handler,
    reconcile_profile_review_jobs,
)
from companion.people.profile_compiler import compile_profile_context
from companion.people.profile_models import ProfileFieldKey, ProfileSource
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider
from companion.routing.models import DataControlRequest

LOG = logging.getLogger("test.v0972_pipeline")


async def _person(rt, name="Pipeline"):
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


def _engine(rt, provider):
    return ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        provider,
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )


def _handler(rt, provider):
    return build_profile_review_handler(_engine(rt, provider))


def _add(field_key, content, event_id, *, confidence=0.95):
    return ProfileReviewDecision(
        person_id="ignored",
        base_revision=0,
        decision="PROPOSE_CHANGES",
        reason="test",
        changes=[
            ProfileChangeProposal(
                operation="ADD",
                field_key=field_key,
                proposed_content=content,
                evidence_event_ids=[event_id],
                confidence=confidence,
            )
        ],
    )


def _no_change(base_revision):
    return ProfileReviewDecision(
        person_id="ignored",
        base_revision=base_revision,
        decision="NO_CHANGE",
        reason="nothing stable",
        changes=[],
    )


async def _revision(rt, person):
    return (await rt.profile_service.get(person.person_id)).metadata.revision


async def test_reported_speech_never_preemptively_changes_the_profile(rt, say):
    person = await _person(rt)
    await rt.profile_service.set_field(
        person.person_id,
        ProfileFieldKey.FORMALITY.value,
        "casual",
        source=ProfileSource.DIRECT_SELF_REPORT,
        actor="trusted_cli",
    )
    revision = await _revision(rt, person)

    # The legacy regex would have read this quoted sentence as a preference.
    report = await say(
        person.person_id,
        "agent",
        '我昨天看到一个人说"你说话正式一点"。',
        conversation="reported",
    )

    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == revision
    assert profile.communication_value(ProfileFieldKey.FORMALITY) == "casual"
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []

    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, report.event_id
    )
    assert receipt is not None and receipt["status"] == "PENDING"

    # The review may conclude that nothing stable was said, and must not
    # invent an ordinary candidate from the quoted preference.
    await _handler(rt, ScriptedProfileRefinementProvider([_no_change(revision)]))(
        {"person_id": person.person_id, "source_event_id": report.event_id},
        rt.repos,
        LOG,
    )
    assert await _revision(rt, person) == revision
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []


async def test_the_review_reads_the_complete_profile_before_proposing(rt, say):
    person = await _person(rt)
    await rt.profile_service.set_field(
        person.person_id,
        ProfileFieldKey.FORMALITY.value,
        "casual",
        source=ProfileSource.DIRECT_SELF_REPORT,
        actor="trusted_cli",
    )
    revision = await _revision(rt, person)
    report = await say(
        person.person_id, "agent", "我最近在研究STM32。", conversation="stm32"
    )

    seen: list[dict] = []

    def propose(payload):
        seen.append(payload)
        assert payload["profile_revision"] == revision
        settings = payload["existing_profile"]["communication"]["settings"]
        assert settings["communication.formality"] == "casual"
        assert "formality=casual" in payload["compact_summary"]
        # The model is told exactly which field keys it may target.
        assert "projects.active" in payload["allowed_field_keys"]
        # Strict identity is never offered to the model; the local direct
        # self-report path is the only writer.
        assert "identity.name" not in payload["allowed_field_keys"]
        return _add("projects.active", "STM32 学习板", report.event_id)

    await _handler(rt, ScriptedProfileRefinementProvider([propose]))(
        {"person_id": person.person_id, "source_event_id": report.event_id},
        rt.repos,
        LOG,
    )
    assert len(seen) == 1
    entries = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE", "TENTATIVE")
    )
    assert [entry["content"] for entry in entries] == ["STM32 学习板"]
    assert await _revision(rt, person) == revision + 1


async def test_an_explicit_address_phrase_in_chat_writes_directly(rt, say):
    """An explicit self-report is now the command; no confirmation is asked."""
    person = await _person(rt)
    revision = await _revision(rt, person)
    report = await say(
        person.person_id, "agent", "以后叫我小明吧。", conversation="address"
    )

    profile = await rt.profile_service.get(person.person_id)
    # The one statement supplies both a preferred name and the address form
    # it implies; each is a separately audited revision.
    assert profile.metadata.revision == revision + 2
    assert profile.communication_value(ProfileFieldKey.ADDRESS) == "小明"
    entries = await rt.repos.profile.list_entries(
        person.person_id,
        statuses=("ACTIVE", "TENTATIVE"),
        field_key="identity.preferred_name",
    )
    assert [entry["content"] for entry in entries] == ["小明"]
    assert entries[0]["source"] == ProfileSource.DIRECT_SELF_REPORT.value
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []
    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, report.event_id
    )
    # The direct write never pretends the asynchronous review already ran.
    assert receipt is not None and receipt["status"] == "PENDING"


async def test_model_identity_proposal_is_rejected_but_self_report_writes(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我叫阿明。")
    revision = await _revision(rt, person)
    engine = _engine(
        rt, ScriptedProfileRefinementProvider(_add("identity.name", "阿明", event.event_id))
    )
    applied = await engine.apply_proposals(
        await engine.review(person.person_id, event.event_id)
    )
    assert applied.status == "REJECTED"
    assert await _revision(rt, person) == revision
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []

    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="阿明",
        source_event_id=event.event_id,
    )
    assert result.status == "APPLIED"
    assert result.revision == revision + 1
    entries = await rt.repos.profile.list_entries(
        person.person_id, field_key="identity.name"
    )
    assert [entry["content"] for entry in entries] == ["阿明"]
    assert entries[0]["evidence_event_ids"] == [event.event_id]


async def test_a_summary_write_failure_rolls_back_the_whole_review(rt, monkeypatch):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    revision = await _revision(rt, person)
    engine = _engine(
        rt, ScriptedProfileRefinementProvider(_add("projects.active", "机器狗", event.event_id))
    )
    result = await engine.review(person.person_id, event.event_id)

    async def explode(person_id, summary, *, updated_by):
        raise RuntimeError("summary unavailable")

    monkeypatch.setattr(rt.repos.profile, "set_summary", explode)
    with pytest.raises(RuntimeError):
        await engine.apply_proposals(result)

    assert await _revision(rt, person) == revision
    assert await rt.repos.profile.list_entries(person.person_id) == []
    assert (
        await rt.repos.profile.list_revisions(person.person_id)
    ) == []


async def test_a_receipt_write_failure_does_not_duplicate_the_change_on_retry(rt, monkeypatch):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    scheduler = ProfileReviewScheduler(rt.repos.profile, rt.repos.job)
    review_id = await scheduler.request(person.person_id, event.event_id)
    revision = await _revision(rt, person)

    original = rt.repos.profile.update_review_receipt
    failures = {"count": 0}

    async def flaky(review_id_, **kwargs):
        if kwargs.get("status") == "COMPLETED" and failures["count"] == 0:
            failures["count"] += 1
            raise RuntimeError("receipt write unavailable")
        return await original(review_id_, **kwargs)

    monkeypatch.setattr(rt.repos.profile, "update_review_receipt", flaky)
    handler = _handler(
        rt,
        ScriptedProfileRefinementProvider(
            [
                _add("projects.active", "机器狗", event.event_id),
                _add("projects.active", "机器狗", event.event_id),
            ]
        ),
    )
    with pytest.raises(Exception):
        await handler(
            {"person_id": person.person_id, "source_event_id": event.event_id},
            rt.repos,
            LOG,
        )

    # The profile change committed; only its receipt acknowledgement failed.
    entries = await rt.repos.profile.list_entries(person.person_id)
    assert [entry["content"] for entry in entries] == ["机器狗"]
    assert await _revision(rt, person) == revision + 1
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "RUNNING"

    # A retry re-reads and re-applies nothing: same event, same revision.
    assert await scheduler.retry(person.person_id, event.event_id) is not None
    await handler(
        {"person_id": person.person_id, "source_event_id": event.event_id},
        rt.repos,
        LOG,
    )
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "COMPLETED"
    entries = await rt.repos.profile.list_entries(person.person_id)
    assert [entry["content"] for entry in entries] == ["机器狗"]
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert [item["revision"] for item in revisions] == [revision + 1]


async def test_a_revision_conflict_changes_nothing(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    revision = await _revision(rt, person)
    engine = _engine(
        rt, ScriptedProfileRefinementProvider(_add("projects.active", "机器狗", event.event_id))
    )
    result = await engine.review(person.person_id, event.event_id)

    await rt.profile_service.set_field(
        person.person_id,
        ProfileFieldKey.FORMALITY.value,
        "casual",
        source=ProfileSource.DIRECT_SELF_REPORT,
        actor="trusted_cli",
    )
    assert await _revision(rt, person) == revision + 1

    applied = await engine.apply_proposals(result)
    assert applied.status == "REVISION_CONFLICT"
    assert await _revision(rt, person) == revision + 1
    assert await rt.repos.profile.list_entries(person.person_id) == []


async def test_restart_reconciliation_finishes_an_orphaned_review(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做机器狗。")
    scheduler = ProfileReviewScheduler(rt.repos.profile, rt.repos.job)
    review_id = await scheduler.request(person.person_id, event.event_id)

    # Simulate a restart that lost the durable job row.
    await rt.db.execute(
        "DELETE FROM processing_jobs WHERE job_key = ?",
        (f"profile-review:{person.person_id}:{event.event_id}",),
    )
    fixed = await reconcile_profile_review_jobs(rt.repos, person_id=person.person_id)
    assert fixed["enqueued"] == 1

    revision = await _revision(rt, person)
    await _handler(rt, ScriptedProfileRefinementProvider([_no_change(revision)]))(
        {"person_id": person.person_id, "source_event_id": event.event_id},
        rt.repos,
        LOG,
    )
    receipt = await rt.repos.profile.get_review_receipt_by_id(review_id)
    assert receipt["status"] == "COMPLETED"
    assert await _revision(rt, person) == revision


def test_the_refinement_prompt_declares_its_json_schema():
    from companion.prompts.profile_refinement import (
        PROFILE_REFINEMENT_SYSTEM_PROMPT,
    )

    # A real model needs the exact key names; without them it invents its own
    # (observed: {"action": ..., "proposals": ...}) and every review fails.
    for marker in ('"decision"', '"changes"', '"field_key"', "allowed_field_keys"):
        assert marker in PROFILE_REFINEMENT_SYSTEM_PROMPT
    for forbidden in ("action", "proposals"):
        assert forbidden in PROFILE_REFINEMENT_SYSTEM_PROMPT  # named as forbidden


async def test_the_real_provider_takes_ids_from_the_trusted_payload():
    import json
    from types import SimpleNamespace

    from companion.providers.deepseek_profile import (
        DeepSeekProfileRefinementProvider,
    )

    class Client:
        async def complete(self, *args, **kwargs):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "decision": "NO_CHANGE",
                        "reason": "ok",
                        "changes": [],
                        "person_id": "model_invented",
                        "base_revision": 999,
                    }
                ),
                model="fake",
            )

    provider = DeepSeekProfileRefinementProvider(Client())
    decision = await provider.review(
        {"person_id": "person_trusted", "profile_revision": 7}
    )
    assert decision.person_id == "person_trusted"
    assert decision.base_revision == 7
    assert decision.decision == "NO_CHANGE"


async def test_the_summary_drops_a_deleted_entry(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的项目代号是蓝鲸。")
    engine = _engine(
        rt, ScriptedProfileRefinementProvider(_add("projects.active", "蓝鲸", event.event_id))
    )
    await engine.apply_proposals(await engine.review(person.person_id, event.event_id))
    before = await rt.profile_service.get(person.person_id)
    assert "蓝鲸" in before.compact_summary

    pending = await rt.data_control.resolve_request(
        DataControlRequest(
            operation="DELETE",
            target_type="EVENT",
            target_id=event.event_id,
            scope="SELF",
        ),
        event,
    )
    result = await rt.data_control.execute_confirmed(pending.operation_id)
    assert result.status == "COMPLETED"

    after = await rt.profile_service.get(person.person_id)
    assert after.entries == {}
    assert "蓝鲸" not in after.compact_summary
    assert after.compact_summary == compile_profile_context(after)
    assert await rt.repos.profile.list_entries(person.person_id) == []
