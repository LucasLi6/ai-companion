"""V0.9.9.24 P2.1: cross-category semantic duplicate defence.

Every model response is scripted and every database is a temporary SQLite file.
These tests pin the write-boundary rules that a prompt alone cannot guarantee:

* the same fact stored under another field is not added twice;
* the same fact is attached as evidence to the existing entry instead;
* a different attribute is never merged, even when the text looks alike;
* Chinese attribute synonyms resolve to the same canonical attribute;
* contradictory polarity/time semantics are reported as unresolved.
"""

from __future__ import annotations

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_facts import ExtractedPersonFact, fact_index_row
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider


async def _person(rt, name="Semantic"):
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


def _source(rt):
    return ProfileReviewContextSource(
        rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
    )


def _engine(rt, provider):
    return ProfileRefinementEngine(
        rt.profile_service, rt.repos.profile, provider, _source(rt)
    )


def _fact(
    fact_id,
    value,
    *,
    attribute,
    events,
    quotes=(),
    assertion="DIRECT",
    scope="STABLE",
):
    return ExtractedPersonFact(
        fact_id=fact_id,
        entity="self",
        attribute=attribute,
        value=value,
        assertion_type=assertion,
        temporal_scope=scope,
        evidence_event_ids=list(events),
        evidence_quotes=list(quotes),
    )


def _change(field_key, *, fact, relation="DISTINCT", targets=(), content=None):
    return ProfileChangeProposal(
        operation="ADD",
        field_key=field_key,
        target_entry_ids=list(targets),
        fact_id=fact.fact_id,
        relation=relation,
        proposed_content=content or fact.value,
        evidence_event_ids=list(fact.evidence_event_ids),
        evidence_quotes=list(fact.evidence_quotes),
        confidence=0.9,
    )


def _decision(*changes, facts=()):
    return ProfileReviewDecision(
        person_id="placeholder",
        base_revision=0,
        decision="PROPOSE_CHANGES",
        reason="test",
        facts=list(facts),
        changes=list(changes),
    )


async def _run(rt, person, event, provider):
    engine = _engine(rt, provider)
    review = await engine.review(person.person_id, event.event_id)
    applied = await engine.apply_proposals(review)
    return review, applied


async def _entries(rt, person_id, field_key=None):
    return await rt.repos.profile.list_entries(
        person_id,
        statuses=("ACTIVE", "TENTATIVE"),
        field_key=field_key,
    )


# -- service-level: duplicate across fields ---------------------------------


async def test_cross_category_duplicate_add_is_redirected_to_same(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我在机器人公司工作过。", "c1")
    first_fact = _fact(
        "f1",
        "在机器人公司工作过",
        attribute="experience",
        events=[first.event_id],
        quotes=["我在机器人公司工作过"],
    )
    review, applied = await _run(
        rt,
        person,
        first,
        ScriptedProfileRefinementProvider(
            [_decision(
                _change(
                    "knowledge.experience",
                    fact=first_fact,
                    content="在机器人公司工作过",
                ),
                facts=[first_fact],
            )]
        ),
    )
    assert review.status == "ACCEPTED", review.rejected
    assert applied.status == "APPLIED"
    entry_id = (await _entries(rt, person.person_id))[0]["entry_id"]

    second = await _event(rt, person, "我以前在机器人公司工作过。", "c2")
    second_fact = _fact(
        "f1",
        "在机器人公司工作过",
        attribute="experience",
        events=[second.event_id],
        quotes=["我以前在机器人公司工作过"],
    )
    review, applied = await _run(
        rt,
        person,
        second,
        ScriptedProfileRefinementProvider(
            [_decision(
                _change(
                    "background.history",
                    fact=second_fact,
                    content="在机器人公司工作过",
                ),
                facts=[second_fact],
            )]
        ),
    )

    # No second copy: the change was redirected to evidence-only SAME.
    assert review.status == "ACCEPTED", review.rejected
    assert applied.status == "NO_CHANGE"
    entries = await _entries(rt, person.person_id)
    assert [(item["field_key"], item["content"]) for item in entries] == [
        ("knowledge.experience", "在机器人公司工作过")
    ]
    assert entries[0]["entry_id"] == entry_id
    evidence = await rt.repos.profile.list_evidence(person.person_id)
    assert any(
        item["field_key"] == "knowledge.experience"
        and item["source_event_id"] == second.event_id
        for item in evidence
    )


async def test_different_attribute_is_never_merged(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我会弹吉他。", "c1")
    skill_fact = _fact(
        "f1",
        "会弹吉他",
        attribute="skill",
        events=[first.event_id],
        quotes=["我会弹吉他"],
    )
    await _run(
        rt,
        person,
        first,
        ScriptedProfileRefinementProvider(
            [_decision(
                _change("knowledge.skills", fact=skill_fact),
                facts=[skill_fact],
            )]
        ),
    )
    assert len(await _entries(rt, person.person_id, "knowledge.skills")) == 1

    second = await _event(rt, person, "我喜欢弹吉他。", "c2")
    interest_fact = _fact(
        "f1",
        "喜欢弹吉他",
        attribute="interest",
        events=[second.event_id],
        quotes=["我喜欢弹吉他"],
    )
    review, applied = await _run(
        rt,
        person,
        second,
        ScriptedProfileRefinementProvider(
            [_decision(
                _change("interests.likes", fact=interest_fact),
                facts=[interest_fact],
            )]
        ),
    )

    assert review.status == "ACCEPTED", review.rejected
    assert applied.status == "APPLIED"
    assert len(await _entries(rt, person.person_id, "knowledge.skills")) == 1
    assert len(await _entries(rt, person.person_id, "interests.likes")) == 1


async def test_chinese_attribute_synonyms_collide(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我的爱好是爵士乐。", "c1")
    first_fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="爱好",
        events=[first.event_id],
        quotes=["我的爱好是爵士乐"],
    )
    await _run(
        rt,
        person,
        first,
        ScriptedProfileRefinementProvider(
            [_decision(
                _change("interests.topics", fact=first_fact),
                facts=[first_fact],
            )]
        ),
    )

    second = await _event(rt, person, "爵士乐是我的兴趣。", "c2")
    second_fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="兴趣",
        events=[second.event_id],
        quotes=["爵士乐是我的兴趣"],
    )
    review, applied = await _run(
        rt,
        person,
        second,
        ScriptedProfileRefinementProvider(
            [_decision(
                _change("knowledge.domains", fact=second_fact),
                facts=[second_fact],
            )]
        ),
    )

    assert review.status == "ACCEPTED", review.rejected
    assert applied.status == "NO_CHANGE"
    assert len(await _entries(rt, person.person_id)) == 1


async def test_different_detail_extends_without_deleting_the_original(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我正在做一个机器人项目。", "c1")
    first_fact = _fact(
        "f1",
        "正在做一个机器人项目",
        attribute="project",
        events=[first.event_id],
        quotes=["我正在做一个机器人项目"],
        scope="ONGOING",
    )
    await _run(
        rt,
        person,
        first,
        ScriptedProfileRefinementProvider(
            [_decision(
                _change("projects.active", fact=first_fact),
                facts=[first_fact],
            )]
        ),
    )

    second = await _event(rt, person, "机器人项目下周要参加比赛。", "c2")
    second_fact = _fact(
        "f1",
        "机器人项目下周要参加比赛",
        attribute="project",
        events=[second.event_id],
        quotes=["机器人项目下周要参加比赛"],
        scope="ONGOING",
    )
    review, applied = await _run(
        rt,
        person,
        second,
        ScriptedProfileRefinementProvider(
            [_decision(
                _change("projects.active", fact=second_fact),
                facts=[second_fact],
            )]
        ),
    )

    assert review.status == "ACCEPTED", review.rejected
    assert applied.status == "APPLIED"
    contents = {
        item["content"] for item in await _entries(rt, person.person_id)
    }
    assert contents == {"正在做一个机器人项目", "机器人项目下周要参加比赛"}


# -- repository-level: unresolved semantics ---------------------------------


async def _seed_indexed(rt, person, field_key, content, fact):
    entry_id = await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key=field_key,
        content=content,
        status="ACTIVE",
        source="INTERACTION_INFERENCE",
        confidence=0.9,
        evidence_event_ids=list(fact.evidence_event_ids),
        evidence_memory_ids=[],
    )
    await rt.repos.profile.upsert_fact_index(
        fact_index_row(
            person_id=person.person_id,
            entry_id=entry_id,
            field_key=field_key,
            content=content,
            fact=fact,
        )
    )
    return entry_id


async def test_correction_of_an_identical_value_is_unresolved(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    stored = _fact(
        "f1", "喜欢爵士乐", attribute="interest", events=[event.event_id]
    )
    await _seed_indexed(
        rt, person, "interests.likes", "喜欢爵士乐", stored
    )
    incoming = _fact(
        "f2",
        "喜欢爵士乐",
        attribute="interest",
        events=[event.event_id],
        assertion="CORRECTION",
    )

    collisions = await rt.repos.profile.find_semantic_collisions(
        person.person_id, incoming, field_key="interests.likes"
    )

    assert len(collisions) == 1
    assert collisions[0]["relation"] == "SEMANTIC_UNRESOLVED"
    assert collisions[0]["reason"] == "explicit_correction_matches_stored_fact"


async def test_incompatible_time_scopes_are_unresolved(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我正在学吉他。")
    stored = _fact(
        "f1",
        "学吉他",
        attribute="goal",
        events=[event.event_id],
        scope="ONGOING",
    )
    await _seed_indexed(rt, person, "goals.short_term", "学吉他", stored)
    incoming = _fact(
        "f2",
        "学吉他",
        attribute="goal",
        events=[event.event_id],
        scope="FUTURE",
    )

    collisions = await rt.repos.profile.find_semantic_collisions(
        person.person_id, incoming, field_key="goals.general"
    )

    assert len(collisions) == 1
    assert collisions[0]["relation"] == "SEMANTIC_UNRESOLVED"
    assert collisions[0]["reason"] == "time_scope_conflict"


async def test_fact_index_fallback_sees_entries_without_rows(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key="interests.likes",
        content="喜欢爵士乐",
        status="ACTIVE",
        source="INTERACTION_INFERENCE",
        confidence=0.9,
        evidence_event_ids=[event.event_id],
        evidence_memory_ids=[],
    )
    incoming = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="likes",
        events=[event.event_id],
    )

    collisions = await rt.repos.profile.find_semantic_collisions(
        person.person_id, incoming, field_key="interests.likes"
    )

    assert [item["relation"] for item in collisions] == ["DUPLICATE"]


# -- P2.2: SAME vs content modification on human-confirmed facts ------------


async def _human_confirmed_entry(rt, person, field_key, content, events):
    return await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key=field_key,
        content=content,
        status="ACTIVE",
        source="USER_CORRECTION",
        confidence=1.0,
        evidence_event_ids=list(events),
        evidence_memory_ids=[],
    )


async def test_same_may_add_evidence_to_a_human_confirmed_fact(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。", "c1")
    entry_id = await _human_confirmed_entry(
        rt, person, "interests.likes", "喜欢爵士乐", [first.event_id]
    )

    second = await _event(rt, person, "我还是喜欢爵士乐。", "c2")
    fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="interest",
        events=[second.event_id],
        quotes=["我还是喜欢爵士乐"],
    )
    review, applied = await _run(
        rt,
        person,
        second,
        ScriptedProfileRefinementProvider(
            [_decision(
                ProfileChangeProposal(
                    operation="SAME",
                    field_key="interests.likes",
                    target_entry_ids=[entry_id],
                    fact_id=fact.fact_id,
                    fact=fact,
                    relation="SAME",
                    evidence_event_ids=[second.event_id],
                    evidence_quotes=["我还是喜欢爵士乐"],
                ),
                facts=[fact],
            )]
        ),
    )

    assert review.status == "ACCEPTED", review.rejected
    assert applied.status == "NO_CHANGE"
    entry = await rt.repos.profile.get_entry(entry_id)
    assert entry["content"] == "喜欢爵士乐"
    assert entry["source"] == "USER_CORRECTION"
    evidence = await rt.repos.profile.list_evidence(person.person_id)
    assert any(
        item["source_event_id"] == second.event_id
        for item in evidence
    )


async def test_update_of_a_human_confirmed_fact_is_still_protected(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。", "c1")
    entry_id = await _human_confirmed_entry(
        rt, person, "interests.likes", "喜欢爵士乐", [first.event_id]
    )

    second = await _event(rt, person, "我现在喜欢蓝调。", "c2")
    fact = _fact(
        "f1",
        "喜欢蓝调",
        attribute="interest",
        events=[second.event_id],
        quotes=["我现在喜欢蓝调"],
    )
    review, _ = await _run(
        rt,
        person,
        second,
        ScriptedProfileRefinementProvider(
            [_decision(
                ProfileChangeProposal(
                    operation="UPDATE",
                    field_key="interests.likes",
                    target_entry_ids=[entry_id],
                    fact_id=fact.fact_id,
                    fact=fact,
                    relation="EXTENDS",
                    proposed_content="喜欢蓝调",
                    evidence_event_ids=[second.event_id],
                    evidence_quotes=["我现在喜欢蓝调"],
                ),
                facts=[fact],
            )]
        ),
    )

    assert review.status == "REJECTED"
    assert any(
        item["reason"] == "human_confirmed_target_requires_review"
        for item in review.rejected
    )
    entry = await rt.repos.profile.get_entry(entry_id)
    assert entry["content"] == "喜欢爵士乐"


async def test_evidence_insert_is_idempotent(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")

    first = await rt.repos.profile.insert_evidence(
        person_id=person.person_id,
        field_key="interests.likes",
        source_event_id=event.event_id,
        evidence_type="EVENT",
    )
    second = await rt.repos.profile.insert_evidence(
        person_id=person.person_id,
        field_key="interests.likes",
        source_event_id=event.event_id,
        evidence_type="EVENT",
    )

    assert first == second
    evidence = await rt.repos.profile.list_evidence(person.person_id)
    assert len(evidence) == 1


# -- P2.3: conservative same-field consolidation ----------------------------


def test_same_field_relation_distinguishes_same_extends_and_distinct():
    from companion.people.profile_service import _same_field_relation

    assert _same_field_relation("保持正式", "保持正式") == "SAME"
    assert _same_field_relation("保持正式", "保持正式且简洁") == "EXTENDS"
    assert _same_field_relation("喜欢爵士乐", "喜欢摄影") == "DISTINCT"
    assert _same_field_relation("喜欢爵士乐", "不喜欢爵士乐") == "UNRESOLVED"
    assert _same_field_relation("", "喜欢爵士乐") == "UNRESOLVED"


async def _insert_with_source(rt, person, field_key, content, source, events):
    return await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key=field_key,
        content=content,
        status="ACTIVE",
        source=source,
        confidence=1.0 if source == "USER_CORRECTION" else 0.9,
        evidence_event_ids=list(events),
        evidence_memory_ids=[],
    )


async def test_consolidation_keeps_a_human_confirmed_canonical(rt):
    person = await _person(rt)
    event = await _event(rt, person, "说话正式且简洁。")
    confirmed_id = await _insert_with_source(
        rt,
        person,
        "communication.preferred_tone",
        "保持正式且简洁",
        "USER_CORRECTION",
        [event.event_id],
    )
    inference_id = await _insert_with_source(
        rt,
        person,
        "communication.preferred_tone",
        "保持正式",
        "INTERACTION_INFERENCE",
        [event.event_id],
    )

    result = await rt.profile_service.consolidate(
        person.person_id, field_keys=["communication.preferred_tone"]
    )

    assert result.status == "APPLIED"
    assert result.entries_merged == 1
    confirmed = await rt.repos.profile.get_entry(confirmed_id)
    inference = await rt.repos.profile.get_entry(inference_id)
    assert confirmed["status"] == "ACTIVE"
    assert confirmed["content"] == "保持正式且简洁"
    assert confirmed["source"] == "USER_CORRECTION"
    assert inference["status"] == "MERGED"
    assert inference["superseded_by"] == confirmed_id


async def test_consolidation_never_archives_a_human_confirmed_row(rt):
    person = await _person(rt)
    event = await _event(rt, person, "说话正式一点。")
    short_id = await _insert_with_source(
        rt,
        person,
        "communication.preferred_tone",
        "保持正式",
        "USER_CORRECTION",
        [event.event_id],
    )
    long_id = await _insert_with_source(
        rt,
        person,
        "communication.preferred_tone",
        "保持正式且简洁",
        "INTERACTION_INFERENCE",
        [event.event_id],
    )

    result = await rt.profile_service.consolidate(
        person.person_id, field_keys=["communication.preferred_tone"]
    )

    # The inference has detail the confirmed row lacks; discarding it would
    # lose "简洁", so nothing is merged.
    assert result.entries_merged == 0
    assert (await rt.repos.profile.get_entry(short_id))["status"] == "ACTIVE"
    assert (await rt.repos.profile.get_entry(long_id))["status"] == "ACTIVE"


async def test_consolidation_leaves_two_confirmations_alone(rt):
    person = await _person(rt)
    event = await _event(rt, person, "说话正式一点。")
    first_id = await _insert_with_source(
        rt,
        person,
        "communication.preferred_tone",
        "保持正式",
        "USER_CORRECTION",
        [event.event_id],
    )
    second_id = await _insert_with_source(
        rt,
        person,
        "communication.preferred_tone",
        "保持正式且简洁",
        "USER_CORRECTION",
        [event.event_id],
    )

    result = await rt.profile_service.consolidate(
        person.person_id, field_keys=["communication.preferred_tone"]
    )

    assert result.entries_merged == 0
    assert (await rt.repos.profile.get_entry(first_id))["status"] == "ACTIVE"
    assert (await rt.repos.profile.get_entry(second_id))["status"] == "ACTIVE"


# -- P2.4: batch limits, retry idempotency and atomic rollback ---------------


def test_batch_limits_share_one_review_constant():
    from companion.people import profile_service
    from companion.people.profile_refinement import (
        MAX_CHANGES_PER_REVIEW,
        MAX_MODEL_CHANGES,
    )

    assert MAX_CHANGES_PER_REVIEW == 10
    assert profile_service.MAX_REFINEMENT_CHANGES == MAX_CHANGES_PER_REVIEW
    # The transport envelope must be larger than one review's write limit;
    # overflow is recorded locally, not clipped below what the review uses.
    assert MAX_MODEL_CHANGES > MAX_CHANGES_PER_REVIEW


async def test_retrying_the_same_source_event_does_not_duplicate(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="interest",
        events=[event.event_id],
        quotes=["我喜欢爵士乐"],
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(_change("interests.likes", fact=fact), facts=[fact])]
    )
    review, applied = await _run(rt, person, event, provider)
    assert applied.status == "APPLIED"

    second = await _engine(rt, provider).apply_proposals(review)
    assert second.status == "ALREADY_APPLIED"
    assert len(await _entries(rt, person.person_id)) == 1

    cur = await rt.db.execute(
        "SELECT COUNT(*) FROM person_profile_evidence WHERE person_id = ?",
        (person.person_id,),
    )
    assert (await cur.fetchone())[0] == 1


async def test_failed_second_change_rolls_back_the_first(rt, monkeypatch):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐，也喜欢蓝调。")
    first_fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="interest",
        events=[event.event_id],
        quotes=["我喜欢爵士乐"],
    )
    second_fact = _fact(
        "f2",
        "喜欢蓝调",
        attribute="interest",
        events=[event.event_id],
        quotes=["也喜欢蓝调"],
    )
    provider = ScriptedProfileRefinementProvider(
        [
            _decision(
                _change("interests.likes", fact=first_fact),
                _change("interests.topics", fact=second_fact),
                facts=[first_fact, second_fact],
            )
        ]
    )
    review = await _engine(rt, provider).review(
        person.person_id, event.event_id
    )
    assert review.status == "ACCEPTED", review.rejected
    before = await rt.repos.profile.get(person.person_id)

    original = rt.profile_service._apply_change_locked
    calls = {"count": 0}

    async def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated write failure")
        return await original(*args, **kwargs)

    monkeypatch.setattr(rt.profile_service, "_apply_change_locked", flaky)

    with pytest.raises(RuntimeError):
        await _engine(rt, provider).apply_proposals(review)

    assert await _entries(rt, person.person_id) == []
    after = await rt.repos.profile.get(person.person_id)
    assert after["revision"] == before["revision"]
    cur = await rt.db.execute(
        "SELECT COUNT(*) FROM person_profile_revisions WHERE person_id = ?",
        (person.person_id,),
    )
    assert (await cur.fetchone())[0] == 0


async def test_revision_conflict_never_replays_the_old_proposal(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="interest",
        events=[event.event_id],
        quotes=["我喜欢爵士乐"],
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(_change("interests.likes", fact=fact), facts=[fact])]
    )
    engine = _engine(rt, provider)
    review = await engine.review(person.person_id, event.event_id)
    assert review.status == "ACCEPTED"

    # Someone else commits first: the old decision's base_revision is stale.
    base = int((await rt.repos.profile.get(person.person_id))["revision"])
    await rt.repos.profile.bump_revision(
        person.person_id,
        expected_revision=base,
        updated_by="test",
    )

    applied = await engine.apply_proposals(review)
    assert applied.status == "REVISION_CONFLICT"
    assert await _entries(rt, person.person_id) == []
