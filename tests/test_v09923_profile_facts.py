"""V0.9.9.23: structured person-fact extraction and semantic writes.

Offline only: every model response is a scripted decision and every database is
a temporary SQLite file. These tests pin the behaviours the Profile rewrite
exists for: first-occurrence writes, deduplication, relation decisions,
one-off/temporary filtering, identity protection, no truncation and no silent
drops.
"""

from __future__ import annotations

import asyncio

import pytest

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_facts import ExtractedPersonFact
from companion.people.profile_refinement import (
    MAX_ENTRY_CHARS,
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.providers.deepseek_profile import ScriptedProfileRefinementProvider

IDENTITY_NAME = "identity.name"
IDENTITY_PREFERRED = "identity.preferred_name"


# -- helpers ---------------------------------------------------------------


async def _person(rt, name="Fact", **consent):
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


def _engine(rt, provider):
    return ProfileRefinementEngine(
        rt.profile_service, rt.repos.profile, provider, _source(rt)
    )


def _fact(
    fact_id: str,
    value: str,
    *,
    attribute: str,
    events,
    quotes=(),
    assertion: str = "DIRECT",
    scope: str = "STABLE",
    classification=None,
    entity: str = "self",
) -> ExtractedPersonFact:
    return ExtractedPersonFact(
        fact_id=fact_id,
        entity=entity,
        attribute=attribute,
        value=value,
        assertion_type=assertion,
        temporal_scope=scope,
        classification=classification,
        evidence_event_ids=list(events),
        evidence_quotes=list(quotes),
    )


def _change(
    field_key: str,
    *,
    fact: ExtractedPersonFact,
    relation: str,
    targets=(),
    content=None,
    operation="ADD",
) -> ProfileChangeProposal:
    return ProfileChangeProposal(
        operation=operation,
        field_key=field_key,
        target_entry_ids=list(targets),
        fact_id=fact.fact_id,
        relation=relation,
        proposed_content=content,
        evidence_event_ids=list(fact.evidence_event_ids),
        evidence_quotes=list(fact.evidence_quotes),
    )


def _plain_add(field_key, content, event_id, quote=None, confidence=0.9):
    return ProfileChangeProposal(
        operation="ADD",
        field_key=field_key,
        proposed_content=content,
        evidence_event_ids=[event_id],
        evidence_quotes=[quote] if quote else [],
        confidence=confidence,
    )


def _decision(*changes, facts=(), reason="test", decision=None):
    return ProfileReviewDecision(
        person_id="placeholder",
        base_revision=0,
        decision=decision or ("PROPOSE_CHANGES" if changes else "NO_CHANGE"),
        reason=reason,
        facts=list(facts),
        changes=list(changes),
    )


async def _review(rt, provider, person_id, event_id):
    return await _engine(rt, provider).review(person_id, event_id)


async def _entries(rt, person_id, field_key=None):
    return await rt.repos.profile.list_entries(
        person_id,
        statuses=("ACTIVE", "TENTATIVE"),
        field_key=field_key,
    )


async def _insert(rt, person, field_key, content, events=()):
    return await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key=field_key,
        content=content,
        status="ACTIVE",
        source="INTERACTION_INFERENCE",
        confidence=0.9,
        evidence_event_ids=list(events),
        evidence_memory_ids=[],
    )


async def _run(rt, person, event, provider):
    review = await _review(rt, provider, person.person_id, event.event_id)
    applied = await _engine(rt, provider).apply_proposals(review)
    return review, applied


# -- 1. first occurrence ----------------------------------------------------


async def test_explicit_goal_is_written_on_first_occurrence(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我今年的目标是学会弹吉他。")
    fact = _fact(
        "f1",
        "今年的目标是学会弹吉他",
        attribute="goal",
        events=[event.event_id],
        quotes=["我今年的目标是学会弹吉他"],
        scope="FUTURE",
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change("goals.short_term", fact=fact, relation="DISTINCT"),
            facts=[fact],
        )]
    )

    review, applied = await _run(rt, person, event, provider)

    assert review.status == "ACCEPTED"
    assert applied.status == "APPLIED"
    entries = await _entries(rt, person.person_id)
    assert [(item["field_key"], item["content"]) for item in entries] == [
        ("goals.short_term", "今年的目标是学会弹吉他")
    ]


# -- 2. deduplication -------------------------------------------------------


async def test_ten_repeats_keep_exactly_one_entry(rt):
    person = await _person(rt)
    for index in range(10):
        event = await _event(rt, person, "我喜欢爵士乐。", f"c{index}")
        provider = ScriptedProfileRefinementProvider(
            [_decision(_plain_add(
                "interests.likes", "喜欢爵士乐", event.event_id,
                quote="我喜欢爵士乐",
            ))]
        )
        review, applied = await _run(rt, person, event, provider)
        if index == 0:
            assert applied.status == "APPLIED"
        else:
            assert review.status in ("NO_CHANGE", "REJECTED")

    entries = await _entries(rt, person.person_id)
    assert [item["content"] for item in entries] == ["喜欢爵士乐"]


async def test_paraphrase_is_same_and_only_adds_evidence(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。", "c1")
    await _insert(rt, person, "interests.likes", "喜欢爵士乐", [first.event_id])
    before = await rt.repos.profile.get(person.person_id)
    entry_id = (await _entries(rt, person.person_id))[0]["entry_id"]

    second = await _event(rt, person, "我对爵士乐挺感兴趣的。", "c2")
    fact = _fact(
        "f1",
        "对爵士乐感兴趣",
        attribute="interest",
        events=[second.event_id],
        quotes=["我对爵士乐挺感兴趣的"],
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change(
                "interests.likes", fact=fact, relation="SAME",
                targets=[entry_id],
            ),
            facts=[fact],
        )]
    )

    review, applied = await _run(rt, person, second, provider)

    assert review.status == "ACCEPTED", review.rejected
    assert applied.status == "NO_CHANGE"
    after = await rt.repos.profile.get(person.person_id)
    assert after["revision"] == before["revision"]
    entries = await _entries(rt, person.person_id)
    assert [item["content"] for item in entries] == ["喜欢爵士乐"]
    evidence = await rt.repos.profile.list_evidence(person.person_id)
    assert any(
        item["field_key"] == "interests.likes"
        and item["source_event_id"] == second.event_id
        for item in evidence
    )


async def test_new_detail_extends_the_original_entry(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我正在做一个机器人项目。", "c1")
    entry_id = await _insert(
        rt, person, "projects.active", "正在做一个机器人项目", [first.event_id]
    )

    second = await _event(rt, person, "我们的机器人项目下周要参加比赛。", "c2")
    fact = _fact(
        "f1",
        "正在做一个机器人项目，下周要参加比赛",
        attribute="project",
        events=[second.event_id],
        quotes=["我们的机器人项目下周要参加比赛"],
        scope="ONGOING",
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change(
                "projects.active", fact=fact, relation="EXTENDS",
                targets=[entry_id],
            ),
            facts=[fact],
        )]
    )

    review, applied = await _run(rt, person, second, provider)

    assert review.status == "ACCEPTED"
    assert applied.status == "APPLIED"
    entries = await _entries(rt, person.person_id, "projects.active")
    assert len(entries) == 1
    assert entries[0]["content"] == "正在做一个机器人项目，下周要参加比赛"
    assert set(entries[0]["evidence_event_ids"]) == {
        first.event_id, second.event_id,
    }


async def test_interest_and_skill_are_not_merged(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢摄影。", "c1")
    await _insert(rt, person, "interests.likes", "喜欢摄影", [first.event_id])

    second = await _event(rt, person, "我会摄影，也拍过一些作品。", "c2")
    fact = _fact(
        "f1",
        "摄影技能",
        attribute="skill",
        events=[second.event_id],
        quotes=["我会摄影"],
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change("knowledge.skills", fact=fact, relation="DISTINCT"),
            facts=[fact],
        )]
    )

    review, applied = await _run(rt, person, second, provider)

    assert review.status == "ACCEPTED"
    assert applied.status == "APPLIED"
    entries = await _entries(rt, person.person_id)
    assert {(item["field_key"], item["content"]) for item in entries} == {
        ("interests.likes", "喜欢摄影"),
        ("knowledge.skills", "摄影技能"),
    }


# -- 3. state changes -------------------------------------------------------


async def test_project_state_change_moves_active_to_completed(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我正在做机器人项目。", "c1")
    entry_id = await _insert(
        rt, person, "projects.active", "正在做机器人项目", [first.event_id]
    )

    second = await _event(rt, person, "机器人项目已经完成了。", "c2")
    fact = _fact(
        "f1",
        "机器人项目已经完成",
        attribute="project",
        events=[second.event_id],
        quotes=["机器人项目已经完成了"],
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change(
                "projects.completed", fact=fact, relation="STATE_CHANGE",
                targets=[entry_id],
            ),
            facts=[fact],
        )]
    )

    review, applied = await _run(rt, person, second, provider)

    assert review.status == "ACCEPTED"
    assert applied.status == "APPLIED"
    assert await _entries(rt, person.person_id, "projects.active") == []
    completed = await _entries(rt, person.person_id, "projects.completed")
    assert [item["content"] for item in completed] == ["机器人项目已经完成"]
    old = await rt.repos.profile.get_entry(entry_id)
    assert old["status"] == "ARCHIVED"
    assert old["superseded_by"] == completed[0]["entry_id"]


# -- 4. one-off / temporary / observed -------------------------------------


async def test_one_off_event_never_enters_the_profile(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我上周末去图书馆借了几本书。")
    # The model claims PROFILE; local classification is what decides.
    fact = _fact(
        "f1",
        "上周末去图书馆借书",
        attribute="habit",
        events=[event.event_id],
        quotes=["我上周末去图书馆借了几本书"],
        scope="EVENT",
        classification="PROFILE",
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change("habits.stable_habits", fact=fact, relation="DISTINCT"),
            facts=[fact],
        )]
    )

    review, _ = await _run(rt, person, event, provider)

    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "fact_classified_as_scene_memory"
    assert await _entries(rt, person.person_id) == []


async def test_temporary_state_never_becomes_a_stable_habit(rt):
    person = await _person(rt)
    event = await _event(rt, person, "这周在准备考试，最近特别累。")
    fact = _fact(
        "f1",
        "最近特别累",
        attribute="habit",
        events=[event.event_id],
        quotes=["最近特别累"],
        scope="TEMPORARY",
        classification="PROFILE",
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change("habits.stable_habits", fact=fact, relation="DISTINCT"),
            facts=[fact],
        )]
    )

    review, _ = await _run(rt, person, event, provider)

    assert review.status == "REJECTED", review.rejected
    assert review.rejected[0]["reason"] == "fact_classified_as_scene_memory"
    assert await _entries(rt, person.person_id) == []


async def test_repeated_behavior_pattern_does_not_invent_an_interest(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我给你讲个笑话吧。", "c1")
    second = await _event(rt, person, "再讲一个笑话吧。", "c1")
    fact = _fact(
        "f1",
        "会主动讲笑话",
        attribute="behavior_pattern",
        events=[first.event_id, second.event_id],
        quotes=["讲个笑话"],
        assertion="OBSERVED",
        scope="STABLE",
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change(
                "personality.behavior_patterns", fact=fact, relation="DISTINCT",
            ),
            facts=[fact],
        )]
    )

    review, applied = await _run(rt, person, second, provider)

    assert review.status == "ACCEPTED", review.rejected
    assert applied.status == "APPLIED"
    entries = await _entries(rt, person.person_id)
    assert [item["field_key"] for item in entries] == [
        "personality.behavior_patterns"
    ]
    assert [item["content"] for item in entries] == ["会主动讲笑话"]
    # The observed behaviour never became "喜欢讲笑话".
    assert "喜欢" not in entries[0]["content"]


# -- 5. identity protection -------------------------------------------------


async def test_non_name_value_never_overwrites_identity(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我是李约。", "c1")
    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key=IDENTITY_NAME,
        value="李约",
        source_event_id=first.event_id,
    )
    assert result.status == "APPLIED"

    second = await _event(rt, person, "你叫我顺其自然就好。", "c2")
    rejected = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key=IDENTITY_NAME,
        value="顺其自然",
        source_event_id=second.event_id,
    )

    assert rejected.status == "REJECTED"
    assert rejected.reason == "not_an_explicit_self_report"
    names = await _entries(rt, person.person_id, IDENTITY_NAME)
    assert [item["content"] for item in names] == ["李约"]


# -- 6. corrections and duplicates --------------------------------------


async def test_explicit_correction_updates_the_target(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。", "c1")
    entry_id = await _insert(
        rt, person, "interests.likes", "喜欢爵士乐", [first.event_id]
    )

    second = await _event(
        rt, person, "其实我不喜欢爵士乐了，我现在喜欢古典乐。", "c2"
    )
    fact = _fact(
        "f1",
        "不喜欢爵士乐，喜欢古典乐",
        attribute="interest",
        events=[second.event_id],
        quotes=["其实我不喜欢爵士乐了"],
        assertion="CORRECTION",
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change(
                "interests.likes", fact=fact, relation="CORRECTS",
                targets=[entry_id],
            ),
            facts=[fact],
        )]
    )

    review, applied = await _run(rt, person, second, provider)

    assert review.status == "ACCEPTED"
    assert applied.status == "APPLIED"
    entries = await _entries(rt, person.person_id, "interests.likes")
    assert [item["content"] for item in entries] == ["不喜欢爵士乐，喜欢古典乐"]
    assert len(entries) == 1


async def test_cross_field_duplicate_is_not_added_again(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。", "c1")
    entry_id = await _insert(
        rt, person, "interests.topics", "喜欢爵士乐", [first.event_id]
    )
    before = await rt.repos.profile.get(person.person_id)

    second = await _event(rt, person, "我挺喜欢爵士乐的。", "c2")
    fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="interest",
        events=[second.event_id],
        quotes=["我挺喜欢爵士乐的"],
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change(
                "interests.likes", fact=fact, relation="SAME",
                targets=[entry_id],
            ),
            facts=[fact],
        )]
    )

    review, applied = await _run(rt, person, second, provider)

    assert review.status == "ACCEPTED"
    assert applied.status == "NO_CHANGE"
    assert await _entries(rt, person.person_id, "interests.likes") == []
    assert (await rt.repos.profile.get(person.person_id))["revision"] == before[
        "revision"
    ]


# -- 7. evidence and target validation ---------------------------------


async def test_invalid_target_entry_id_is_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我在做一个机器人项目。")
    fact = _fact(
        "f1",
        "正在做一个机器人项目",
        attribute="project",
        events=[event.event_id],
        quotes=["我在做一个机器人项目"],
        scope="ONGOING",
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change(
                "projects.active", fact=fact, relation="EXTENDS",
                targets=["pfe_does_not_exist"],
            ),
            facts=[fact],
        )]
    )

    review, _ = await _run(rt, person, event, provider)

    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "target_entry_not_in_this_profile"
    assert await _entries(rt, person.person_id) == []


async def test_fabricated_evidence_is_rejected(rt):
    person = await _person(rt)
    event = await _event(rt, person, "随便聊聊。")
    fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="interest",
        events=["evt_fabricated"],
        quotes=["我喜欢爵士乐"],
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change("interests.likes", fact=fact, relation="DISTINCT"),
            facts=[fact],
        )]
    )

    review, _ = await _run(rt, person, event, provider)

    assert review.status == "REJECTED"
    assert review.rejected[0]["reason"] == "fact_evidence_event_not_authorized"
    assert await _entries(rt, person.person_id) == []


async def test_concurrent_reviews_write_only_once(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。", "c1")
    second = await _event(rt, person, "我喜欢摄影。", "c2")

    decision_a = _decision(_plain_add(
        "interests.likes", "喜欢爵士乐", first.event_id, quote="我喜欢爵士乐",
    ))
    decision_b = _decision(_plain_add(
        "interests.likes", "喜欢摄影", second.event_id, quote="我喜欢摄影",
    ))
    provider_a = ScriptedProfileRefinementProvider([decision_a])
    provider_b = ScriptedProfileRefinementProvider([decision_b])
    review_a = await _review(rt, provider_a, person.person_id, first.event_id)
    review_b = await _review(rt, provider_b, person.person_id, second.event_id)

    a, b = await asyncio.gather(
        _engine(rt, provider_a).apply_proposals(review_a),
        _engine(rt, provider_b).apply_proposals(review_b),
    )

    assert {a.status, b.status} == {"APPLIED", "REVISION_CONFLICT"}
    assert len(await _entries(rt, person.person_id)) == 1


# -- 8. length and batch limits ----------------------------------------


async def test_content_is_never_silently_truncated(rt):
    person = await _person(rt)
    service = rt.profile_service

    exact = "长" * MAX_ENTRY_CHARS
    assert service._normalise_entry_content(exact) == exact
    assert len(service._normalise_entry_content(exact)) == MAX_ENTRY_CHARS
    with pytest.raises(ValueError) as excinfo:
        service._normalise_entry_content("长" * (MAX_ENTRY_CHARS + 1))
    assert "content_exceeds_entry_limit_requires_refinement" in str(
        excinfo.value
    )


async def test_changes_beyond_the_batch_limit_are_recorded_not_dropped(rt):
    labels = [
        "爵士乐", "摄影", "露营", "徒步", "钓鱼", "跑步",
        "游泳", "编程", "写作", "咖啡", "电影", "旅行",
    ]
    person = await _person(rt)
    event = await _event(
        rt, person, "我喜欢" + "、".join(labels) + "。"
    )
    facts = [
        _fact(
            f"f{index}",
            f"喜欢{label}",
            attribute="interest",
            events=[event.event_id],
            quotes=[label],
        )
        for index, label in enumerate(labels, start=1)
    ]
    changes = [
        _change("interests.likes", fact=fact, relation="DISTINCT")
        for fact in facts
    ]
    provider = ScriptedProfileRefinementProvider(
        [_decision(*changes, facts=facts)]
    )

    review, applied = await _run(rt, person, event, provider)

    assert review.status == "ACCEPTED"
    assert len(review.changes) == 10
    over = [item for item in review.rejected if item["reason"] == "change_limit_exceeded"]
    assert len(over) == 2
    assert all(item["fact_id"] for item in over)
    assert len(await _entries(rt, person.person_id)) == 10


# -- 9. consolidation and source invalidation ----------------------------


async def test_every_extracted_fact_gets_a_recorded_result(rt):
    """A fact with no relation is recorded, and NO_CHANGE must not hide it."""
    person = await _person(rt)
    event = await _event(rt, person, "昨天去看了一场电影，很喜欢。")
    goal = _fact(
        "f1",
        "想学会弹吉他",
        attribute="goal",
        events=[event.event_id],
        scope="FUTURE",
    )
    outing = _fact(
        "f2",
        "昨天去看了一场电影",
        attribute="shared_context",
        events=[event.event_id],
        scope="EVENT",
    )
    # The model extracted two facts but forgot to decide any relation for them.
    provider = ScriptedProfileRefinementProvider(
        [_decision(decision="NO_CHANGE", facts=[goal, outing])]
    )

    review = await _review(rt, provider, person.person_id, event.event_id)

    assert review.status == "NO_CHANGE"
    reasons = {item["fact_id"]: item["reason"] for item in review.rejected}
    assert reasons == {
        "f1": "fact_has_no_relation_decision",
        "f2": "fact_not_profile_relevant",
    }
    assert await _entries(rt, person.person_id) == []


async def test_capacity_is_resolved_by_consolidation_before_rejecting(rt):
    """A full field merges its lossless duplicates instead of dropping a fact."""
    person = await _person(rt)
    seed = await _event(rt, person, "说话正式一点。")
    contents = [
        "保持正式",
        "保持正式且简洁",
        "语气温和",
        "语气直接",
        "少用表情",
        "多用例子",
        "适当幽默",
        "先讲结论",
    ]
    for content in contents:
        await _insert(
            rt, person, "communication.preferred_tone", content, [seed.event_id]
        )
    event = await _event(rt, person, "偶尔可以幽默一点。")
    fact = _fact(
        "f1",
        "偶尔幽默",
        attribute="preferred_tone",
        events=[event.event_id],
        quotes=["偶尔可以幽默一点"],
        assertion="DIRECT",
    )
    provider = ScriptedProfileRefinementProvider(
        [
            _decision(
                _change(
                    "communication.preferred_tone",
                    fact=fact,
                    relation="DISTINCT",
                    content="偶尔幽默",
                ),
                facts=[fact],
            )
        ]
    )

    review, applied = await _run(rt, person, event, provider)

    assert review.status == "ACCEPTED"
    assert applied.applied, applied.rejected
    entries = await _entries(
        rt, person.person_id, "communication.preferred_tone"
    )
    stored = {item["content"] for item in entries}
    assert len(entries) == 8
    assert "偶尔幽默" in stored
    assert "保持正式且简洁" in stored
    assert "保持正式" not in stored


async def test_consolidation_does_not_merge_different_attributes(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我喜欢爵士乐。", "c1")
    await _insert(rt, person, "interests.likes", "喜欢爵士乐", [first.event_id])
    await _insert(rt, person, "interests.likes", "喜欢摄影", [first.event_id])
    await _insert(
        rt,
        person,
        "interests.likes",
        "喜欢爵士乐和蓝调",
        [first.event_id],
    )

    result = await rt.profile_service.consolidate(
        person.person_id, field_keys=["interests.likes"]
    )

    assert result.status == "APPLIED"
    assert result.entries_merged == 1
    contents = {
        item["content"] for item in await _entries(rt, person.person_id)
    }
    assert contents == {"喜欢爵士乐和蓝调", "喜欢摄影"}


async def test_deleted_source_invalidates_entry_and_evidence(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    provider = ScriptedProfileRefinementProvider(
        [_decision(_plain_add(
            "interests.likes", "喜欢爵士乐", event.event_id, quote="我喜欢爵士乐",
        ))]
    )
    _, applied = await _run(rt, person, event, provider)
    assert applied.status == "APPLIED"
    entry_id = (await _entries(rt, person.person_id))[0]["entry_id"]

    invalidated = await rt.profile_service.invalidate_profile_source(
        person.person_id, event_id=event.event_id
    )

    assert invalidated.entries_archived == 1
    assert await _entries(rt, person.person_id) == []
    entry = await rt.repos.profile.get_entry(entry_id)
    assert entry["status"] == "ARCHIVED"
    assert entry["evidence_event_ids"] == []
    cur = await rt.db.execute(
        "SELECT COUNT(*) FROM person_profile_fact_index WHERE entry_id = ?",
        (entry_id,),
    )
    assert (await cur.fetchone())[0] == 0


async def test_fact_index_records_semantic_identity(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")
    fact = _fact(
        "f1",
        "喜欢爵士乐",
        attribute="interest",
        events=[event.event_id],
        quotes=["我喜欢爵士乐"],
        scope="ONGOING",
    )
    provider = ScriptedProfileRefinementProvider(
        [_decision(
            _change("interests.likes", fact=fact, relation="DISTINCT"),
            facts=[fact],
        )]
    )
    _, applied = await _run(rt, person, event, provider)
    assert applied.status == "APPLIED"

    rows = await rt.repos.profile.list_fact_index(person.person_id)
    assert len(rows) == 1
    assert rows[0]["entity_key"] == "self"
    assert rows[0]["attribute"] == "interest"
    assert rows[0]["temporal_scope"] == "ONGOING"
    assert rows[0]["canonical_value"] == "喜欢爵士乐"
