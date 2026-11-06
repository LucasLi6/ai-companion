"""V0.9.7: profile refinement engine, validation and atomic application."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from companion.cognition.privacy import ForegroundPrivacyGateway
from companion.domain.enums import EventSource
from companion.domain.models import Consent, make_utterance_event, utcnow
from companion.people.profile_refinement import (
    ProfileChangeProposal,
    ProfileRefinementEngine,
    ProfileRefinementProviderError,
    ProfileRefinementValidationError,
    ProfileReviewContextSource,
    ProfileReviewDecision,
)
from companion.people.profile_service import (
    ProfileProposalStale,
    TRUSTED_CONFIRMATION,
)
from companion.providers.deepseek_profile import (
    DeepSeekProfileRefinementProvider,
    ScriptedProfileRefinementProvider,
)


async def _person(rt, name="Owner"):
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


async def _entry(rt, person, field_key, content, source="DIRECT_SELF_REPORT"):
    return await rt.repos.profile.insert_entry(
        person_id=person.person_id,
        field_key=field_key,
        content=content,
        status="ACTIVE",
        source=source,
        confidence=1.0,
        evidence_event_ids=[],
        evidence_memory_ids=[],
    )


def _engine(rt, decision):
    provider = ScriptedProfileRefinementProvider(decision)
    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        provider,
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    return engine, provider


def _decision(operation, field_key, *, content=None, targets=None, evidence=None,
              confidence=0.95, person_id="ignored", revision=0, quotes=None):
    return ProfileReviewDecision(
        person_id=person_id,
        base_revision=revision,
        decision="PROPOSE_CHANGES",
        reason="test",
        changes=[
            ProfileChangeProposal(
                operation=operation,
                field_key=field_key,
                target_entry_ids=targets or [],
                proposed_content=content,
                evidence_event_ids=evidence or [],
                evidence_quotes=quotes or [],
                confidence=confidence,
            )
        ],
    )


async def test_existing_profile_is_enough_returns_no_change(rt):
    person = await _person(rt, "NoChange")
    event = await _event(rt, person, "今天有点累。")
    engine, _ = _engine(
        rt,
        ProfileReviewDecision(
            person_id=person.person_id,
            base_revision=1,
            decision="NO_CHANGE",
            reason="small talk",
            changes=[],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    assert review.status == "NO_CHANGE"
    applied = await engine.apply_proposals(review)
    assert applied.status == "NO_CHANGE"
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 1


async def test_refine_deepens_existing_understanding_without_new_field(rt):
    person = await _person(rt, "Refiner")
    entry_id = await _entry(
        rt, person, "interests.topics", "喜欢电子工程，正在开发 Ava"
    )
    event = await _event(rt, person, "Ava 的长期记忆设计我还在改。")
    engine, _ = _engine(
        rt,
        _decision(
            "REFINE",
            "interests.topics",
            targets=[entry_id],
            content="喜欢电子工程，正在开发 Ava，重点在研究长期记忆设计",
            evidence=[event.event_id],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    assert review.has_changes()
    applied = await engine.apply_proposals(review)
    assert applied.status == "APPLIED"
    entries = await rt.repos.profile.list_entries(person.person_id)
    assert len(entries) == 1
    assert "长期记忆" in entries[0]["content"]
    assert entries[0]["entry_id"] == entry_id


async def test_different_interests_coexist_without_replacing_each_other(rt):
    person = await _person(rt, "Coexist")
    first = await _event(rt, person, "我在做机器人狗。")
    engine1, _ = _engine(
        rt,
        _decision(
            "ADD",
            "interests.topics",
            content="机器人狗",
            evidence=[first.event_id],
        ),
    )
    review1 = await engine1.review(person.person_id, first.event_id)
    assert (await engine1.apply_proposals(review1)).status == "APPLIED"

    second = await _event(rt, person, "最近在听爵士。")
    engine2, _ = _engine(
        rt,
        _decision(
            "ADD",
            "interests.topics",
            content="爵士乐",
            evidence=[second.event_id],
        ),
    )
    review2 = await engine2.review(person.person_id, second.event_id)
    assert (await engine2.apply_proposals(review2)).status == "APPLIED"

    texts = sorted(
        entry["content"]
        for entry in await rt.repos.profile.list_entries(person.person_id)
        if entry["field_key"] == "interests.topics"
    )
    assert texts == ["机器人狗", "爵士乐"]


async def test_one_bad_mood_does_not_rewrite_personality(rt):
    person = await _person(rt, "Mood")
    entry_id = await _entry(rt, person, "personality.traits", "技术决策理性")
    event = await _event(rt, person, "今天烦死了，不想写了。")
    engine, _ = _engine(
        rt,
        ProfileReviewDecision(
            person_id=person.person_id,
            base_revision=1,
            decision="NO_CHANGE",
            reason="single emotional reaction is not a stable trait",
            changes=[],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    await engine.apply_proposals(review)
    entry = await rt.repos.profile.get_entry(entry_id)
    assert entry["content"] == "技术决策理性"
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 1


async def test_correction_replaces_fact_and_keeps_history(rt):
    person = await _person(rt, "Correction")
    entry_id = await _entry(rt, person, "background.education", "大学")
    event = await _event(rt, person, "不是大学，我还在读高中。")
    engine, _ = _engine(
        rt,
        _decision(
            "CORRECT",
            "background.education",
            targets=[entry_id],
            content="高中",
            evidence=[event.event_id],
            quotes=["不是大学，我还在读高中。"],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    applied = await engine.apply_proposals(review)
    assert applied.status == "APPLIED"
    assert (await rt.repos.profile.get_entry(entry_id))["content"] == "高中"
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert revisions[-1]["changes"]["changes"][0]["before"] == ["大学"]


async def test_obsolete_project_moves_to_completed_and_keeps_old_row(rt):
    person = await _person(rt, "Project")
    entry_id = await _entry(rt, person, "projects.active", "准备期末考试")
    event = await _event(rt, person, "考试结束了，考得还行。")
    engine, _ = _engine(
        rt,
        _decision(
            "UPDATE",
            "projects.completed",
            targets=[entry_id],
            content="期末考试（已完成）",
            evidence=[event.event_id],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    applied = await engine.apply_proposals(review)
    assert applied.status == "APPLIED"
    old = await rt.repos.profile.get_entry(entry_id)
    assert old["status"] == "ARCHIVED"
    assert old["superseded_by"]
    active = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="projects.active"
    )
    completed = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="projects.completed"
    )
    assert active == []
    assert [item["content"] for item in completed] == ["期末考试（已完成）"]


async def test_confirmed_name_requires_explicit_confirmation(rt):
    person = await _person(rt, "Identity")
    entry_id = await _entry(rt, person, "identity.name", "小明")
    event = await _event(rt, person, "以后叫我阿明吧。")
    engine, _ = _engine(
        rt,
        _decision(
            "CORRECT",
            "identity.name",
            targets=[entry_id],
            content="阿明",
            evidence=[event.event_id],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    assert review.pending_changes()
    applied = await engine.apply_proposals(review)
    assert applied.status == "PENDING_CONFIRMATION"
    assert (await rt.repos.profile.get_entry(entry_id))["content"] == "小明"
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 1
    candidate = applied.pending[0]
    pending = await rt.repos.profile.find_pending(person.person_id, "identity.name")
    assert pending and pending[0]["requires_confirmation"]

    revision = await rt.profile_service.confirm_profile_proposal(
        pending[0]["candidate_id"],
        actor="user",
        authority=TRUSTED_CONFIRMATION,
    )
    assert revision == 2
    assert (await rt.repos.profile.get_entry(entry_id))["content"] == "阿明"
    assert (await rt.repos.profile.get_candidate(
        pending[0]["candidate_id"]
    ))["status"] == "CONFIRMED"


async def test_lossy_merge_is_rejected(rt):
    person = await _person(rt, "Lossy")
    first = await _entry(rt, person, "knowledge.skills", "STM32 嵌入式开发")
    second = await _entry(rt, person, "knowledge.skills", "RTOS 实时系统")
    event = await _event(rt, person, "反正就是喜欢科技。")
    engine, _ = _engine(
        rt,
        _decision(
            "MERGE",
            "knowledge.skills",
            targets=[first, second],
            content="喜欢科技",
            evidence=[event.event_id],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    assert review.status == "REJECTED"
    applied = await engine.apply_proposals(review)
    assert applied.status == "NO_CHANGE"
    entries = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",)
    )
    assert {item["content"] for item in entries} == {
        "STM32 嵌入式开发",
        "RTOS 实时系统",
    }


async def test_stale_revision_cannot_overwrite_newer_profile(rt):
    person = await _person(rt, "Race")
    first_event = await _event(rt, person, "我在做机器人。")
    second_event = await _event(rt, person, "还在做机械臂。", conversation="c2")
    # Both reviews read the same revision before either commits.
    provider = ScriptedProfileRefinementProvider(
        [
            _decision(
                "ADD",
                "projects.active",
                content="机器人项目",
                evidence=[first_event.event_id],
            ),
            _decision(
                "ADD",
                "projects.active",
                content="机械臂项目",
                evidence=[second_event.event_id],
            ),
        ]
    )
    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        provider,
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    first_review = await engine.review(person.person_id, first_event.event_id)
    second_review = await engine.review(person.person_id, second_event.event_id)
    assert first_review.base_revision == second_review.base_revision == 1
    assert (await engine.apply_proposals(first_review)).status == "APPLIED"
    second = await engine.apply_proposals(second_review)
    assert second.status == "REVISION_CONFLICT"
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 2
    entries = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",)
    )
    assert [item["content"] for item in entries] == ["机器人项目"]


async def test_invalid_evidence_and_provider_failure_leave_no_partial_change(rt):
    person = await _person(rt, "Failure")
    event = await _event(rt, person, "随便说说。")
    engine, _ = _engine(
        rt,
        _decision(
            "ADD",
            "projects.active",
            content="不存在的项目",
            evidence=["evt_invented"],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    assert review.status == "REJECTED"
    applied = await engine.apply_proposals(review)
    assert applied.status == "NO_CHANGE"
    assert await rt.repos.profile.list_entries(person.person_id) == []

    failing, _ = _engine(
        rt, ProfileReviewDecision(person_id=person.person_id, base_revision=1,
                                  decision="NO_CHANGE", changes=[])
    )
    failing.provider.error = ProfileRefinementProviderError("api down")
    with pytest.raises(ProfileRefinementProviderError):
        await failing.review(person.person_id, event.event_id)
    assert await rt.repos.profile.list_entries(person.person_id) == []


async def test_malformed_model_json_is_a_provider_failure(rt):
    person = await _person(rt, "BadJSON")
    event = await _event(rt, person, "你好。")

    class BadClient:
        async def complete(self, *args, **kwargs):
            return SimpleNamespace(content="this is not json", model="fake")

    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        DeepSeekProfileRefinementProvider(BadClient()),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    # Malformed JSON is retried once with a format-repair turn; two failures
    # are a validation error, never a silent NO_CHANGE.
    with pytest.raises(ProfileRefinementValidationError):
        await engine.review(person.person_id, event.event_id)
    assert await rt.repos.profile.list_revisions(person.person_id) == []


async def test_database_failure_rolls_back_whole_review(rt, monkeypatch):
    person = await _person(rt, "Rollback")
    entry_id = await _entry(rt, person, "interests.topics", "机器人")
    event = await _event(rt, person, "机器人还在做。")
    engine, _ = _engine(
        rt,
        _decision(
            "UPDATE",
            "interests.topics",
            targets=[entry_id],
            content="机器人（持续在做）",
            evidence=[event.event_id],
        ),
    )
    review = await engine.review(person.person_id, event.event_id)

    async def explode(*args, **kwargs):
        raise RuntimeError("db failure")

    monkeypatch.setattr(rt.repos.profile, "update_entry", explode)
    with pytest.raises(RuntimeError):
        await engine.apply_proposals(review)
    monkeypatch.undo()
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 1
    assert (await rt.repos.profile.get_entry(entry_id))["content"] == "机器人"
    assert await rt.repos.profile.list_revisions(person.person_id) == []


async def test_same_event_is_idempotent_and_permission_withdrawal_blocks_apply(rt):
    person = await _person(rt, "Idem")
    event = await _event(rt, person, "我在做机械臂。")
    engine, _ = _engine(
        rt,
        _decision(
            "ADD", "projects.active", content="机械臂", evidence=[event.event_id]
        ),
    )
    review = await engine.review(person.person_id, event.event_id)
    first = await engine.apply_proposals(review)
    assert first.status == "APPLIED"
    second = await engine.apply_proposals(review)
    assert second.status == "ALREADY_APPLIED"
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 2
    assert len(await rt.repos.profile.list_entries(person.person_id)) == 1

    consent = await rt.registry.get_consent(person.person_id)
    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=consent.allow_processing,
            allow_persistence=True,
            allow_memory_learning=True,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )
    revoked = await engine.apply_proposals(review)
    assert revoked.status == "UNAUTHORIZED"
    event2 = await _event(rt, person, "又改了一版。", conversation="c2")
    engine2, _ = _engine(
        rt,
        _decision(
            "ADD", "projects.active", content="机械臂 v2", evidence=[event2.event_id]
        ),
    )
    review2 = await engine2.review(person.person_id, event2.event_id)
    assert review2.status == "UNAUTHORIZED"


async def test_sixty_turn_conversation_stays_concise_sourced_and_complete(rt):
    """Scenario 12: ordinary turns change nothing; real changes accumulate once."""
    person = await _person(rt, "LongRun")
    texts = [
        "今天有点累。",
        "随便聊聊。",
        "我喜欢机器人。",
        "我在做一个机器人项目。",
        "今天心情不好，不想写代码。",
        "天气不错。",
        "最近在听爵士乐。",
        "机器人还在调试。",
        "今天有点烦。",
        "这个项目挺难的。",
        "我在做机器人。",
        "我喜欢机器人。",
        "先不说这个了。",
        "今天加班。",
        "音乐不错。",
        "有点困。",
        "我想想。",
        "机器人又坏了。",
        "随便。",
        "今天没什么事。",
        "爵士乐真不错。",
        "最近在听爵士。",
        "好累。",
        "明天再说。",
        "嗯。",
        "机器人进展一般。",
        "今天不高兴。",
        "先这样。",
        "改天聊。",
        "没什么。",
        "Ava 的长期记忆我还在改。",
        "这个模块有点难。",
        "调试中。",
        "今天状态不好。",
        "继续做。",
        "累了。",
        "机器人还没好。",
        "有点挫败。",
        "先休息。",
        "嗯嗯。",
        "Ava 的长期记忆做完了。",
        "终于结束了。",
        "今天还行。",
        "随便说说。",
        "我还在弄别的。",
        "我其实叫阿明。",
        "嗯。",
        "好。",
        "继续。",
        "没什么特别的。",
        "我在学 STM32。",
        "也在学嵌入式开发。",
        "这两块可以一起说。",
        "今天挺平静。",
        "嗯。",
        "好的。",
        "先这样。",
        "改天。",
        "晚安。",
        "没了。",
    ]
    assert len(texts) == 60
    events = [
        await _event(rt, person, text, conversation=f"long{index % 3}")
        for index, text in enumerate(texts)
    ]
    ids: dict[str, str] = {}

    def no_change(payload):
        return ProfileReviewDecision(
            person_id="placeholder",
            base_revision=0,
            decision="NO_CHANGE",
            reason="ordinary turn",
            changes=[],
        )

    decisions: list = [no_change for _ in range(60)]
    decisions[10] = _decision(
        "ADD", "interests.topics", content="机器人", evidence=[events[10].event_id]
    )
    # A repeated mention is not a second fact.
    decisions[11] = _decision(
        "ADD", "interests.topics", content="机器人", evidence=[events[11].event_id]
    )
    decisions[21] = _decision(
        "ADD", "interests.topics", content="爵士乐", evidence=[events[21].event_id]
    )
    decisions[30] = _decision(
        "ADD",
        "projects.active",
        content="Ava 长期记忆",
        evidence=[events[30].event_id],
    )
    decisions[40] = lambda payload: _decision(
        "UPDATE",
        "projects.completed",
        targets=[ids["project"]],
        content="Ava 长期记忆（已完成）",
        evidence=[events[40].event_id],
    )
    decisions[45] = _decision(
        "ADD", "identity.name", content="阿明", evidence=[events[45].event_id]
    )
    decisions[50] = _decision(
        "ADD",
        "knowledge.skills",
        content="STM32",
        evidence=[events[50].event_id],
    )
    decisions[51] = _decision(
        "ADD",
        "knowledge.skills",
        content="嵌入式开发",
        evidence=[events[51].event_id],
    )
    decisions[52] = lambda payload: _decision(
        "MERGE",
        "knowledge.skills",
        targets=[ids["skill1"], ids["skill2"]],
        content="STM32 嵌入式开发",
        evidence=[events[52].event_id],
    )

    engine = _engine(rt, decisions)[0]
    for index, event in enumerate(events):
        review = await engine.review(person.person_id, event.event_id)
        result = await engine.apply_proposals(review)
        if index == 10:
            ids["robot"] = result.applied[0]["entry_id"]
        elif index == 30:
            ids["project"] = result.applied[0]["entry_id"]
        elif index == 50:
            ids["skill1"] = result.applied[0]["entry_id"]
        elif index == 51:
            ids["skill2"] = result.applied[0]["entry_id"]

    profile = await rt.profile_service.get(person.person_id)
    # Seven real changes: two interests, project start, project completion,
    # two skills and their merge. Ordinary or repeated turns add nothing.
    assert profile.metadata.revision == 8
    active = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",)
    )
    assert sorted(item["content"] for item in active) == sorted(
        ["机器人", "爵士乐", "Ava 长期记忆（已完成）", "STM32 嵌入式开发"]
    )
    assert all(item["evidence_event_ids"] for item in active)
    assert not any(item["field_key"].startswith("personality.") for item in active)
    assert not any("喜欢科技" in item["content"] for item in active)

    project = await rt.repos.profile.get_entry(ids["project"])
    assert project["status"] == "ARCHIVED"
    assert project["superseded_by"]
    assert (await rt.repos.profile.get_entry(ids["skill1"]))["status"] == "MERGED"
    assert (await rt.repos.profile.get_entry(ids["skill2"]))["status"] == "MERGED"

    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert len(revisions) == 7
    assert revisions[-1]["changes"]["changes"][0]["operation"] == "MERGE"
    assert any(
        change["operation"] == "UPDATE" and change["before"] == ["Ava 长期记忆"]
        for revision in revisions
        for change in revision["changes"]["changes"]
    )

    pending = await rt.repos.profile.find_pending(person.person_id, "identity.name")
    assert len(pending) == 1
    assert pending[0]["requires_confirmation"]
    assert not any(item["field_key"] == "identity.name" for item in active)

    # The identity proposal was made at turn 45 on the then-current revision;
    # four later changes have moved the profile. It must not overwrite that
    # newer state: confirmation is explicitly stale, nothing is bumped, and the
    # candidate is kept for re-review.
    with pytest.raises(ProfileProposalStale):
        await rt.profile_service.confirm_profile_proposal(
            pending[0]["candidate_id"],
            actor="user",
            authority=TRUSTED_CONFIRMATION,
        )
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 8
    stale = await rt.repos.profile.get_candidate(pending[0]["candidate_id"])
    assert stale["status"] == "NEEDS_REVIEW"
    assert not any(
        item["field_key"] == "identity.name" for item in active
    )

    # Re-reviewing the same turn against the *current* profile produces a
    # fresh proposal bound to revision 8; only that may be confirmed.
    re_review_engine, _ = _engine(rt, decisions[45])
    review = await re_review_engine.review(person.person_id, events[45].event_id)
    reapplied = await re_review_engine.apply_proposals(review)
    assert reapplied.status == "PENDING_CONFIRMATION"
    fresh = [
        item
        for item in await rt.repos.profile.find_pending(
            person.person_id, "identity.name"
        )
        if item["status"] == "PENDING" and item["expected_revision"] == 8
    ]
    assert len(fresh) == 1

    # The strict identity change only happens through the explicit path.
    assert (
        await rt.profile_service.confirm_profile_proposal(
            fresh[0]["candidate_id"],
            actor="user",
            authority=TRUSTED_CONFIRMATION,
        )
        == 9
    )
    final = await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="identity.name"
    )
    assert [item["content"] for item in final] == ["阿明"]
