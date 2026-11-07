"""Regressions for name self-reports and truthful memory status.

No external model or network is required.
"""

from companion.domain.enums import (
    AssertionType,
    CertaintyCategory,
    EventSource,
    MemoryKind,
    MemoryOperation,
    MemoryStatus,
)
from companion.domain.models import MemoryCandidate, make_utterance_event, new_id
from companion.memory.consolidation import MemoryWriter
from companion.memory.keys import claim_is_grounded, extract_claim_key
from companion.people.profile_confirmation import detect_self_reports
from companion.prompts.response import _outcome_line


def test_name_paraphrase_is_grounded_in_exact_first_person_introduction():
    content = "用户自称名字为“李约”"
    claim = extract_claim_key(content)
    assert claim is not None
    assert claim.entity_key == "self"
    assert claim.attribute_key == "identity:name"
    assert claim_is_grounded(content, "我是李约", evidence_span="我是李约")
    assert not claim_is_grounded(content, "我不是李约")
    assert not claim_is_grounded(content, "他说我是李约")
    assert not claim_is_grounded(content, "我是李明")
    assert not claim_is_grounded(content, "我叫李明")


def test_name_intro_is_detected_but_role_and_hypothetical_are_not():
    found = detect_self_reports("我是李约")
    assert [(x.field_key, x.value) for x in found] == [("identity.name", "李约")]
    for text in ("我是学生", "我是程序员", "我是高三生", "如果我是李约", "他说我是李约"):
        assert detect_self_reports(text) == (), text


def test_memory_result_cannot_call_candidate_a_saved_active_memory():
    assert "待确认候选" in _outcome_line(
        {"outcome": "SAVED", "detail": "status=CANDIDATE"}
    )
    assert "可检索" in _outcome_line(
        {"outcome": "SAVED", "detail": "status=ACTIVE"}
    )
    assert "待确认候选" not in _outcome_line(
        {"outcome": "SAVED", "detail": "status=ACTIVE"}
    )


async def test_paraphrased_name_becomes_retrievable_and_still_needs_profile_confirmation(rt):
    person = await rt.registry.register(
        "Owner",
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("identity-introduction")
    event = make_utterance_event(
        conversation_id="identity-introduction",
        text="我是李约",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.repos.utterance.save(event)
    proposal = MemoryCandidate(
        candidate_id=new_id("memc"),
        source_event_id=event.event_id,
        subject_person_id=person.person_id,
        content="用户自称名字为“李约”",
        operation=MemoryOperation.ADD,
        assertion_type=AssertionType.SELF_REPORT,
        certainty_category=CertaintyCategory.DIRECT,
        topic="name",
        kind=MemoryKind.PERSONAL_FACT,
        source_evidence_span="我是李约",
        explicit_remember_request=True,
    )
    await rt.repos.analysis.save_candidate(proposal)
    outcomes = await MemoryWriter(rt.repos, rt.memory_service).consolidate_event(
        event.event_id, only_explicit=True
    )
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == "SAVED"
    saved = await rt.memory_service.repo.get(outcomes[0]["memory_id"])
    assert saved.status == MemoryStatus.ACTIVE
    recalled = await rt.memory.retrieve("我的名字是什么", person.person_id)
    assert any(item["memory_id"] == saved.memory_id for item in recalled)

    # Profile name is a separate, stricter confirmation path.
    from companion.people.profile_confirmation import ProfileConfirmationService
    service = ProfileConfirmationService(profile_service=rt.profile_service, repos=rt.repos)
    question = await service.begin_turn(event=event, person_id=person.person_id)
    assert question and "李约" in question
    assert await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE",), field_key="identity.name"
    ) == []


async def test_explicit_followup_promotes_only_verified_legacy_name_candidate(rt):
    from companion.domain.enums import MemoryStatus
    from companion.memory.explicit_operations import ExplicitMemoryOperationService
    from companion.memory.models import AccessContext
    from companion.routing.models import (
        ProcessingIntent, RouteAction, RouteAddressee, ValidatedRouteDecision,
        RoutingContext,
    )
    from companion.domain.enums import ReplyNecessity

    person = await rt.registry.register(
        "Owner", "owner", allow_processing=True,
        allow_persistence=True, allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("legacy")
    source = make_utterance_event(
        conversation_id="legacy", text="我是李约", addressee_hint="agent",
        speaker_person_id=person.person_id, source=EventSource.TEST,
    )
    await rt.repos.utterance.save(source)
    old_candidate = MemoryCandidate(
        candidate_id=new_id("memc"), source_event_id=source.event_id,
        subject_person_id=person.person_id,
        content="用户自称名字为“李约”",
        operation=MemoryOperation.ADD, assertion_type=AssertionType.SELF_REPORT,
        certainty_category=CertaintyCategory.DIRECT,
        topic="name", kind=MemoryKind.PERSONAL_FACT,
        source_evidence_span="我是李约", explicit_remember_request=True,
    )
    await rt.repos.analysis.save_candidate(old_candidate)
    access = AccessContext(actor_id=person.person_id, role="cognitive")
    old = await rt.memory_service.create(
        kind=MemoryKind.PERSONAL_FACT, content=old_candidate.content,
        access=access, status=MemoryStatus.CANDIDATE,
        subjects=[("PERSON", person.person_id)],
        self_reported=True, model_inferred=True,
    )
    await rt.memory_service.repo.add_provenance(
        old.memory_id,
        source_event_id=source.event_id,
        candidate_id=old_candidate.candidate_id,
        speaker_person_id=person.person_id,
        subject_person_id=person.person_id,
        topic="name",
        assertion_type="SELF_REPORT",
        certainty_category="DIRECT",
        self_reported=True,
        model_inferred=True,
    )
    assert await rt.memory.retrieve("我的名字", person.person_id) == []

    followup = make_utterance_event(
        conversation_id="legacy",
        text="你写入记忆库啊，为什么不写入？",
        addressee_hint="agent",
        speaker_person_id=person.person_id, source=EventSource.TEST,
    )
    await rt.repos.utterance.save(followup)
    view = RoutingContext(
        conversation_id="legacy",
        current_event_id=followup.event_id,
        current_speaker_id=person.person_id,
        current_text=followup.text,
        recent_turns=[{
            "role": "human", "speaker_person_id": person.person_id,
            "text": "我是谁",
        }],
    )
    route = ValidatedRouteDecision(
        action=RouteAction.RESPOND,
        reply_necessity=ReplyNecessity.REQUIRED,
        addressee_assessment=RouteAddressee.AGENT,
        processing_intents=[ProcessingIntent.MEMORY_SAVE],
    )

    class NeverExtract:
        async def extract_events(self, *args, **kwargs):
            raise AssertionError("verified legacy promotion requires no model")

    writer = MemoryWriter(rt.repos, rt.memory_service)
    service = ExplicitMemoryOperationService(
        repos=rt.repos, cognitive=NeverExtract(), memory_writer=writer
    )
    result = await service.execute(followup, route, None, view)
    assert result.status == "UPDATED"
    assert result.outcomes[0]["memory_id"] == old.memory_id
    upgraded = await rt.memory_service.repo.get(old.memory_id)
    assert upgraded.status == MemoryStatus.ACTIVE
    assert upgraded.version == 2
    assert any(
        item["memory_id"] == old.memory_id
        for item in await rt.memory.retrieve("我的名字", person.person_id)
    )
    repeated = await service.execute(followup, route, None, view)
    assert repeated.status == "UPDATED"
    assert (await rt.memory_service.repo.get(old.memory_id)).version == 2
