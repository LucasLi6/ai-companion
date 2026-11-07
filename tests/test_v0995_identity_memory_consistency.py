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
