"""V0.9.9.6: explicit identity self-reports write directly; no queue.

The old confirmation flow asked a question, stored a proposal as a pending
candidate and required a later "yes". That flow is gone. An explicit
first-person statement is itself the authorization: the local detector reads
the persisted utterance, and the service writes it in one transaction. Nothing
is asked and no candidate row is created.
"""

from __future__ import annotations

import asyncio

import pytest

from companion.domain.enums import EventSource
from companion.domain.models import make_utterance_event
from companion.people.profile_confirmation import ProfileDirectSelfReportService


async def _person(rt, name="Owner", *, learning=True):
    person = await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=learning,
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


async def _name_entries(rt, person, field_key="identity.name"):
    return await rt.repos.profile.list_entries(
        person.person_id, statuses=("ACTIVE", "TENTATIVE"), field_key=field_key
    )


async def _assert_no_candidates(rt, person):
    assert await rt.repos.profile.list_legacy_candidates(person.person_id) == []


async def test_explicit_name_statement_writes_directly_without_candidate(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我是李约。")

    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="李约",
        source_event_id=event.event_id,
    )

    assert result.status == "APPLIED"
    assert result.revision == 2
    assert [item["content"] for item in await _name_entries(rt, person)] == ["李约"]
    revisions = await rt.repos.profile.list_revisions(person.person_id)
    assert len(revisions) == 1
    assert revisions[0]["changes"]["kind"] == "direct_self_report"
    assert revisions[0]["source_event_id"] == event.event_id
    await _assert_no_candidates(rt, person)


async def test_english_preferred_name_writes_directly(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我的英文名是 Lucas。")

    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.preferred_name",
        value="Lucas",
        source_event_id=event.event_id,
    )

    assert result.status == "APPLIED"
    entries = await _name_entries(rt, person, "identity.preferred_name")
    assert [item["content"] for item in entries] == ["Lucas"]


async def test_bare_introduction_is_written_by_the_turn_hook(rt):
    person = await _person(rt)
    event = await _event(rt, person, "你好，我叫阿明。")

    service = ProfileDirectSelfReportService(
        profile_service=rt.profile_service, repos=rt.repos
    )
    answer = await service.begin_turn(event=event, person_id=person.person_id)

    assert answer is None  # no confirmation question is ever returned
    assert [item["content"] for item in await _name_entries(rt, person)] == ["阿明"]
    await _assert_no_candidates(rt, person)


async def test_duplicate_self_report_does_not_bump_revision(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我叫阿明。")
    second = await _event(rt, person, "我叫阿明。", conversation="c2")

    one = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="阿明",
        source_event_id=first.event_id,
    )
    two = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="阿明",
        source_event_id=second.event_id,
    )

    assert one.status == "APPLIED"
    assert two.status == "NO_CHANGE"
    assert two.revision == 2
    profile = await rt.profile_service.get(person.person_id)
    assert profile.metadata.revision == 2
    assert len(await rt.repos.profile.list_revisions(person.person_id)) == 1
    assert len(await _name_entries(rt, person)) == 1


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("如果我是李约呢？", "李约"),
        ("假如我叫阿明。", "阿明"),
        ("他说他叫李约。", "李约"),
        ("我朋友叫阿明。", "阿明"),
        ("我叫阿明吗？", "阿明"),
        ('他跟我说“我叫阿明”。', "阿明"),
    ],
)
async def test_hypothetical_question_and_third_party_are_rejected(rt, text, value):
    person = await _person(rt)
    event = await _event(rt, person, text)

    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value=value,
        source_event_id=event.event_id,
    )

    assert result.status == "REJECTED"
    assert result.reason == "not_an_explicit_self_report"
    assert await _name_entries(rt, person) == []
    assert (await rt.profile_service.get(person.person_id)).metadata.revision == 1
    await _assert_no_candidates(rt, person)


async def test_caller_cannot_invent_a_name_the_person_did_not_say(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我是李约。")

    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="王五",
        source_event_id=event.event_id,
    )

    assert result.status == "REJECTED"
    assert result.reason == "not_an_explicit_self_report"
    assert await _name_entries(rt, person) == []


async def test_event_from_another_speaker_is_not_evidence(rt):
    owner = await _person(rt, "Owner")
    other = await _person(rt, "Other")
    event = await _event(rt, other, "我叫阿明。")

    result = await rt.profile_service.apply_direct_self_report(
        person_id=owner.person_id,
        field_key="identity.name",
        value="阿明",
        source_event_id=event.event_id,
    )

    assert result.status == "REJECTED"
    assert result.reason == "source_event_not_owned_by_person"
    assert await _name_entries(rt, owner) == []


async def test_missing_or_deleted_event_is_not_evidence(rt):
    person = await _person(rt)
    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="阿明",
        source_event_id="evt_missing",
    )
    assert result.status == "REJECTED"
    assert result.reason == "source_event_not_found"

    event = await _event(rt, person, "我叫阿明。")
    await rt.repos.profile.db.execute(
        "DELETE FROM utterances WHERE event_id = ?", (event.event_id,)
    )
    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="阿明",
        source_event_id=event.event_id,
    )
    assert result.status == "REJECTED"
    assert result.reason == "source_event_not_found"


async def test_learning_consent_blocks_direct_write(rt):
    person = await _person(rt, "NoLearning", learning=False)
    event = await _event(rt, person, "我叫阿明。")

    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="identity.name",
        value="阿明",
        source_event_id=event.event_id,
    )

    assert result.status == "UNAUTHORIZED"
    assert await _name_entries(rt, person) == []


async def test_non_identity_field_cannot_use_the_direct_path(rt):
    person = await _person(rt)
    event = await _event(rt, person, "我喜欢爵士乐。")

    result = await rt.profile_service.apply_direct_self_report(
        person_id=person.person_id,
        field_key="interests.likes",
        value="爵士乐",
        source_event_id=event.event_id,
    )

    assert result.status == "REJECTED"
    assert result.reason == "field_not_self_reportable"


async def test_concurrent_duplicate_self_reports_write_once(rt):
    person = await _person(rt)
    first = await _event(rt, person, "我叫阿明。")
    second = await _event(rt, person, "我叫阿明。", conversation="c2")

    results = await asyncio.gather(
        rt.profile_service.apply_direct_self_report(
            person_id=person.person_id,
            field_key="identity.name",
            value="阿明",
            source_event_id=first.event_id,
        ),
        rt.profile_service.apply_direct_self_report(
            person_id=person.person_id,
            field_key="identity.name",
            value="阿明",
            source_event_id=second.event_id,
        ),
    )

    assert {item.status for item in results} <= {"APPLIED", "REVISION_CONFLICT", "NO_CHANGE"}
    assert len(await _name_entries(rt, person)) == 1
    assert len(await rt.repos.profile.list_revisions(person.person_id)) == 1
    await _assert_no_candidates(rt, person)


async def test_ambiguous_name_list_is_not_a_self_report(rt):
    """Requirement: "李约，lucas，记住了吗" does not silently set a name."""
    person = await _person(rt)
    event = await _event(rt, person, "李约，lucas，记住了吗")

    service = ProfileDirectSelfReportService(
        profile_service=rt.profile_service, repos=rt.repos
    )
    assert await service.begin_turn(event=event, person_id=person.person_id) is None

    assert await _name_entries(rt, person) == []
    assert await _name_entries(rt, person, "identity.preferred_name") == []
    await _assert_no_candidates(rt, person)
