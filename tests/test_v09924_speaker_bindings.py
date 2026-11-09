"""V0.9.9.24 P3.1: explicit speaker bindings and safe identity resolution.

A similarity score can never create or resolve a binding. These offline tests
pin the four independent gates: a decided match, a human-created binding, an
ACTIVE person, and consent that allows processing and persistence.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import EventSource, RegistrationStatus
from companion.domain.models import Consent, utcnow
from companion.people.speaker_bindings import (
    SpeakerBindingError,
    SpeakerBindingService,
)
from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    ERROR,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    PENDING,
    QUALITY_HIGH,
    SpeakerResult,
)


async def _person(rt, name="Bound", **consent):
    values = dict(
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=False,
    )
    values.update(consent)
    person = await rt.registry.register(name, "owner", **values)
    return person


def _result(speaker_id="S-0001", *, status=KNOWN, provisional=False):
    return SpeakerResult(
        utterance_id=1,
        speaker_id=speaker_id,
        person_id=None,
        status=status,
        reid_quality=QUALITY_HIGH,
        similarity=0.62,
        margin=0.12,
        voiced_seconds=2.4,
        provisional=provisional,
        model_id="fake",
        inference_ms=1.0,
        completed_at=0.0,
    )


def _service(rt) -> SpeakerBindingService:
    return rt.speaker_bindings


async def test_binding_table_exists_with_schema_version_32(rt):
    cur = await rt.db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
        " AND name = 'person_speaker_bindings'"
    )
    assert await cur.fetchone() is not None
    cur = await rt.db.execute("SELECT MAX(version) FROM schema_version")
    assert (await cur.fetchone())[0] == 32


async def test_bind_and_resolve_a_known_speaker(rt):
    person = await _person(rt)
    binding = await _service(rt).bind_speaker(
        person.person_id, "S-0001", "explicit_confirmation"
    )

    assert binding["person_id"] == person.person_id
    assert binding["verified_by"] == "explicit_confirmation"
    assert await _service(rt).resolve_speaker(_result()) == person.person_id


async def test_candidate_match_can_resolve_when_bound(rt):
    person = await _person(rt)
    await _service(rt).bind_speaker(person.person_id, "TEMP-0001", "owner")

    resolved = await _service(rt).resolve_speaker(
        _result("TEMP-0001", status=CANDIDATE_MATCH)
    )

    assert resolved == person.person_id


@pytest.mark.parametrize(
    "status",
    [PENDING, NEW_CANDIDATE, AMBIGUOUS, INSUFFICIENT_AUDIO, ERROR],
)
async def test_unreliable_status_never_resolves_even_when_bound(rt, status):
    person = await _person(rt)
    await _service(rt).bind_speaker(person.person_id, "S-0001", "owner")

    assert await _service(rt).resolve_speaker(_result(status=status)) is None


async def test_provisional_sample_never_resolves(rt):
    person = await _person(rt)
    await _service(rt).bind_speaker(person.person_id, "S-0001", "owner")

    assert (
        await _service(rt).resolve_speaker(_result(provisional=True)) is None
    )


async def test_unbound_speaker_never_resolves(rt):
    await _person(rt)
    assert await _service(rt).resolve_speaker(_result("S-9999")) is None


async def test_revoked_or_suspended_person_never_resolves(rt):
    person = await _person(rt)
    await _service(rt).bind_speaker(person.person_id, "S-0001", "owner")

    await rt.registry.set_status(person.person_id, RegistrationStatus.SUSPENDED)
    assert await _service(rt).resolve_speaker(_result()) is None

    await rt.registry.set_status(person.person_id, RegistrationStatus.REVOKED)
    assert await _service(rt).resolve_speaker(_result()) is None


async def test_withdrawn_processing_consent_never_resolves(rt):
    person = await _person(rt)
    await _service(rt).bind_speaker(person.person_id, "S-0001", "owner")
    assert await _service(rt).resolve_speaker(_result()) == person.person_id

    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=False,
            allow_persistence=True,
            allow_memory_learning=True,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )
    assert await _service(rt).resolve_speaker(_result()) is None


async def test_withdrawn_persistence_consent_never_resolves(rt):
    person = await _person(rt)
    await _service(rt).bind_speaker(person.person_id, "S-0001", "owner")

    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=False,
            allow_memory_learning=True,
            allow_external_ai=False,
            updated_at=utcnow(),
        )
    )
    assert await _service(rt).resolve_speaker(_result()) is None


async def test_rebinding_same_person_is_idempotent_and_moving_is_refused(rt):
    first = await _person(rt, "First")
    second = await _person(rt, "Second")

    await _service(rt).bind_speaker(first.person_id, "S-0001", "owner")
    again = await _service(rt).bind_speaker(first.person_id, "S-0001", "owner")
    assert again["person_id"] == first.person_id
    assert len(await _service(rt).list_bindings()) == 1

    with pytest.raises(SpeakerBindingError) as error:
        await _service(rt).bind_speaker(second.person_id, "S-0001", "owner")
    assert error.value.code == "speaker_already_bound"


async def test_unbind_stops_resolution(rt):
    person = await _person(rt)
    await _service(rt).bind_speaker(person.person_id, "S-0001", "owner")
    assert await _service(rt).resolve_speaker(_result()) == person.person_id

    assert await _service(rt).unbind_speaker("S-0001") is True
    assert await _service(rt).get_binding("S-0001") is None
    assert await _service(rt).resolve_speaker(_result()) is None
    assert await _service(rt).unbind_speaker("S-0001") is False


async def test_binding_requires_a_verifier_and_an_active_person(rt):
    person = await _person(rt)
    with pytest.raises(SpeakerBindingError) as error:
        await _service(rt).bind_speaker(person.person_id, "S-0001", "")
    assert error.value.code == "binding_requires_person_speaker_and_verifier"

    await rt.registry.set_status(person.person_id, RegistrationStatus.SUSPENDED)
    with pytest.raises(SpeakerBindingError) as error:
        await _service(rt).bind_speaker(person.person_id, "S-0002", "owner")
    assert error.value.code == "person_not_active"


async def test_deleting_a_person_removes_its_bindings(rt):
    person = await _person(rt)
    await _service(rt).bind_speaker(person.person_id, "S-0001", "owner")
    await rt.db.execute("DELETE FROM persons WHERE person_id = ?", (person.person_id,))

    assert await _service(rt).list_bindings(person.person_id) == []


# -- P3.2: CLI voice turn resolution ----------------------------------------


def test_binding_bridge_is_off_by_default():
    from companion.cli import voice_person_binding_enabled

    assert voice_person_binding_enabled({}) is False
    assert voice_person_binding_enabled({"AVA_VOICE_PERSON_BINDING": "0"}) is False
    assert voice_person_binding_enabled({"AVA_VOICE_PERSON_BINDING": "true"}) is True


async def test_voice_turn_resolves_the_bound_person_not_the_startup_person(rt):
    from companion.cli import resolve_voice_person_id

    startup = await _person(rt, "Startup")
    speaker = await _person(rt, "Speaker")
    await _service(rt).bind_speaker(speaker.person_id, "S-0001", "owner")

    resolved = await resolve_voice_person_id(
        rt,
        _result(),
        default_person_id=startup.person_id,
        enabled=True,
    )

    assert resolved == speaker.person_id
    assert resolved != startup.person_id


async def test_unbound_voice_turn_is_anonymous_never_startup(rt):
    from companion.cli import resolve_voice_person_id

    startup = await _person(rt, "Startup")
    for status in (KNOWN, NEW_CANDIDATE, AMBIGUOUS, PENDING):
        resolved = await resolve_voice_person_id(
            rt,
            _result("S-9999", status=status),
            default_person_id=startup.person_id,
            enabled=True,
        )
        assert resolved is None


async def test_voice_turn_without_a_speaker_snapshot_is_anonymous(rt):
    from companion.cli import resolve_voice_person_id

    startup = await _person(rt, "Startup")
    assert (
        await resolve_voice_person_id(
            rt, None, default_person_id=startup.person_id, enabled=True
        )
        is None
    )


async def test_disabled_bridge_keeps_the_startup_person(rt):
    from companion.cli import resolve_voice_person_id

    startup = await _person(rt, "Startup")
    assert (
        await resolve_voice_person_id(
            rt, _result("S-9999"), default_person_id=startup.person_id,
            enabled=False,
        )
        == startup.person_id
    )


async def test_multi_person_turn_taking_never_leaks_attribution(rt):
    """A voice identity belongs to one turn only.

    Two bound speakers alternate, an unbindable voice interleaves, and each
    accepted turn carries exactly its own resolved person — never the previous
    turn's, never the startup person.
    """
    from companion.cli import handle_user_message, resolve_voice_person_id

    startup = await _person(rt, "Startup")
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")
    await _service(rt).bind_speaker(alice.person_id, "S-A", "owner")
    await _service(rt).bind_speaker(bob.person_id, "S-B", "owner")
    await rt.conversations.get_or_create("voice-c1")

    seen: list[Optional[str]] = []

    async def runner(rt_, event, **kwargs):
        seen.append(event.speaker_person_id)
        return {"delivered": True, "status": "WAIT", "action": "WAIT"}

    turns = [
        ("S-A", alice.person_id),
        ("S-B", bob.person_id),
        ("S-UNKNOWN", None),
        ("S-B", bob.person_id),
        ("S-A", alice.person_id),
        ("S-UNKNOWN", None),
        ("S-A", alice.person_id),
    ]
    for speaker_id, expected in turns:
        resolved = await resolve_voice_person_id(
            rt,
            _result(speaker_id),
            default_person_id=startup.person_id,
            enabled=True,
        )
        assert resolved == expected
        await handle_user_message(
            rt,
            text=f"turn from {speaker_id}",
            source=EventSource.VOICE,
            submitted_at=0.0,
            conversation_id="voice-c1",
            person_id=resolved,
            addressee="agent",
            turn_runner=runner,
        )

    assert seen == [person_id for _speaker, person_id in turns]
    # While interleaved, no turn was ever attributed to a speaker other than
    # its own or to the startup person.
    assert startup.person_id not in seen
    assert seen[1] == bob.person_id and seen[2] is None and seen[3] == bob.person_id


async def test_anonymous_voice_event_does_not_carry_the_startup_person(rt):
    from companion.cli import handle_user_message

    startup = await _person(rt, "Startup")
    await rt.conversations.get_or_create("voice-c1")
    captured = {}

    async def runner(rt_, event, **kwargs):
        captured["event"] = event
        return {"delivered": True, "status": "WAIT", "action": "WAIT"}

    await handle_user_message(
        rt,
        text="这是谁在说话？",
        source=EventSource.VOICE,
        submitted_at=0.0,
        conversation_id="voice-c1",
        person_id=None,
        addressee="agent",
        turn_runner=runner,
    )

    event = captured["event"]
    assert event.speaker_person_id is None
    assert event.speaker_person_id != startup.person_id
