"""V0.9.9.31.1 Part B: explicit Speaker -> Person -> Profile links.

The service must never create a second binding table, never infer a person
from similarity, never copy profile content into the speaker store, and never
turn a link into consent. These tests exercise the four operations offline
with the real speaker store and real main database.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import RegistrationStatus
from companion.people.speaker_profile_link import (
    LINK_ALIAS_UNRESOLVED,
    LINK_BINDING_CONFLICT,
    LINK_PERSON_INACTIVE,
    LINK_PERSON_MISSING,
    LINK_UNKNOWN_SPEAKER,
    PROFILE_BINDING_CONFLICT,
    PROFILE_CONSENT_MISSING,
    PROFILE_MISSING,
    PROFILE_NO_BINDING,
    PROFILE_READY,
    PROFILE_UNKNOWN_SPEAKER,
    SpeakerLinkError,
    SpeakerProfileLinkService,
)
from companion.speaker_id.store import SpeakerStore


@pytest.fixture
def store(tmp_path):
    value = SpeakerStore(tmp_path / "speakers.sqlite3")
    value.initialize()
    yield value
    value.close()


def stable(store: SpeakerStore) -> str:
    speaker_id = store.promote_speaker(store.create_candidate())
    assert speaker_id is not None
    return speaker_id


def service(rt, store) -> SpeakerProfileLinkService:
    return SpeakerProfileLinkService(
        rt.speaker_bindings,
        rt.repos.person,
        store,
        rt.profile_service,
        audit=rt.repos.audit,
    )


async def person(rt, name="Person", *, persistence=True, processing=True):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=processing,
        allow_persistence=persistence,
        allow_memory_learning=True,
    )


# ------------------------------------------------------------------ preview
async def test_preview_never_writes_a_binding_or_a_profile(rt, store):
    speaker_id = stable(store)
    owner = await person(rt, persistence=False)
    link = service(rt, store)

    preview = await link.preview_link(speaker_id, owner.person_id)

    assert preview.canonical_speaker_id == speaker_id
    assert preview.can_link is True
    assert preview.binding is None
    assert preview.display_name == owner.display_name
    assert preview.speaker_status == "KNOWN"
    assert preview.alias_complete is True
    assert preview.person_active is True
    # Nothing was written: no binding, no profile.
    assert await rt.speaker_bindings.bindings_for_speaker_ids([speaker_id]) == []
    assert await rt.repos.profile.get(owner.person_id) is None
    assert await rt.repos.audit.list() == [] or all(
        row["action"] != "link_speaker_profile"
        for row in await rt.repos.audit.list()
    )


async def test_preview_without_a_person_never_looks_linkable(rt, store):
    speaker_id = stable(store)
    link = service(rt, store)

    preview = await link.preview_link(speaker_id)

    assert preview.can_link is False
    assert preview.reason == "no_person_selected"
    assert preview.person_id is None


async def test_preview_reports_unresolvable_alias_chains(rt, store):
    speaker_id = stable(store)
    conn = store._connection
    conn.execute(
        "INSERT OR REPLACE INTO speaker_aliases(canonical_speaker_id,"
        " alias_speaker_id, created_at) VALUES ('S-9999', ?, 0.0)",
        (speaker_id,),
    )
    link = service(rt, store)
    owner = await person(rt)

    preview = await link.preview_link(speaker_id, owner.person_id)
    assert preview.can_link is False
    assert preview.reason in (LINK_ALIAS_UNRESOLVED, LINK_UNKNOWN_SPEAKER)

    with pytest.raises(SpeakerLinkError) as error:
        await link.confirm_link(speaker_id, owner.person_id)
    assert error.value.code in (LINK_ALIAS_UNRESOLVED, LINK_UNKNOWN_SPEAKER)


async def test_preview_refuses_moving_a_bound_voice(rt, store):
    speaker_id = stable(store)
    first = await person(rt, "First")
    second = await person(rt, "Second")
    await rt.speaker_bindings.bind_speaker(
        first.person_id, speaker_id, "explicit_confirmation"
    )
    link = service(rt, store)

    preview = await link.preview_link(speaker_id, second.person_id)

    assert preview.can_link is False
    assert preview.reason == LINK_BINDING_CONFLICT
    assert preview.conflicting_person_ids == (first.person_id,)

    with pytest.raises(SpeakerLinkError) as error:
        await link.confirm_link(speaker_id, second.person_id)
    assert error.value.code == LINK_BINDING_CONFLICT
    binding = await rt.speaker_bindings.get_binding(speaker_id)
    assert binding is not None and binding["person_id"] == first.person_id


async def test_preview_reports_a_known_person_without_a_profile(rt, store):
    speaker_id = stable(store)
    owner = await person(rt, persistence=True)
    await rt.db.execute(
        "DELETE FROM person_interaction_profiles WHERE person_id = ?",
        (owner.person_id,),
    )
    link = service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)

    resolution = await link.resolve_profile(speaker_id)

    assert resolution.person_id == owner.person_id
    assert resolution.display_name == owner.display_name
    assert resolution.state == PROFILE_MISSING
    assert resolution.profile is None


# ------------------------------------------------------------------ confirm
async def test_confirm_links_and_audits_without_changing_consent(rt, store):
    speaker_id = stable(store)
    owner = await person(rt)
    before = await rt.repos.person.get_consent(owner.person_id)
    link = service(rt, store)

    result = await link.confirm_link(
        speaker_id, owner.person_id, verified_by="local_cli_confirmation"
    )

    assert result.created is True
    assert result.binding["speaker_id"] == speaker_id
    assert result.binding["person_id"] == owner.person_id
    binding = await rt.speaker_bindings.get_binding(speaker_id)
    assert binding is not None
    after = await rt.repos.person.get_consent(owner.person_id)
    assert after == before
    assert store.enrollment_consent(speaker_id) is None
    # Linking never enables profile learning on its own.
    assert await rt.profile_service.can_learn(owner.person_id) == before.allow_memory_learning
    audit = [row for row in await rt.repos.audit.list() if row["action"] == "link_speaker_profile"]
    assert len(audit) == 1
    assert audit[0]["entity_id"] == speaker_id
    assert owner.person_id in audit[0]["detail"]
    assert "display_name" not in audit[0]["detail"]


async def test_confirm_is_idempotent_and_keeps_the_alias_binding(rt, store):
    temporary = store.create_candidate()
    canonical = store.promote_speaker(temporary)
    assert canonical is not None
    owner = await person(rt)
    link = service(rt, store)

    first = await link.confirm_link(canonical, owner.person_id)
    # The human may still hold the pre-promotion id.
    second = await link.confirm_link(temporary, owner.person_id)

    assert first.created is True
    assert second.created is False
    assert second.binding["speaker_id"] == canonical
    assert (
        await rt.speaker_bindings.get_binding(canonical)
    )["person_id"] == owner.person_id


async def test_confirm_refuses_missing_and_inactive_people(rt, store):
    speaker_id = stable(store)
    link = service(rt, store)

    with pytest.raises(SpeakerLinkError) as error:
        await link.confirm_link(speaker_id, "person_missing")
    assert error.value.code == LINK_PERSON_MISSING

    owner = await person(rt)
    await rt.registry.set_status(owner.person_id, RegistrationStatus.SUSPENDED)
    with pytest.raises(SpeakerLinkError) as error:
        await link.confirm_link(speaker_id, owner.person_id)
    assert error.value.code == LINK_PERSON_INACTIVE
    assert await rt.speaker_bindings.get_binding(speaker_id) is None


async def test_confirm_rejects_blank_or_implicit_identity(rt, store):
    speaker_id = stable(store)
    owner = await person(rt)
    link = service(rt, store)

    for value in ("", "   ", None):
        with pytest.raises(SpeakerLinkError):
            await link.confirm_link(speaker_id, value)  # type: ignore[arg-type]
    with pytest.raises(SpeakerLinkError):
        await link.confirm_link(speaker_id, owner.person_id, verified_by="")
    assert await rt.speaker_bindings.get_binding(speaker_id) is None


# ------------------------------------------------------------------ resolve
async def test_resolve_profile_follows_speaker_to_person_to_profile(rt, store):
    speaker_id = stable(store)
    owner = await person(rt, "张三")
    link = service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)

    resolution = await link.resolve_profile(speaker_id)

    assert resolution.state == PROFILE_READY
    assert resolution.person_id == owner.person_id
    assert resolution.display_name == "张三"
    assert resolution.profile is not None
    assert resolution.profile.person_id == owner.person_id
    assert resolution.consent_flags == {
        "processing": True,
        "persistence": True,
        "memory_learning": True,
        "external_ai": False,
    }


async def test_resolve_profile_distinguishes_unknown_voice_from_unknown_person(
    rt, store
):
    link = service(rt, store)

    unknown = await link.resolve_profile("S-9999")
    assert unknown.state == PROFILE_UNKNOWN_SPEAKER
    assert unknown.person_id is None

    speaker_id = stable(store)
    unbound = await link.resolve_profile(speaker_id)
    assert unbound.state == PROFILE_NO_BINDING
    assert unbound.person_id is None


async def test_resolve_profile_keeps_the_person_when_consent_is_missing(
    rt, store
):
    speaker_id = stable(store)
    owner = await person(rt, persistence=True)
    link = service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)
    consent = await rt.repos.person.get_consent(owner.person_id)
    assert consent is not None
    consent.allow_persistence = False
    consent.allow_processing = False
    await rt.registry.set_consent(consent)

    resolution = await link.resolve_profile(speaker_id)

    # The person is still known; the profile is simply not readable.
    assert resolution.state == PROFILE_CONSENT_MISSING
    assert resolution.person_id == owner.person_id
    assert resolution.display_name == owner.display_name
    assert resolution.profile is None
    assert resolution.consent_flags["persistence"] is False


async def test_resolve_profile_reports_alias_bound_to_two_people(rt, store):
    canonical = stable(store)
    merged = stable(store)
    first = await person(rt, "First")
    second = await person(rt, "Second")
    await rt.speaker_bindings.bind_speaker(first.person_id, canonical, "owner")
    await rt.speaker_bindings.bind_speaker(second.person_id, merged, "owner")
    store.merge_speakers(canonical, merged)
    link = service(rt, store)

    resolution = await link.resolve_profile(canonical)

    assert resolution.state == PROFILE_BINDING_CONFLICT
    assert resolution.person_id is None


async def test_profile_content_is_never_copied_into_the_speaker_store(
    rt, store
):
    speaker_id = stable(store)
    owner = await person(rt, "Private Person")
    link = service(rt, store)
    await link.confirm_link(speaker_id, owner.person_id)
    await link.resolve_profile(speaker_id)

    row = store.get_profile(speaker_id)
    assert row is not None
    values = " ".join(str(value) for value in row.values())
    assert "Private Person" not in values
    assert "张三" not in values
    assert owner.person_id not in values


# ------------------------------------------------------------------- unlink
async def test_unlink_removes_only_the_binding(rt, store):
    canonical = stable(store)
    merged = store.create_candidate()
    store.merge_speakers(canonical, merged)
    owner = await person(rt)
    link = service(rt, store)
    await link.confirm_link(canonical, owner.person_id)
    profile_before = await rt.repos.profile.get(owner.person_id)

    assert await link.unlink(canonical) is True

    assert await rt.speaker_bindings.bindings_for_speaker_ids(
        [canonical, merged]
    ) == []
    # The canonical voice profile survives an unlink, and the merged id still
    # resolves to it as an alias.
    assert store.get_profile(canonical) is not None
    assert store.resolve_canonical_speaker_id(merged) == canonical
    assert await rt.repos.profile.get(owner.person_id) == profile_before
    audit = [
        row
        for row in await rt.repos.audit.list()
        if row["action"] == "unlink_speaker_profile"
    ]
    assert len(audit) == 1


async def test_unlink_unknown_voice_is_a_refusal_not_a_deletion(rt, store):
    link = service(rt, store)
    with pytest.raises(SpeakerLinkError) as error:
        await link.unlink("S-9999")
    assert error.value.code == LINK_UNKNOWN_SPEAKER

    # An unbound but existing voice is a quiet no-op.
    speaker_id = stable(store)
    assert await link.unlink(speaker_id) is False


class _BrokenBindings:
    """A binding lookup that errors must refuse, never look unbound."""

    async def bindings_for_speaker_ids(self, speaker_ids):
        raise RuntimeError("binding table unavailable")

    async def get_binding(self, speaker_id):
        raise RuntimeError("binding table unavailable")

    async def unbind_speaker(self, speaker_id, *, related_speaker_ids=None):
        raise RuntimeError("binding table unavailable")

    async def bind_speaker(self, *args, **kwargs):
        raise RuntimeError("binding table unavailable")


async def test_binding_store_errors_fail_closed(rt, store):
    speaker_id = stable(store)
    owner = await person(rt)
    link = SpeakerProfileLinkService(
        _BrokenBindings(),
        rt.repos.person,
        store,
        rt.profile_service,
        audit=rt.repos.audit,
    )

    with pytest.raises(SpeakerLinkError) as preview_error:
        await link.preview_link(speaker_id, owner.person_id)
    assert preview_error.value.code == "link_lookup_failed"

    with pytest.raises(SpeakerLinkError) as resolve_error:
        await link.resolve_profile(speaker_id)
    assert resolve_error.value.code == "link_lookup_failed"

    with pytest.raises(SpeakerLinkError) as unlink_error:
        await link.unlink(speaker_id)
    assert unlink_error.value.code == "link_lookup_failed"
