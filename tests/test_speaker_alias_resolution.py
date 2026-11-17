"""V0.9.9.31: canonical speaker ids, promotion/merge and binding conflicts.

Speaker profiles and their aliases live in the speaker SQLite store; person
bindings live in Ava's main database. A merge can therefore never update
bindings atomically. These tests pin the safe resolution rules: follow any
valid alias to the live canonical id, and refuse to choose a person when the
canonical id and its aliases disagree.
"""

from __future__ import annotations

import pytest

from companion.domain.enums import RegistrationStatus
from companion.people.identity_resolver import (
    IdentityResolver,
    IdentityState,
    REASON_ALIAS_RESOLUTION_FAILED,
    REASON_BINDING_CONFLICT,
    REASON_SPEAKER_UNBOUND,
)
from companion.people.speaker_bindings import (
    SpeakerBindingError,
    SpeakerBindingService,
)
from companion.speaker_id.models import KNOWN, QUALITY_HIGH, SpeakerResult
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


def result(speaker_id: str) -> SpeakerResult:
    return SpeakerResult(
        utterance_id=1,
        speaker_id=speaker_id,
        person_id=None,
        status=KNOWN,
        reid_quality=QUALITY_HIGH,
        similarity=0.8,
        margin=0.3,
        voiced_seconds=2.0,
        provisional=False,
        model_id="fake",
        inference_ms=1.0,
        completed_at=0.0,
    )


async def person(rt, name="Person"):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
    )


def resolver(rt, store):
    return IdentityResolver(rt.repos.person, rt.speaker_bindings, store)


# --------------------------------------------------------------- promotion
def test_current_id_resolves_to_itself(store):
    speaker_id = stable(store)
    assert store.resolve_canonical_speaker_id(speaker_id) == speaker_id


def test_promotion_keeps_the_old_alias_valid(store):
    temporary = store.create_candidate()
    canonical = store.promote_speaker(temporary)

    assert canonical is not None
    assert store.resolve_canonical_speaker_id(temporary) == canonical
    assert store.resolve_canonical_speaker_id(canonical) == canonical
    assert sorted(store.related_speaker_ids(canonical)) == sorted(
        [canonical, temporary]
    )


async def test_binding_on_a_promoted_alias_still_resolves(rt, store):
    temporary = store.create_candidate()
    canonical = store.promote_speaker(temporary)
    assert canonical is not None
    owner = await person(rt)
    # The human bound the id they saw before (or after) promotion.
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, temporary, "explicit_confirmation"
    )

    identity = await resolver(rt, store).resolve(
        result(temporary), run_id="run", utterance_id=1
    )

    assert identity.state is IdentityState.CONFIRMED
    assert identity.person_id == owner.person_id
    assert identity.canonical_speaker_id == canonical
    assert identity.raw_speaker_id == temporary


# ------------------------------------------------------------- multi-hop
def test_multi_hop_alias_chain_reaches_the_live_id(store):
    base = stable(store)
    second = stable(store)
    # Simulate an older promotion history: OLD -> second -> base. The live id
    # is ``base``; only the last hop has a profile.
    conn = store._connection
    conn.execute(
        "INSERT INTO speaker_aliases(canonical_speaker_id, alias_speaker_id,"
        " created_at) VALUES (?, ?, ?)",
        (second, "OLD-0001", 0.0),
    )
    conn.execute(
        "INSERT INTO speaker_aliases(canonical_speaker_id, alias_speaker_id,"
        " created_at) VALUES (?, ?, ?)",
        (base, second, 0.0),
    )

    assert store.resolve_canonical_speaker_id("OLD-0001") == base
    assert base in store.related_speaker_ids("OLD-0001")


def test_alias_cycle_is_detected_not_guessed(store):
    first = stable(store)
    second = stable(store)
    conn = store._connection
    conn.execute(
        "INSERT OR REPLACE INTO speaker_aliases(canonical_speaker_id,"
        " alias_speaker_id, created_at) VALUES (?, ?, ?)",
        (second, first, 0.0),
    )
    conn.execute(
        "INSERT OR REPLACE INTO speaker_aliases(canonical_speaker_id,"
        " alias_speaker_id, created_at) VALUES (?, ?, ?)",
        (first, second, 0.0),
    )

    assert store.resolve_canonical_speaker_id(first) is None
    assert store.resolve_canonical_speaker_id(second) is None


def test_hop_budget_exhaustion_returns_none(store):
    base = stable(store)
    conn = store._connection
    # A long chain with a live base at the end; a 2-hop budget must refuse.
    previous = "A-0000"
    for index in range(1, 6):
        current = f"A-{index:04d}"
        conn.execute(
            "INSERT INTO speaker_aliases(canonical_speaker_id,"
            " alias_speaker_id, created_at) VALUES (?, ?, ?)",
            (current, previous, 0.0),
        )
        previous = current
    conn.execute(
        "INSERT INTO speaker_aliases(canonical_speaker_id, alias_speaker_id,"
        " created_at) VALUES (?, ?, ?)",
        (base, previous, 0.0),
    )

    assert store.resolve_canonical_speaker_id("A-0000", max_hops=2) is None


def test_deleted_or_unknown_id_has_no_canonical(store):
    assert store.resolve_canonical_speaker_id("S-9999") is None
    temporary = store.create_candidate()
    store.delete_speaker(temporary)
    assert store.resolve_canonical_speaker_id(temporary) is None


# -------------------------------------------------- alias closure completeness
def test_alias_closure_is_complete_for_a_normal_merge(store):
    canonical = stable(store)
    merged = store.create_candidate()
    assert store.merge_speakers(canonical, merged) is True

    closure = store.resolve_alias_closure(canonical)
    assert closure.complete is True
    assert closure.canonical == canonical
    assert merged in closure.speaker_ids
    assert set(closure.speaker_ids) == {
        canonical,
        merged,
        *(
            row["alias_speaker_id"]
            for row in store.aliases()
            if row["canonical_speaker_id"] == canonical
        ),
    }
    assert closure.reason is None


def test_alias_closure_reports_a_truncated_traversal(store):
    canonical = stable(store)
    merged = store.create_candidate()
    store.merge_speakers(canonical, merged)

    closure = store.resolve_alias_closure(canonical, max_ids=1)
    assert closure.complete is False
    assert closure.canonical == canonical
    assert closure.speaker_ids == (canonical,)
    assert closure.reason == "alias_limit_exceeded"


def test_alias_closure_reports_unknown_and_broken_chains(store):
    unknown = store.resolve_alias_closure("S-9999")
    assert unknown.complete is False
    assert unknown.canonical is None
    assert unknown.reason == "unknown_id"

    live = stable(store)
    conn = store._connection
    # A live profile that is also an alias of a deleted parent: the chain
    # cannot reach a live canonical id.
    conn.execute(
        "INSERT OR REPLACE INTO speaker_aliases(canonical_speaker_id,"
        " alias_speaker_id, created_at) VALUES ('S-9999', ?, 0.0)",
        (live,),
    )
    broken = store.resolve_alias_closure(live)
    assert broken.complete is False
    assert broken.canonical is None
    assert broken.reason == "broken_alias_chain"


class _CanonicalOnlyStore:
    """A store that can answer the old API but cannot prove completeness."""

    def resolve_canonical_speaker_id(self, speaker_id):
        return speaker_id

    def related_speaker_ids(self, speaker_id):
        return [speaker_id]


async def test_resolver_never_degrades_to_canonical_only_lookup(rt, store):
    speaker_id = stable(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, speaker_id, "explicit_confirmation"
    )

    identity = await resolver(rt, _CanonicalOnlyStore()).resolve(
        result(speaker_id), run_id="run", utterance_id=1
    )

    assert identity.state is IdentityState.UNKNOWN
    assert identity.person_id is None
    assert identity.reason == REASON_ALIAS_RESOLUTION_FAILED


async def test_alias_cycle_fails_closed_with_an_explicit_reason(rt, store):
    first = stable(store)
    second = stable(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, first, "explicit_confirmation"
    )
    conn = store._connection
    conn.execute(
        "INSERT OR REPLACE INTO speaker_aliases(canonical_speaker_id,"
        " alias_speaker_id, created_at) VALUES (?, ?, 0.0)",
        (second, first),
    )
    conn.execute(
        "INSERT OR REPLACE INTO speaker_aliases(canonical_speaker_id,"
        " alias_speaker_id, created_at) VALUES (?, ?, 0.0)",
        (first, second),
    )

    identity = await resolver(rt, store).resolve(
        result(first), run_id="run", utterance_id=1
    )

    assert identity.state is IdentityState.UNKNOWN
    assert identity.person_id is None
    assert identity.reason == REASON_ALIAS_RESOLUTION_FAILED


# ------------------------------------------------------------------ merge
async def test_merge_still_resolves_binding_on_the_old_id(rt, store):
    canonical = stable(store)
    merged = stable(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, merged, "explicit_confirmation"
    )

    assert store.merge_speakers(canonical, merged) is True
    assert store.resolve_canonical_speaker_id(merged) == canonical

    identity = await resolver(rt, store).resolve(
        result(merged), run_id="run", utterance_id=1
    )
    assert identity.state is IdentityState.CONFIRMED
    assert identity.person_id == owner.person_id
    assert identity.canonical_speaker_id == canonical


async def test_merge_of_different_people_is_a_conflict_not_a_choice(rt, store):
    canonical = stable(store)
    merged = stable(store)
    first = await person(rt, "First")
    second = await person(rt, "Second")
    await rt.speaker_bindings.bind_speaker(
        first.person_id, canonical, "owner"
    )
    await rt.speaker_bindings.bind_speaker(
        second.person_id, merged, "owner"
    )

    assert store.merge_speakers(canonical, merged) is True

    identity = await resolver(rt, store).resolve(
        result(canonical), run_id="run", utterance_id=1
    )
    assert identity.state is IdentityState.UNKNOWN
    assert identity.person_id is None
    assert identity.reason == REASON_BINDING_CONFLICT


async def test_bind_refuses_a_voice_bound_to_another_person(rt, store):
    canonical = stable(store)
    merged = stable(store)
    first = await person(rt, "First")
    second = await person(rt, "Second")
    await rt.speaker_bindings.bind_speaker(
        first.person_id, merged, "owner"
    )
    store.merge_speakers(canonical, merged)
    related = store.related_speaker_ids(canonical)

    # ``canonical`` itself has no exact binding, but one of its aliases is
    # bound to First. Binding the whole voice to Second would silently move a
    # person, so it is refused.
    with pytest.raises(SpeakerBindingError) as error:
        await rt.speaker_bindings.bind_speaker(
            second.person_id,
            canonical,
            "owner",
            related_speaker_ids=related,
        )
    assert error.value.code == "speaker_alias_bound_elsewhere"

    # The original manual binding was not overwritten.
    binding = await rt.speaker_bindings.get_binding(merged)
    assert binding is not None and binding["person_id"] == first.person_id


async def test_unbind_removes_all_related_aliases(rt, store):
    canonical = stable(store)
    merged = stable(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(owner.person_id, merged, "owner")
    store.merge_speakers(canonical, merged)
    related = store.related_speaker_ids(canonical)

    assert await rt.speaker_bindings.unbind_speaker(
        canonical, related_speaker_ids=related
    )
    assert await rt.speaker_bindings.bindings_for_speaker_ids(related) == []


async def test_deleted_speaker_binding_is_unusable(rt, store):
    speaker_id = stable(store)
    owner = await person(rt)
    await rt.speaker_bindings.bind_speaker(
        owner.person_id, speaker_id, "owner"
    )
    store.delete_speaker(speaker_id)

    identity = await resolver(rt, store).resolve(
        result(speaker_id), run_id="run", utterance_id=1
    )
    assert identity.state is IdentityState.UNKNOWN
    assert identity.reason == REASON_SPEAKER_UNBOUND


async def test_inactive_person_binding_cannot_be_rebound(rt, store):
    speaker_id = stable(store)
    owner = await person(rt)
    await rt.registry.set_status(owner.person_id, RegistrationStatus.SUSPENDED)

    with pytest.raises(SpeakerBindingError) as error:
        await rt.speaker_bindings.bind_speaker(
            owner.person_id, speaker_id, "owner"
        )
    assert error.value.code == "person_not_active"


def test_binding_service_is_importable():
    assert SpeakerBindingService is not None
