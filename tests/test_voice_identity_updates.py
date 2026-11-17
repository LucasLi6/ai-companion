"""V0.9.9.31.1: late identity updates are display-only and never rewrite a turn.

The planner is pure: it reads the attribution-ledger entry and a resolved
identity and says what (if anything) the surface may annotate. These tests pin
the rules the REPL relies on:

* successive upgrades are shown as they arrive;
* a submitted turn is never re-submitted, downgraded or re-attributed;
* two contradictory confirmed people become an explicit conflict, never
  "latest wins".
"""

from __future__ import annotations

import pytest

from companion.people.identity_resolver import IdentityState, ResolvedIdentity
from companion.voice.identity_updates import (
    CONFIRMED,
    CONFLICT_PENDING,
    CONFLICT_SUBMITTED,
    RELABELED,
    TENTATIVE,
    plan_late_identity,
)


def identity(
    state,
    *,
    speaker=None,
    person=None,
    name="Unknown",
    reason="test",
):
    return ResolvedIdentity(
        run_id="run-a",
        utterance_id=1,
        raw_speaker_id=speaker,
        canonical_speaker_id=speaker,
        person_id=person,
        display_name=name,
        state=state,
        reason=reason,
    )


def new_entry(**overrides):
    entry = {
        "state": IdentityState.UNKNOWN,
        "submitted": False,
        "event_id": None,
        "last_displayed_state": IdentityState.UNKNOWN,
        "last_displayed_speaker_id": None,
        "last_displayed_person_id": None,
    }
    entry.update(overrides)
    return entry


def apply(entry, plan) -> None:
    """Apply exactly what the REPL applies for a plan."""
    if plan is None:
        return
    if plan.kind in (CONFLICT_PENDING, CONFLICT_SUBMITTED):
        entry["identity_conflict"] = True
        return
    if plan.state is not None:
        entry["state"] = IdentityState(plan.state)
        entry["last_displayed_state"] = entry["state"]
    if plan.speaker_id is not None:
        entry["last_displayed_speaker_id"] = plan.speaker_id
    if plan.person_id is not None:
        entry["last_displayed_person_id"] = plan.person_id


def test_successive_upgrades_are_displayed_one_by_one():
    entry = new_entry()

    first = plan_late_identity(
        entry, identity(IdentityState.TENTATIVE, speaker="TEMP-0002")
    )
    assert first.kind == TENTATIVE
    assert "TEMP-0002" in first.message
    apply(entry, first)

    second = plan_late_identity(
        entry, identity(IdentityState.TENTATIVE, speaker="S-0004")
    )
    assert second.kind == TENTATIVE
    assert "S-0004" in second.message
    apply(entry, second)

    third = plan_late_identity(
        entry,
        identity(
            IdentityState.CONFIRMED,
            speaker="S-0004",
            person="person_abc123",
            name="张三",
        ),
    )
    assert third.kind == CONFIRMED
    assert "张三" in third.message
    apply(entry, third)

    assert entry["state"] is IdentityState.CONFIRMED
    assert entry["last_displayed_person_id"] == "person_abc123"
    # The same revision repeated changes nothing.
    assert (
        plan_late_identity(
            entry,
            identity(
                IdentityState.CONFIRMED,
                speaker="S-0004",
                person="person_abc123",
                name="张三",
            ),
        )
        is None
    )


def test_a_late_unknown_never_erases_a_displayed_label():
    entry = new_entry()
    apply(
        entry,
        plan_late_identity(
            entry, identity(IdentityState.TENTATIVE, speaker="TEMP-0002")
        ),
    )

    assert (
        plan_late_identity(entry, identity(IdentityState.UNKNOWN)) is None
    )
    assert entry["last_displayed_speaker_id"] == "TEMP-0002"


def test_pending_turn_reports_two_confirmed_people_as_a_conflict():
    entry = new_entry()

    first = plan_late_identity(
        entry,
        identity(
            IdentityState.CONFIRMED,
            speaker="S-0001",
            person="person_a",
            name="A",
        ),
    )
    assert first.kind == CONFIRMED
    apply(entry, first)

    second = plan_late_identity(
        entry,
        identity(
            IdentityState.CONFIRMED,
            speaker="S-0002",
            person="person_b",
            name="B",
        ),
    )
    assert second.kind == CONFLICT_PENDING
    apply(entry, second)
    assert entry["identity_conflict"] is True
    # The first confirmed person is never silently replaced.
    assert entry["last_displayed_person_id"] == "person_a"
    # Later revisions cannot talk the conflict away.
    assert (
        plan_late_identity(
            entry,
            identity(
                IdentityState.CONFIRMED,
                speaker="S-0002",
                person="person_b",
                name="B",
            ),
        )
        is None
    )


def test_submitted_turn_is_never_re_attributed():
    entry = new_entry(
        state=IdentityState.CONFIRMED,
        submitted=True,
        event_id="evt-1",
        last_displayed_state=IdentityState.CONFIRMED,
        last_displayed_speaker_id="S-0001",
        last_displayed_person_id="person_a",
    )

    plan = plan_late_identity(
        entry,
        identity(
            IdentityState.CONFIRMED,
            speaker="S-0002",
            person="person_b",
            name="B",
        ),
    )

    assert plan.kind == CONFLICT_SUBMITTED
    apply(entry, plan)
    assert entry["identity_conflict"] is True
    assert entry["last_displayed_person_id"] == "person_a"
    assert entry["event_id"] == "evt-1"


def test_submitted_turn_may_refine_its_voice_label_only():
    entry = new_entry(
        state=IdentityState.CONFIRMED,
        submitted=True,
        last_displayed_state=IdentityState.CONFIRMED,
        last_displayed_speaker_id="TEMP-0002",
        last_displayed_person_id="person_a",
    )

    plan = plan_late_identity(
        entry,
        identity(
            IdentityState.CONFIRMED,
            speaker="S-0004",
            person="person_a",
            name="A",
        ),
    )

    assert plan.kind == RELABELED
    apply(entry, plan)
    assert entry["last_displayed_speaker_id"] == "S-0004"
    assert entry["last_displayed_person_id"] == "person_a"


@pytest.mark.parametrize(
    "state",
    [IdentityState.UNKNOWN, IdentityState.TENTATIVE],
)
def test_a_submitted_turn_is_never_downgraded(state):
    entry = new_entry(
        state=IdentityState.CONFIRMED,
        submitted=True,
        last_displayed_state=IdentityState.CONFIRMED,
        last_displayed_speaker_id="S-0001",
        last_displayed_person_id="person_a",
    )

    plan = plan_late_identity(
        entry, identity(state, speaker="TEMP-0009")
    )

    assert plan is None
    assert entry["state"] is IdentityState.CONFIRMED
