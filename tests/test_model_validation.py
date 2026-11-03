"""P2 — pydantic model validation: UTC datetimes, non-empty ids, lengths,
illegal status/action rejection, and the event identity fingerprint.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from companion.domain.enums import AddresseeHint, PersonRole
from companion.domain.models import Person, UtteranceEvent, make_utterance_event


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError):
        Person(
            person_id="p1",
            display_name="x",
            role=PersonRole.GUEST,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )


def test_non_utc_offset_datetime_rejected():
    with pytest.raises(ValidationError):
        Person(
            person_id="p1",
            display_name="x",
            role=PersonRole.GUEST,
            created_at=datetime.now(timezone(timedelta(hours=8))),
            updated_at=datetime.now(timezone(timedelta(hours=8))),
        )


def test_utc_datetime_accepted():
    now = datetime.now(timezone.utc)
    person = Person(
        person_id="p1", display_name="x", role=PersonRole.GUEST,
        created_at=now, updated_at=now,
    )
    assert person.created_at.tzinfo is not None


def test_empty_event_id_rejected():
    with pytest.raises(ValidationError):
        make_utterance_event(
            conversation_id="c", text="x", addressee_hint="agent", event_id=""
        )


def test_empty_conversation_id_rejected():
    with pytest.raises(ValidationError):
        make_utterance_event(conversation_id="", text="x", addressee_hint="agent")


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        make_utterance_event(conversation_id="c", text="", addressee_hint="agent")


def test_illegal_processing_status_rejected():
    with pytest.raises(ValidationError):
        UtteranceEvent(
            event_id="e1",
            conversation_id="c",
            text="x",
            source="test",
            addressee_hint=AddresseeHint.AGENT,
            occurred_at=datetime.now(timezone.utc),
            processing_status="BOGUS",
        )


def test_illegal_action_type_rejected():
    from companion.domain.enums import ReasonCode
    from companion.domain.models import AgentAction

    with pytest.raises(ValidationError):
        AgentAction(
            action_id="a1",
            event_id="e1",
            action_type="BOGUS",
            reason_code=ReasonCode.UNKNOWN,
            created_at=datetime.now(timezone.utc),
        )


def test_overlong_text_rejected():
    with pytest.raises(ValidationError):
        make_utterance_event(
            conversation_id="c", text="x" * 9000, addressee_hint="agent"
        )


def test_identity_fingerprint_excludes_occurred_at_and_status():
    a = make_utterance_event(
        conversation_id="c", text="x", addressee_hint="agent", event_id="e1",
        occurred_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    b = make_utterance_event(
        conversation_id="c", text="x", addressee_hint="agent", event_id="e1",
        occurred_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert a.identity_fingerprint() == b.identity_fingerprint()


def test_identity_fingerprint_differs_on_text():
    a = make_utterance_event(
        conversation_id="c", text="x", addressee_hint="agent", event_id="e1"
    )
    b = make_utterance_event(
        conversation_id="c", text="y", addressee_hint="agent", event_id="e1"
    )
    assert a.identity_fingerprint() != b.identity_fingerprint()
