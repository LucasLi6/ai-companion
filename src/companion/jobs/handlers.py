"""Job handlers.

V0.0 ships exactly one handler, ``memory_consolidation``. It validates the
source event and its candidate associations, re-checks the *current*
authorisation of both the speaker (source of the statement) and the subject
(the person described) before consolidating, and refuses to turn a third-party
statement into a confirmed fact about its subject. It does **not** promote
candidates into long-term facts — that is the V0.2 memory-learning milestone.
The handler is idempotent: re-running it never duplicates side effects.
"""

from __future__ import annotations

import logging

from companion.domain.enums import MemoryCandidateStatus, RegistrationStatus
from companion.storage.repositories import Repositories


class JobError(Exception):
    """Raised by a job handler to record a business-level failure."""


async def _learning_authorized(repos: Repositories, person_id: str | None) -> bool:
    """A person may only be learned about while active and consenting."""
    if person_id is None:
        return False
    person = await repos.person.get(person_id)
    if person is None or person.registration_status != RegistrationStatus.ACTIVE:
        return False
    consent = await repos.person.get_consent(person_id)
    return consent is not None and consent.allow_memory_learning


async def memory_consolidation(
    payload: dict, repos: Repositories, logger: logging.Logger
) -> None:
    event_id = payload.get("event_id")
    if not event_id:
        raise JobError("memory_consolidation payload missing 'event_id'")

    event = await repos.utterance.get(event_id)
    if event is None:
        raise JobError(f"source event {event_id} not found")

    candidates = await repos.analysis.get_candidates(event_id)

    # Idempotency: a candidate that already reached a terminal state is never
    # re-processed, so re-running this job (after a crash, a retry or a lease
    # expiry) has no further side effects.
    terminal = {
        MemoryCandidateStatus.CONSOLIDATED,
        MemoryCandidateStatus.SKIPPED,
    }
    pending = [c for c in candidates if c.status not in terminal]

    for cand in pending:
        if cand.source_event_id != event_id:
            raise JobError(
                f"candidate {cand.candidate_id} references event {cand.source_event_id},"
                f" expected {event_id}"
            )
        if cand.subject_person_id is not None:
            person = await repos.person.get(cand.subject_person_id)
            if person is None:
                raise JobError(
                    f"candidate {cand.candidate_id} references missing person"
                    f" {cand.subject_person_id}"
                )

    # The speaker is the *source* of the statement; the subject is the person
    # the statement is about. Both must currently authorise learning, and a
    # third-party statement (speaker != subject) must never auto-become a
    # confirmed fact about its subject.
    speaker_id = event.speaker_person_id
    source_authorized = await _learning_authorized(repos, speaker_id)

    for cand in pending:
        subject_id = cand.subject_person_id or speaker_id
        subject_authorized = await _learning_authorized(repos, subject_id)

        if source_authorized and subject_authorized and speaker_id == subject_id:
            await repos.analysis.update_candidate_status(
                cand.candidate_id, MemoryCandidateStatus.CONSOLIDATED
            )
            continue

        if not source_authorized:
            reason = "source_authorization_revoked"
        elif not subject_authorized:
            reason = "subject_privacy_denied"
        else:
            # speaker != subject: a third-party report, not a self-reported fact.
            reason = "third_party_report_not_confirmed"

        await repos.analysis.update_candidate_status(
            cand.candidate_id, MemoryCandidateStatus.SKIPPED
        )
        logger.warning(
            "memory_candidate_skipped candidate_id=%s speaker_person_id=%s"
            " subject_person_id=%s reason=%s",
            cand.candidate_id,
            speaker_id,
            subject_id,
            reason,
        )

    logger.info(
        "memory_consolidation_done event_id=%s candidates=%d processed=%d",
        event_id,
        len(candidates),
        len(pending),
    )
    # V0.0: intentionally writes no rows to memory_facts.
