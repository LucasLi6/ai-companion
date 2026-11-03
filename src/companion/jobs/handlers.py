"""Job handlers.

V0.0 ships exactly one handler, ``memory_consolidation``. It validates the
source event and its candidate associations, re-checks the subject's *current*
authorisation (a person whose learning consent was revoked, or who was
suspended/revoked, must not have new long-term facts produced from an already
queued job), then marks the remaining candidates as consolidated. It does
**not** promote candidates into long-term facts — that is the V0.2
memory-learning milestone. The handler is idempotent: re-running it never
duplicates side effects.
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
    for cand in candidates:
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

    for cand in candidates:
        subject_id = cand.subject_person_id or event.speaker_person_id
        if await _learning_authorized(repos, subject_id):
            await repos.analysis.update_candidate_status(
                cand.candidate_id, MemoryCandidateStatus.CONSOLIDATED
            )
        else:
            # Consent was revoked / person suspended after the job was queued:
            # do not learn, and don't leave the candidate PENDING forever.
            await repos.analysis.update_candidate_status(
                cand.candidate_id, MemoryCandidateStatus.SKIPPED
            )
            logger.warning(
                "memory_candidate_skipped_unauthorized candidate_id=%s subject_person_id=%s",
                cand.candidate_id,
                subject_id,
            )

    logger.info(
        "memory_consolidation_done event_id=%s candidates=%d",
        event_id,
        len(candidates),
    )
    # V0.0: intentionally writes no rows to memory_facts.
