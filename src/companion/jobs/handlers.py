"""Job handlers.

``memory_consolidation`` turns the memory candidates proposed for one event
into durable memories through :class:`MemoryWriter`. Authorisation is
re-checked at execution time (never at enqueue time): if the speaker or the
subject has revoked learning consent by the time the job runs, the candidate
is denied rather than written.

The handler is idempotent: each candidate has at most one durable
candidate -> memory mapping, and re-running the job reads those mappings
instead of writing a second memory.
"""

from __future__ import annotations

import logging

from companion.memory.consolidation import MemoryWriter
from companion.memory.service import build_memory_service
from companion.storage.repositories import Repositories


class JobError(Exception):
    """Raised by a job handler to record a business-level failure."""


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

    service = build_memory_service(repos)
    writer = MemoryWriter(repos, service, logger=logger)
    outcomes = await writer.consolidate_event(event_id)

    logger.info(
        "memory_consolidation_done event_id=%s candidates=%d outcomes=%s",
        event_id,
        len(candidates),
        ",".join(f"{o['candidate_id']}:{o['outcome']}" for o in outcomes) or "none",
    )
