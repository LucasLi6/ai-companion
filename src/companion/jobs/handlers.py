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

from companion.core.fast_path import LOCAL_PROVIDER
from companion.core.safety import assess_safety
from companion.domain.models import new_id
from companion.memory.candidates import prepare_candidates
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


def build_turn_analysis_handler(*, cognitive, context_builder, memory):
    """Deferred cognitive analysis for a Fast Path turn.

    The reply already happened with one model call. This job runs the richer
    analysis afterwards, using only history the speaker could see before the
    event (never the reply that followed), and keeps the durable memory
    pipeline working. It is idempotent: the placeholder local analysis is
    replaced at most once, and re-running after a real analysis is a no-op.
    """

    async def turn_analysis(payload: dict, repos: Repositories, logger: logging.Logger) -> None:
        event_id = payload.get("event_id")
        if not event_id:
            raise JobError("turn_analysis payload missing 'event_id'")

        event = await repos.utterance.get(event_id)
        if event is None:
            raise JobError(f"source event {event_id} not found")
        if event.speaker_person_id is None:
            return

        # Consent is re-checked at execution time, never trusted from enqueue.
        consent = await repos.person.get_consent(event.speaker_person_id)
        if (
            consent is None
            or not consent.allow_processing
            or not consent.allow_memory_learning
        ):
            logger.info("turn_analysis_skipped event_id=%s reason=consent", event_id)
            return

        existing = await repos.analysis.get_analysis(event_id)
        if existing is not None and existing.provider != LOCAL_PROVIDER:
            logger.info(
                "turn_analysis_skipped event_id=%s reason=already_analysed", event_id
            )
            return

        safety = assess_safety(event.text)
        if safety.triggered:
            logger.info("turn_analysis_skipped event_id=%s reason=safety", event_id)
            return

        memories = await memory.retrieve(event.text, event.speaker_person_id)
        if context_builder is None:
            # The Fast Path is only taken when a context builder is wired, so
            # this is a defensive guard rather than an expected state.
            raise JobError("turn_analysis requires a context builder")
        try:
            context = await context_builder.build(
                event, memories, before_event=event
            )
            analysis = await cognitive.analyze(event, context, memories)
        except Exception as exc:  # surfaced to the worker for retry
            raise JobError(f"turn analysis failed: {exc}") from exc

        if analysis.event_id != event_id:
            raise JobError(
                f"analysis event_id mismatch: {analysis.event_id!r} != {event_id!r}"
            )

        candidates = await prepare_candidates(
            repos,
            event,
            analysis.memory_candidates,
            explicit_remember_request=False,
        )
        async with repos.db.transaction():
            await repos.analysis.delete_analysis(event_id)
            await repos.persist_cognitive_outcome(
                analysis,
                new_id("ana"),
                candidates=candidates,
                job_key=f"memory:{event_id}",
                job_payload={"event_id": event_id},
                persona=None,
            )
        logger.info(
            "turn_analysis_done event_id=%s provider=%s candidates=%d",
            event_id,
            analysis.provider,
            len(candidates),
        )

    return turn_analysis
