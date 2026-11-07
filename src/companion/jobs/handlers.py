"""Job handlers.

``memory_consolidation`` turns the memory candidates proposed for one event
into durable memories through :class:`MemoryWriter`. Authorisation is
re-checked at execution time (never at enqueue time): if the speaker or the
subject has revoked learning consent by the time the job runs, the candidate
is denied rather than written.

The handler is idempotent: each candidate has at most one durable
candidate -> memory mapping, and re-running the job reads those mappings
instead of writing a second memory.

``turn_analysis`` is the V0.6.1 lightweight extraction pass. It no longer runs
a full cognitive analysis after every Fast Path reply: it batches up to a few
already-answered events from the *same* speaker and asks one small prompt for
evidence-backed memory candidates. The local ``local_fast_path`` analysis
remains untouched, so an audit never shows a fabricated model judgement.
"""

from __future__ import annotations

import logging

from companion.core.safety import assess_safety
from companion.domain.enums import (
    CognitiveAction,
    JobType,
    MemoryKind,
    MemoryStatus,
)
from companion.memory.candidates import prepare_candidates
from companion.memory.consolidation import MemoryWriter
from companion.memory.episodes import EpisodeWriter, parse_episode_content
from companion.memory.repository import MemoryRepository
from companion.memory.service import build_memory_service
from companion.people.profile_learning import PersonProfileLearningService
from companion.people.profile_service import PersonProfileService
from companion.providers.deepseek_cognitive import CognitivePermissionDenied
from companion.storage.repositories import Repositories

# Five ordinary turns or roughly one minute, whichever comes first. The job is
# scheduled with a delay by the orchestrator; when enough events are already
# waiting it runs immediately and drains the batch.
EXTRACTION_BATCH_SIZE = 5
EXTRACTION_STATUS_PROCESSED = "processed"
EXTRACTION_STATUS_SKIPPED = "skipped"


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

    # V0.7: link the confirmed memory ids back into the speaker's understanding
    # profile, so a profile fact references the long-term memory that supports
    # it instead of duplicating the content. This is provenance only and never
    # changes whether the memory was written.
    if event.speaker_person_id is not None:
        learning = PersonProfileLearningService(
            PersonProfileService(
                repos.profile, repos.person, audit=repos.audit
            )
        )
        for memory_id in {
            outcome.get("memory_id")
            for outcome in outcomes
            if outcome.get("memory_id")
        }:
            try:
                await learning.link_output_memory(
                    event.speaker_person_id, event_id, memory_id
                )
            except Exception as exc:  # pragma: no cover - defensive isolation
                logger.warning(
                    "profile_memory_link_failed event_id=%s memory_id=%s error=%s",
                    event_id,
                    memory_id,
                    type(exc).__name__,
                )

    logger.info(
        "memory_consolidation_done event_id=%s candidates=%d outcomes=%s",
        event_id,
        len(candidates),
        ",".join(f"{o['candidate_id']}:{o['outcome']}" for o in outcomes) or "none",
    )


def build_turn_analysis_handler(
    *,
    cognitive,
    context_builder=None,
    memory=None,
    batch_size: int = EXTRACTION_BATCH_SIZE,
):
    """Deferred, lightweight extraction for Fast Path turns.

    The reply already happened with one model call. This job only extracts
    memory candidates from a same-speaker batch, so an ordinary turn never
    pays for a second full cognitive prompt. Consent is re-checked at
    execution time and each event is marked at most once in
    ``turn_extractions``; a retry after a successful batch is a no-op.
    """

    async def turn_analysis(
        payload: dict, repos: Repositories, logger: logging.Logger
    ) -> None:
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

        extract_with_episodes = getattr(
            cognitive, "extract_events_with_episodes", None
        )
        extract_plain = getattr(cognitive, "extract_events", None)
        if extract_with_episodes is None and extract_plain is None:
            # A provider without the lightweight pass is not silently given a
            # full analysis instead: that would reintroduce the cost this job
            # exists to remove.
            logger.info(
                "turn_analysis_skipped event_id=%s reason=provider", event_id
            )
            return

        # One batch, one speaker. Another person's private events are never
        # mixed in, even if they are waiting for extraction too.
        pending_ids = await repos.extraction.pending_events(
            event.conversation_id,
            event.speaker_person_id,
            limit=max(1, batch_size),
        )
        if not pending_ids:
            return
        by_id = await repos.utterance.get_many(pending_ids)
        events = [by_id[i] for i in pending_ids if i in by_id]

        eligible = []
        for candidate_event in events:
            if assess_safety(candidate_event.text).triggered:
                await repos.extraction.mark(
                    candidate_event.event_id, EXTRACTION_STATUS_SKIPPED, 0
                )
                logger.info(
                    "turn_analysis_skipped event_id=%s reason=safety",
                    candidate_event.event_id,
                )
                continue
            eligible.append(candidate_event)
        if not eligible:
            return

        prior_turns = await _prior_turns(
            repos, context_builder, eligible[0], event.speaker_person_id
        )
        replies = await _delivered_replies(repos, eligible)
        recent_episodes = await _recent_episodes(
            repos, event.speaker_person_id
        )
        try:
            if extract_with_episodes is not None:
                outcome = await extract_with_episodes(
                    eligible,
                    speaker_label=event.speaker_label
                    or event.speaker_person_id
                    or "unknown",
                    prior_turns=prior_turns,
                    replies=replies,
                    recent_episodes=recent_episodes,
                )
                extracted = outcome.candidates
                episodes = outcome.episodes
            else:
                # Older/mock providers keep working unchanged; they simply
                # cannot propose episodes.
                extracted = await extract_plain(
                    eligible,
                    speaker_label=event.speaker_label
                    or event.speaker_person_id
                    or "unknown",
                    prior_turns=prior_turns,
                )
                episodes = []
        except CognitivePermissionDenied as exc:
            for candidate_event in eligible:
                await repos.extraction.mark(
                    candidate_event.event_id, EXTRACTION_STATUS_SKIPPED, 0
                )
            logger.info(
                "turn_analysis_skipped event_id=%s reason=permission error=%s",
                event_id,
                exc,
            )
            return
        except Exception as exc:  # noqa: BLE001 — surfaced to the worker for retry
            raise JobError(f"event extraction failed: {exc}") from exc

        # Facts are committed first, and marked processed only after the
        # episode step. A failed episode write therefore retries the job
        # without ever discarding candidates that were legitimately extracted.
        async with repos.db.transaction():
            prepared_counts: dict[str, int] = {}
            for candidate_event in eligible:
                prepared = await prepare_candidates(
                    repos,
                    candidate_event,
                    extracted.get(candidate_event.event_id, []),
                    explicit_remember_request=False,
                )
                prepared_counts[candidate_event.event_id] = len(prepared)
                for candidate in prepared:
                    await repos.analysis.save_candidate(candidate)
                if prepared:
                    await repos.job.enqueue(
                        JobType.MEMORY_CONSOLIDATION,
                        f"memory:{candidate_event.event_id}",
                        {"event_id": candidate_event.event_id},
                    )

        episode_failed = False
        if episodes:
            service = getattr(memory, "service", None)
            if service is None:
                service = build_memory_service(repos)
            writer = EpisodeWriter(repos, service, logger=logger)
            for proposal in episodes:
                try:
                    result = await writer.write(
                        proposal,
                        person_id=event.speaker_person_id,
                        conversation_id=event.conversation_id,
                    )
                    logger.info(
                        "turn_analysis_episode event_id=%s outcome=%s reason=%s",
                        event_id,
                        result.outcome,
                        result.reason,
                    )
                except Exception as exc:  # noqa: BLE001 — retry the batch
                    # Validation failures are returned by the writer, not
                    # raised. Anything raised here is infrastructural (DB), and
                    # must leave the events unprocessed so the worker retries.
                    episode_failed = True
                    logger.warning(
                        "turn_analysis_episode_failed event_id=%s error=%s",
                        event_id,
                        type(exc).__name__,
                    )

        if episode_failed:
            # Deliberately do not mark these events processed: the worker's
            # retry must be able to re-attempt the episode. Candidate rows are
            # already committed and INSERT OR IGNORE makes the re-run a no-op
            # for them, so facts are never lost and never duplicated.
            logger.warning(
                "turn_analysis_episode_retry event_id=%s", event_id
            )
            raise JobError("episode write failed; batch left retryable")

        async with repos.db.transaction():
            for candidate_event in eligible:
                await repos.extraction.mark(
                    candidate_event.event_id,
                    EXTRACTION_STATUS_PROCESSED,
                    prepared_counts.get(candidate_event.event_id, 0),
                )

        logger.info(
            "turn_analysis_done event_id=%s events=%d candidates=%d episodes=%d",
            event_id,
            len(eligible),
            sum(
                len(extracted.get(candidate_event.event_id, []))
                for candidate_event in eligible
            ),
            len(episodes),
        )

    return turn_analysis


async def _prior_turns(
    repos: Repositories,
    context_builder,
    first_event,
    speaker_person_id: str,
) -> list[dict]:
    """A couple of this speaker's preceding turns, for pronoun resolution.

    Only this speaker's own words are included, and only turns the privacy
    gateway still authorises. The batch itself is never visible before its
    first event.
    """
    history = await repos.utterance.recent_context(
        first_event.conversation_id,
        exclude_event_id=None,
        limit=4,
        before_event=first_event,
    )
    history = [
        turn for turn in history if turn.speaker_person_id == speaker_person_id
    ]
    gateway = getattr(context_builder, "privacy_gateway", None)
    if gateway is not None:
        history = await gateway.filter_utterances(history)
    return [
        {"occurred_at": turn.occurred_at.isoformat(), "text": turn.text, "role": "USER"}
        for turn in history[-2:]
    ]


async def _delivered_replies(repos: Repositories, events) -> dict[str, str]:
    """Real, delivered Ava texts for this batch, keyed by the user's event id.

    Only a completed RESPOND is a real reply. WAIT/IGNORE, failed or truncated
    actions are absent, so an episode can never quote an answer Ava did not
    actually finish delivering.
    """
    actions = await repos.analysis.get_actions_for_events(
        [event.event_id for event in events]
    )
    replies: dict[str, str] = {}
    for event_id, action in actions.items():
        text = (action.response_text or "").strip()
        if (
            action.action_type == CognitiveAction.RESPOND
            and action.status == "COMPLETED"
            and text
        ):
            replies[event_id] = text
    return replies


async def _recent_episodes(
    repos: Repositories, person_id: str, limit: int = 5
) -> list[dict]:
    """Short titles/topics of this person's model-written episodes.

    The model only needs to recognise a continuation; it must never see a
    memory id, another person's episode, a human-confirmed record, or an
    episode that was retracted.
    """
    rows = await MemoryRepository(repos.db).list_for_subject(
        person_id,
        statuses=[MemoryStatus.ACTIVE.value],
        kinds=[MemoryKind.EPISODE.value],
        limit=20,
    )
    out: list[dict] = []
    for row in rows:
        if row.get("human_confirmed") or not row.get("model_inferred"):
            continue
        parsed = parse_episode_content(row.get("content"))
        if parsed is None:
            continue
        out.append(
            {
                "title": parsed["title"],
                "topic": parsed.get("topic") or "",
                "status": parsed.get("status") or "UNKNOWN",
            }
        )
        if len(out) >= limit:
            break
    return out
