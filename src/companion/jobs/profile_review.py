"""Profile review scheduling and the background PROFILE_REVIEW handler.

A conversation turn never waits for profile reflection. At commit time the
trusted path records a receipt and one durable job in the *same* SQLite
transaction as the turn. The worker later loads the complete profile, the
authorized evidence and the model proposal, then applies only what local rules
accept.

Ordering per person is enforced by processing the oldest pending receipt
first. Receipt ``(person_id, source_event_id)`` is unique, so a retried turn
cannot queue the same event twice, and ``apply_refinement`` refuses to apply
the same event to a second revision.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from companion.domain.enums import JobType
from companion.domain.models import utcnow
from companion.people.profile_service import ProfileUnauthorized

TERMINAL_RECEIPT_STATUSES = frozenset(
    {"COMPLETED", "SKIPPED", "FAILED", "CANCELLED"}
)
MAX_RECEIPTS_PER_JOB = 20


class JobError(Exception):
    """Business-level failure; the worker records it and may retry."""


def profile_review_job_key(person_id: str, source_event_id: str) -> str:
    return f"profile-review:{person_id}:{source_event_id}"


class ProfileReviewScheduler:
    """Atomic receipt + job request used inside the turn commit.

    ``request`` is called while the turn's transaction is open. Both writes
    join that transaction, so a rolled-back turn leaves no review behind and a
    committed turn cannot lose its review request.
    """

    def __init__(
        self,
        repository,
        jobs,
        *,
        persons=None,
        delay_seconds: float = 0.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repository = repository
        self.jobs = jobs
        self.persons = persons
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.logger = logger or logging.getLogger("companion.profile_review")

    async def request(self, person_id: str, source_event_id: str) -> Optional[str]:
        if not person_id or not source_event_id:
            return None
        if self.persons is not None:
            consent = await self.persons.get_consent(person_id)
            if consent is None or not (
                consent.allow_processing
                and consent.allow_persistence
                and consent.allow_memory_learning
                and consent.allow_external_ai
            ):
                # A local-only person is never sent to a refinement model.
                return None
        review_id = await self.repository.ensure_review_receipt(
            person_id=person_id,
            source_event_id=source_event_id,
            job_key=profile_review_job_key(person_id, source_event_id),
        )
        receipt = await self.repository.get_review_receipt_by_id(review_id)
        if receipt is not None and receipt["status"] in TERMINAL_RECEIPT_STATUSES:
            return review_id
        await self.jobs.enqueue(
            JobType.PROFILE_REVIEW,
            profile_review_job_key(person_id, source_event_id),
            {"person_id": person_id, "source_event_id": source_event_id},
            next_run_at=(
                utcnow() + timedelta(seconds=self.delay_seconds)
                if self.delay_seconds > 0
                else None
            ),
        )
        return review_id


def build_profile_review_handler(
    engine,
    *,
    logger: Optional[logging.Logger] = None,
    max_receipts: int = MAX_RECEIPTS_PER_JOB,
):
    """Build the worker handler around a ProfileRefinementEngine."""

    async def profile_review(payload: dict, repos, log: logging.Logger) -> None:
        person_id = str(payload.get("person_id") or "")
        source_event_id = str(payload.get("source_event_id") or "")
        if not person_id or not source_event_id:
            raise JobError("profile_review payload missing person_id/source_event_id")

        repository = repos.profile
        await repository.recover_stale_review_receipts(person_id)
        receipt = await repository.get_review_receipt(person_id, source_event_id)
        if receipt is None:
            raise JobError(
                f"profile review receipt missing for {person_id}/{source_event_id}"
            )
        if receipt["status"] in TERMINAL_RECEIPT_STATUSES:
            return

        pending = await repository.list_review_receipts(
            person_id, status="PENDING", limit=max(1, int(max_receipts))
        )
        # Oldest first: a later turn may never commit before an earlier one.
        # Receipts are created in event order by the turn transaction.
        processed = 0
        for item in pending:
            if item["status"] != "PENDING":
                continue
            if not await repository.claim_review_receipt(item["review_id"]):
                continue
            outcome = await _review_one(engine, repository, item, log)
            processed += 1
            if outcome == "retry":
                break
        if processed == 0:
            current = await repository.get_review_receipt_by_id(
                receipt["review_id"]
            )
            if current is not None and current["status"] == "PENDING":
                # Nothing claimed (another worker owns it); leave the job
                # pending rather than falsely reporting completion.
                raise JobError("profile review receipt is being processed")
        log.info(
            "profile_review_done event_id=%s person_id=%s receipts=%d",
            source_event_id,
            person_id,
            processed,
        )

    return profile_review


async def _review_one(engine, repository, receipt: dict, log: logging.Logger) -> str:
    person_id = receipt["person_id"]
    source_event_id = receipt["source_event_id"]
    try:
        result = await engine.review(person_id, source_event_id)
    except ProfileUnauthorized:
        await _complete(repository, receipt, "SKIPPED", "unauthorized")
        return "done"
    except Exception as exc:
        await repository.update_review_receipt(
            receipt["review_id"],
            status="PENDING",
            last_error=f"{type(exc).__name__}: {exc}"[:500],
            increment_attempt=True,
        )
        raise JobError(f"profile review failed: {exc}") from exc

    if result.status in ("NO_PROFILE", "UNAUTHORIZED"):
        await _complete(repository, receipt, "SKIPPED", result.reason or result.status)
        return "done"

    try:
        applied = await engine.apply_proposals(result)
        if applied.status == "REVISION_CONFLICT":
            # Reload the full profile and ask again against the newer
            # revision. Never just replace base_revision on the old proposal.
            log.info(
                "profile_review_revision_conflict person_id=%s event_id=%s",
                person_id,
                source_event_id,
            )
            result = await engine.review(person_id, source_event_id)
            applied = await engine.apply_proposals(result)
    except Exception as exc:
        await repository.update_review_receipt(
            receipt["review_id"],
            status="PENDING",
            last_error=f"{type(exc).__name__}: {exc}"[:500],
            increment_attempt=True,
        )
        raise JobError(f"profile review apply failed: {exc}") from exc

    changes = (
        list(applied.applied)
        + list(applied.pending)
        + list(applied.rejected)
    )
    if applied.status == "REVISION_CONFLICT":
        await repository.update_review_receipt(
            receipt["review_id"],
            status="PENDING",
            base_revision=applied.base_revision,
            decision=result.decision,
            reason="profile_revision_changed",
            changes=changes,
            increment_attempt=True,
            last_error="revision_conflict_after_retry",
        )
        return "retry"
    if applied.status == "UNAUTHORIZED":
        await _complete(repository, receipt, "SKIPPED", applied.reason)
        return "done"

    decision = (
        "NO_CHANGE"
        if applied.status == "NO_CHANGE"
        else "PENDING_CONFIRMATION"
        if applied.status == "PENDING_CONFIRMATION"
        else "PROPOSE_CHANGES"
    )
    await repository.update_review_receipt(
        receipt["review_id"],
        status="COMPLETED",
        base_revision=applied.base_revision or result.base_revision,
        result_revision=applied.new_revision,
        decision=decision,
        reason=applied.reason or result.reason,
        changes=changes,
        completed=True,
    )
    log.info(
        "profile_review_applied person_id=%s event_id=%s status=%s"
        " revision=%s->%s applied=%d pending=%d",
        person_id,
        source_event_id,
        applied.status,
        applied.base_revision,
        applied.new_revision,
        len(applied.applied),
        len(applied.pending),
    )
    return "done"


async def _complete(
    repository, receipt: dict, status: str, reason: str
) -> None:
    await repository.update_review_receipt(
        receipt["review_id"],
        status=status,
        decision=status,
        reason=(reason or status)[:500],
        completed=True,
    )


__all__ = [
    "JobError",
    "MAX_RECEIPTS_PER_JOB",
    "ProfileReviewScheduler",
    "build_profile_review_handler",
    "profile_review_job_key",
]
