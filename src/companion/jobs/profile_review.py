"""Profile review scheduling and the background PROFILE_REVIEW handler.

A conversation turn never waits for profile reflection. At commit time the
trusted path records a receipt and one durable job in the *same* SQLite
transaction as the turn. The worker later loads the complete profile, the
authorized evidence and the model proposal, then applies only what local rules
accept.

V0.9.7.1 hardens execution:

- a *person-level* lease, not just the per-event receipt, means at most one
  review can run for a person at a time — across workers and processes;
- the lease is renewed around the model call and re-checked immediately before
  any profile write, so a worker that lost its slot can never commit;
- receipts follow the persisted conversation order (oldest first);
- a job that exhausts its retries leaves an explainable ``FAILED`` receipt
  instead of an endlessly ``PENDING`` one;
- ``reconcile_profile_review_jobs`` repairs orphaned receipts without
  replaying changes that already committed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from typing import Optional

from companion.domain.enums import JobStatus, JobType
from companion.domain.models import utcnow
from companion.people.profile_refinement import ProfileReviewExecution
from companion.people.profile_service import (
    PersonProfileService,
    ProfileUnauthorized,
    ReviewExecutionLost,
)
from companion.tasks.context import current_task_context
from companion.tasks.models import TaskPaused

TERMINAL_RECEIPT_STATUSES = frozenset(
    {"COMPLETED", "SKIPPED", "FAILED", "CANCELLED"}
)
MAX_RECEIPTS_PER_JOB = 20
DEFAULT_REVIEW_LEASE_SECONDS = 900


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

    async def retry(self, person_id: str, source_event_id: str) -> Optional[str]:
        """Manually retry a failed or orphaned review.

        A retry gets a *new* durable execution record (fresh job key and a
        reset receipt), so it can never be silently deduplicated against the
        exhausted job row and its recovery state is visibly cleared.
        """
        if not person_id or not source_event_id:
            return None
        receipt = await self.repository.get_review_receipt(
            person_id, source_event_id
        )
        if receipt is None:
            return None
        if receipt["status"] == "COMPLETED":
            # A completed review already applied its changes; a manual retry
            # must not replay them. Requeueing the same event is a new review
            # only if the caller explicitly deletes the receipt first.
            return None
        new_key = (
            f"{profile_review_job_key(person_id, source_event_id)}"
            f":retry:{uuid.uuid4().hex[:8]}"
        )
        await self.repository.prepare_review_receipt_retry(
            receipt["review_id"], job_key=new_key
        )
        await self.jobs.enqueue(
            JobType.PROFILE_REVIEW,
            new_key,
            {"person_id": person_id, "source_event_id": source_event_id},
        )
        self.logger.info(
            "profile_review_manual_retry person_id=%s event_id=%s review_id=%s",
            person_id,
            source_event_id,
            receipt["review_id"],
        )
        return receipt["review_id"]


def _execution_owner() -> str:
    """A lease owner token unique to this execution.

    Including the job id means a stale execution can never release a newer
    execution's lease, even if the worker owner id is reused.
    """
    context = current_task_context()
    if context is not None and context.task_id:
        return f"{context.task_id}:{context.owner_id or 'direct'}"
    return f"direct:{uuid.uuid4().hex[:12]}"


def build_profile_review_handler(
    engine,
    *,
    logger: Optional[logging.Logger] = None,
    max_receipts: int = MAX_RECEIPTS_PER_JOB,
    lease_seconds: int = DEFAULT_REVIEW_LEASE_SECONDS,
):
    """Build the worker handler around a ProfileRefinementEngine."""

    async def profile_review(payload: dict, repos, log: logging.Logger) -> None:
        person_id = str(payload.get("person_id") or "")
        source_event_id = str(payload.get("source_event_id") or "")
        if not person_id or not source_event_id:
            raise JobError("profile_review payload missing person_id/source_event_id")

        repository = repos.profile
        # Verify the receipt exists before touching the lease: a bogus payload
        # must not create a lease row for an unknown person (FK) and must not
        # produce partial bookkeeping.
        receipt = await repository.get_review_receipt(person_id, source_event_id)
        if receipt is None:
            raise JobError(
                f"profile review receipt missing for {person_id}/{source_event_id}"
            )
        if receipt["status"] in TERMINAL_RECEIPT_STATUSES:
            return

        owner = _execution_owner()
        acquired = await repository.acquire_review_lease(
            person_id, owner, lease_seconds=lease_seconds
        )
        if not acquired:
            # Another worker is reviewing this person. Return the job to
            # PENDING without consuming a retry; the next pass tries again.
            raise TaskPaused({})
        try:
            await reconcile_profile_review_jobs(repos, person_id=person_id, log=log)
            await repository.recover_stale_review_receipts(person_id)
            receipt = await repository.get_review_receipt(
                person_id, source_event_id
            )
            if receipt is None:
                raise JobError(
                    f"profile review receipt missing for {person_id}/{source_event_id}"
                )
            if receipt["status"] in TERMINAL_RECEIPT_STATUSES:
                return

            # A deleted source must never be sent to the model or applied.
            # The receipt is normally CANCELLED by the deletion transaction;
            # this is the defensive check for a job that raced the delete.
            await _cancel_if_source_missing(
                repos, repository, receipt, log
            )
            receipt = await repository.get_review_receipt(
                person_id, source_event_id
            )
            if receipt is None or receipt["status"] in TERMINAL_RECEIPT_STATUSES:
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
                await _cancel_if_source_missing(
                    repos, repository, item, log
                )
                item = await repository.get_review_receipt_by_id(
                    item["review_id"]
                )
                if item is None or item["status"] != "PENDING":
                    continue
                if not await repository.claim_review_receipt(
                    item["review_id"], owner
                ):
                    continue
                outcome = await _review_one(
                    engine,
                    repository,
                    item,
                    log,
                    repos=repos,
                    owner=owner,
                    lease_seconds=lease_seconds,
                )
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
        finally:
            # Only the owner can release; a stale execution's release is a
            # no-op.
            await repository.release_review_lease(person_id, owner)

    return profile_review


async def _renew_or_abort(
    repository, person_id: str, owner: str, lease_seconds: int
) -> None:
    if not await repository.renew_review_lease(
        person_id, owner, lease_seconds=lease_seconds
    ):
        # We lost the slot (expired or taken over). Never write profile data
        # or a receipt result from a stale execution.
        raise JobError("profile_review_lease_lost")


async def _cancel_if_source_missing(
    repos, repository, receipt: dict, log: logging.Logger
) -> bool:
    """Cancel a review whose source event was permanently deleted."""
    if repos is None:
        return False
    event = await repos.utterance.get(receipt["source_event_id"])
    if event is not None:
        return False
    await repository.update_review_receipt(
        receipt["review_id"],
        status="CANCELLED",
        decision="CANCELLED",
        reason="source_event_deleted",
        completed=True,
    )
    log.info(
        "profile_review_source_missing person_id=%s event_id=%s",
        receipt["person_id"],
        receipt["source_event_id"],
    )
    return True


async def _review_one(
    engine,
    repository,
    receipt: dict,
    log: logging.Logger,
    *,
    repos=None,
    owner: Optional[str] = None,
    lease_seconds: int = DEFAULT_REVIEW_LEASE_SECONDS,
) -> str:
    person_id = receipt["person_id"]
    source_event_id = receipt["source_event_id"]
    execution = (
        ProfileReviewExecution(
            person_id=person_id,
            review_id=receipt["review_id"],
            lease_owner_id=owner,
            source_event_id=source_event_id,
        )
        if owner is not None
        else None
    )
    try:
        result = await engine.review(person_id, source_event_id)
    except ProfileUnauthorized:
        await _complete(repository, receipt, "SKIPPED", "unauthorized", owner)
        return "done"
    except Exception as exc:
        await _fail_or_retry(repository, receipt, repos, exc, owner)
        raise JobError(f"profile review failed: {exc}") from exc

    if owner is not None:
        try:
            await _renew_or_abort(repository, person_id, owner, lease_seconds)
        except JobError:
            await repository.release_review_receipt(
                receipt["review_id"],
                reason="review_lease_lost",
                execution_owner_id=owner,
            )
            raise

    if result.status in ("NO_PROFILE", "UNAUTHORIZED"):
        await _complete(
            repository, receipt, "SKIPPED", result.reason or result.status, owner
        )
        return "done"

    try:
        applied = await engine.apply_proposals(result, execution=execution)
        if applied.status == "REVISION_CONFLICT":
            # Reload the full profile and ask again against the newer
            # revision. Never just replace base_revision on the old proposal.
            log.info(
                "profile_review_revision_conflict person_id=%s event_id=%s",
                person_id,
                source_event_id,
            )
            result = await engine.review(person_id, source_event_id)
            if owner is not None:
                await _renew_or_abort(
                    repository, person_id, owner, lease_seconds
                )
            applied = await engine.apply_proposals(result, execution=execution)
    except ReviewExecutionLost:
        await repository.release_review_receipt(
            receipt["review_id"],
            reason="review_lease_lost",
            execution_owner_id=owner,
        )
        raise JobError("profile_review_lease_lost")
    except JobError:
        await repository.release_review_receipt(
            receipt["review_id"],
            reason="review_lease_lost",
            execution_owner_id=owner,
        )
        raise
    except Exception as exc:
        await _fail_or_retry(repository, receipt, repos, exc, owner)
        raise JobError(f"profile review apply failed: {exc}") from exc

    changes = list(applied.applied) + list(applied.rejected)
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
            execution_owner_id=owner,
        )
        return "retry"
    if applied.status == "UNAUTHORIZED":
        await _complete(repository, receipt, "SKIPPED", applied.reason, owner)
        return "done"

    if applied.status == "APPLIED":
        decision = "PROPOSE_CHANGES"
    elif applied.status == "NO_CHANGE":
        decision = "NO_CHANGE"
    else:
        # REJECTED (or any defensive failure status) is a completed review with
        # reasons, not a second confirmation stage.
        decision = "NO_CHANGE"
    await repository.update_review_receipt(
        receipt["review_id"],
        status="COMPLETED",
        base_revision=applied.base_revision or result.base_revision,
        result_revision=applied.new_revision,
        decision=decision,
        reason=applied.reason or result.reason,
        changes=changes,
        completed=True,
        execution_owner_id=owner,
        applied_count=applied.applied_count,
        rejected_count=applied.rejected_count,
    )
    log.info(
        "profile_review_applied person_id=%s event_id=%s status=%s"
        " revision=%s->%s applied=%d rejected=%d",
        person_id,
        source_event_id,
        applied.status,
        applied.base_revision,
        applied.new_revision,
        applied.applied_count,
        applied.rejected_count,
    )
    return "done"


async def _fail_or_retry(
    repository, receipt: dict, repos, exc: Exception, owner: Optional[str] = None
) -> None:
    """Record the failure, or fail the receipt when the job is out of retries.

    The worker only retries a finite number of times. Without this, the last
    failed attempt would leave the receipt ``PENDING`` forever and block every
    later review of the same person.
    """
    error = f"{type(exc).__name__}: {exc}"[:500]
    status = "PENDING"
    exhausted = False
    if repos is not None:
        context = current_task_context()
        if context is not None and context.task_id:
            job = await repos.job.get(context.task_id)
            if job is not None and int(job.attempts) >= int(job.max_attempts):
                exhausted = True
    if exhausted:
        status = "FAILED"
    await repository.update_review_receipt(
        receipt["review_id"],
        status=status,
        decision=status if status == "FAILED" else None,
        reason="retry_exhausted" if exhausted else None,
        last_error=error,
        increment_attempt=True,
        completed=exhausted,
        execution_owner_id=owner,
    )


async def _complete(
    repository, receipt: dict, status: str, reason: str, owner: Optional[str] = None
) -> None:
    await repository.update_review_receipt(
        receipt["review_id"],
        status=status,
        decision=status,
        reason=(reason or status)[:500],
        completed=True,
        execution_owner_id=owner,
    )


async def reconcile_profile_review_jobs(
    repos,
    *,
    person_id: Optional[str] = None,
    limit: int = 200,
    log: Optional[logging.Logger] = None,
) -> dict:
    """Repair review jobs whose receipt and job rows disagree.

    This never replays an applied change: a receipt belongs to one event, and
    ``apply_refinement`` refuses to apply an event twice. Orphaned receipts are
    re-enqueued, jobs that already failed mark their receipt ``FAILED``, and a
    receipt left ``RUNNING`` by a dead worker returns to ``PENDING``.
    """
    logger = log or logging.getLogger("companion.profile_review")
    repository = repos.profile
    receipts = await repository.list_unfinished_review_receipts(
        person_id=person_id, limit=limit
    )
    fixed = {
        "checked": len(receipts),
        "enqueued": 0,
        "reset_running": 0,
        "failed": 0,
        "cancelled": 0,
    }
    for receipt in receipts:
        job_key = receipt.get("job_key") or profile_review_job_key(
            receipt["person_id"], receipt["source_event_id"]
        )
        # Never re-queue or repair a review whose source was deleted. The
        # deletion transaction normally cancels the receipt; this covers a
        # crash between the two writes.
        source = await repos.utterance.get(receipt["source_event_id"])
        if source is None:
            await repository.update_review_receipt(
                receipt["review_id"],
                status="CANCELLED",
                decision="CANCELLED",
                reason="source_event_deleted",
                completed=True,
            )
            fixed["cancelled"] += 1
            continue
        job = await repos.job.get_by_key(job_key) if job_key else None
        if receipt["status"] == "RUNNING":
            if job is None or job.status != JobStatus.RUNNING:
                await repository.update_review_receipt(
                    receipt["review_id"],
                    status="PENDING",
                    last_error="reconciled_orphaned_runner",
                )
                receipt = dict(receipt, status="PENDING")
                fixed["reset_running"] += 1
            else:
                # A live execution owns this receipt; leave it alone.
                continue
        if job is not None and job.status == JobStatus.FAILED:
            await repository.update_review_receipt(
                receipt["review_id"],
                status="FAILED",
                decision="FAILED",
                reason="job_failed",
                last_error=(job.last_error or "job_failed")[:500],
                completed=True,
            )
            fixed["failed"] += 1
            continue
        if job is not None and job.status == JobStatus.CANCELLED:
            await repository.update_review_receipt(
                receipt["review_id"],
                status="CANCELLED",
                decision="CANCELLED",
                reason="job_cancelled",
                completed=True,
            )
            fixed["cancelled"] += 1
            continue
        if job is not None and job.status == JobStatus.COMPLETED:
            # The handler normally writes a terminal receipt before the job
            # completes. Reaching this state means the receipt was reset out
            # of band; do not replay a possibly-applied change silently.
            await repository.update_review_receipt(
                receipt["review_id"],
                status="FAILED",
                decision="FAILED",
                reason="job_completed_without_receipt",
                last_error="job_completed_without_receipt",
                completed=True,
            )
            fixed["failed"] += 1
            continue
        if job is None:
            await repos.job.enqueue(
                JobType.PROFILE_REVIEW,
                job_key,
                {
                    "person_id": receipt["person_id"],
                    "source_event_id": receipt["source_event_id"],
                },
            )
            fixed["enqueued"] += 1
    if any(
        fixed[key] for key in ("enqueued", "reset_running", "failed", "cancelled")
    ):
        logger.info("profile_review_reconciled %s", fixed)
    return fixed


LEGACY_SKIPPED_REASON = "no authorized profile"
MAX_REPAIR_SCAN = 500
MAX_REPAIR_LIMIT = 200


async def repair_skipped_reviews(
    repos,
    *,
    person_id: str,
    dry_run: bool = True,
    limit: int = 30,
    log: Optional[logging.Logger] = None,
) -> dict:
    """Requeue reviews that were skipped only because the profile row was missing.

    V0.9.9.3 fix: older accounts had no ``person_interaction_profiles`` row, so
    every review ended ``SKIPPED`` with reason ``no authorized profile``. This
    repairs exactly those receipts for one person:

    - only ``status=SKIPPED`` with that exact reason (newer skips, such as a
      withdrawn consent or a deleted source, are never touched);
    - re-checks every consent and the ACTIVE status before doing anything;
    - refuses receipts whose source utterance was deleted or whose changes
      already appear in a revision for that event;
    - never touches ``PENDING``/``RUNNING`` receipts, other people, or an
      applied review;
    - ``dry_run`` (the default) performs no writes at all;
    - uses :meth:`ProfileReviewScheduler.retry`, so each attempt gets a fresh
      job key and the receipt is reset through the normal path;
    - bounded by ``limit`` and by ``MAX_REPAIR_SCAN`` scanned receipts, and
      idempotent: a requeued receipt is no longer ``SKIPPED``.
    """
    logger = log or logging.getLogger("companion.profile_review")
    limit = max(1, min(int(limit), MAX_REPAIR_LIMIT))
    summary: dict = {
        "person_id": person_id,
        "dry_run": bool(dry_run),
        "scanned": 0,
        "matched": 0,
        "would_retry": [],
        "retried": [],
        "skipped": [],
    }
    if not person_id:
        summary["skipped"].append({"review_id": "", "reason": "missing_person_id"})
        return summary

    # ``repos.profile`` is the repository in the runtime container; the
    # service owns authorization, snapshot reads and legal-profile repair.
    service = getattr(repos, "profile_service", None)
    if service is None:
        candidate = repos.profile
        service = (
            candidate
            if hasattr(candidate, "can_review")
            else PersonProfileService(
                candidate,
                repos.person,
                audit=getattr(repos, "audit", None),
            )
        )
    repository = getattr(service, "repo", repos.profile)
    receipts = await repository.list_review_receipts(
        person_id, status="SKIPPED", limit=MAX_REPAIR_SCAN
    )
    summary["scanned"] = len(receipts)
    candidates = [
        receipt
        for receipt in receipts
        if str(receipt.get("reason") or "").strip() == LEGACY_SKIPPED_REASON
    ]
    summary["matched"] = len(candidates)
    if not candidates:
        logger.info(
            "profile_review_repair person_id=%s dry_run=%s scanned=%d matched=0",
            person_id,
            bool(dry_run),
            len(receipts),
        )
        return summary

    if not await service.can_review(person_id):
        for receipt in candidates[:limit]:
            summary["skipped"].append(
                {
                    "review_id": receipt["review_id"],
                    "reason": "not_authorized",
                }
            )
        logger.info(
            "profile_review_repair person_id=%s authorized=0 matched=%d skipped=%d",
            person_id,
            len(candidates),
            len(summary["skipped"]),
        )
        return summary

    revisions = await repository.list_revisions(person_id)
    applied_events = {
        str(item.get("source_event_id"))
        for item in revisions
        if item.get("source_event_id")
    }

    profile_exists = await repository.get(person_id) is not None
    summary["profile_exists"] = profile_exists

    if not dry_run:
        created = await service.ensure_profile(
            person_id, actor="profile_review_repair"
        )
        snapshot = await service.get_full_snapshot(person_id)
        if snapshot is None:
            summary["skipped"].append(
                {"review_id": "", "reason": "profile_unreadable"}
            )
            logger.warning(
                "profile_review_repair_profile_unreadable person_id=%s created=%s",
                person_id,
                bool(created),
            )
            return summary

    scheduler = ProfileReviewScheduler(
        repository, repos.job, persons=repos.person
    )
    attempts = 0
    for receipt in candidates:
        if attempts >= limit:
            break
        review_id = receipt["review_id"]
        event_id = receipt["source_event_id"]
        if event_id in applied_events:
            summary["skipped"].append(
                {"review_id": review_id, "reason": "already_applied"}
            )
            continue
        source = await repos.utterance.get(event_id)
        if source is None:
            summary["skipped"].append(
                {"review_id": review_id, "reason": "source_event_deleted"}
            )
            continue
        # Re-read: another process may have retried it since the scan.
        current = await repository.get_review_receipt_by_id(review_id)
        if current is None or current["status"] != "SKIPPED":
            summary["skipped"].append(
                {"review_id": review_id, "reason": "no_longer_skipped"}
            )
            continue
        if str(current.get("reason") or "").strip() != LEGACY_SKIPPED_REASON:
            summary["skipped"].append(
                {"review_id": review_id, "reason": "reason_changed"}
            )
            continue
        attempts += 1
        if dry_run:
            summary["would_retry"].append(
                {"review_id": review_id, "event_id": event_id}
            )
            continue
        result = await scheduler.retry(person_id, event_id)
        if result is None:
            summary["skipped"].append(
                {"review_id": review_id, "reason": "retry_not_scheduled"}
            )
            continue
        summary["retried"].append(
            {"review_id": review_id, "event_id": event_id}
        )

    logger.info(
        "profile_review_repair person_id=%s dry_run=%s scanned=%d matched=%d"
        " would_retry=%d retried=%d skipped=%d profile_exists=%s",
        person_id,
        bool(dry_run),
        summary["scanned"],
        summary["matched"],
        len(summary["would_retry"]),
        len(summary["retried"]),
        len(summary["skipped"]),
        profile_exists,
    )
    return summary


__all__ = [
    "JobError",
    "LEGACY_SKIPPED_REASON",
    "MAX_RECEIPTS_PER_JOB",
    "MAX_REPAIR_LIMIT",
    "MAX_REPAIR_SCAN",
    "ProfileReviewScheduler",
    "build_profile_review_handler",
    "profile_review_job_key",
    "reconcile_profile_review_jobs",
    "repair_skipped_reviews",
]
