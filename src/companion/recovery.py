"""Event recovery after an unclean shutdown.

A crash can leave an utterance row in ``PROCESSING`` (or ``FAILED``) after
some, but not all, durable stages completed. :class:`RecoveryManager` discovers
those events and safely retries them by re-submitting the event through the
orchestrator. Because the orchestrator's pipeline is *resumable* — it re-uses
an existing analysis, candidates, memory job and action instead of clearing and
re-running them — recovery is idempotent:

- a completed analysis is never recomputed or overwritten,
- a completed memory job is never re-created or left stranding new candidates,
- an already-persisted action (and its response) is re-used, not re-emitted.

Recovery never blindly resets every ``PROCESSING`` event to ``PENDING``; it
resumes from the durable records already present.
"""

from __future__ import annotations

from typing import Optional

from companion.domain.enums import ProcessingStatus
from companion.domain.models import ProcessingReport, UtteranceEvent
from companion.storage.repositories import Repositories


class RecoveryError(Exception):
    """Raised when a recovery operation cannot proceed."""


class RecoveryManager:
    def __init__(self, repos: Repositories, orchestrator) -> None:
        self.repos = repos
        self.orchestrator = orchestrator

    async def list_unfinished(self) -> list[UtteranceEvent]:
        """Return events that did not reach ``COMPLETED`` (in-flight or failed)."""
        processing = await self.repos.utterance.list_by_status(
            ProcessingStatus.PROCESSING
        )
        failed = await self.repos.utterance.list_by_status(ProcessingStatus.FAILED)
        # PROCESSING first (older), then FAILED; both oldest-first.
        return processing + failed

    async def retry(self, event_id: str) -> ProcessingReport:
        stored = await self.repos.utterance.get(event_id)
        if stored is None:
            raise RecoveryError(f"event not found: {event_id}")
        # Re-submit with the same immutable identity; the orchestrator resumes
        # from whatever durable stages already exist.
        event = stored.model_copy(update={"processing_status": ProcessingStatus.PENDING})
        return await self.orchestrator.process_utterance(event)
