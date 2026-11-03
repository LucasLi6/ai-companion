"""Foreground privacy gateway for external AI (Phase 2).

The orchestrator already gates ``allow_processing`` before anything runs. This
gateway is the *additional*, stricter boundary that must hold before any private
context — the current utterance, recent conversation history, or retrieved
memories — is sent to an external model (DeepSeek).

Rules enforced here:

- A person must be registered, ``ACTIVE``, and hold **both**
  ``allow_processing`` and ``allow_external_ai`` before their data may leave the
  process. ``allow_memory_learning`` is *not* the same grant.
- Every *other* person whose data would be sent is verified individually:
  speakers appearing in history, and every PERSON subject of a retrieved memory.
- A multi-subject memory is sent only when **all** of its PERSON subjects are
  authorized; otherwise it is dropped (never partially redacted, which would be
  lossy and misleading to the model).
- Non-persistence mode (``allow_persistence=False`` + ``allow_external_ai=True``)
  is fine: the gateway checks only who may be *sent* data, not whether it may be
  *stored* — persistence is the orchestrator's separate concern.

The gateway never makes a decision from model output; every person id it checks
comes from trusted application metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional

from companion.domain.enums import RegistrationStatus

SubjectIdsOf = Callable[[dict], Awaitable[list[str]]]


@dataclass(frozen=True)
class ExternalAIAuthorization:
    """Outcome of one foreground external-AI privacy check."""

    allowed: bool
    reason: str = "external_ai_not_granted"
    denied_person_ids: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.allowed


class ForegroundPrivacyGateway:
    """The single gate between foreground cognition and the external model."""

    def __init__(self, registry, *, logger: Optional[logging.Logger] = None) -> None:
        self.registry = registry
        self.logger = logger or logging.getLogger("companion.privacy")

    async def authorize_event(
        self, person_id: Optional[str]
    ) -> ExternalAIAuthorization:
        """Authorize the *speaker* of the current event for external AI.

        Returns a denied result (never raises) so callers can downgrade to a
        permission-denied cognition instead of crashing the pipeline.
        """
        if person_id is None:
            return ExternalAIAuthorization(False, reason="unregistered_speaker")
        person = await self.registry.get(person_id)
        if person is None:
            return ExternalAIAuthorization(False, reason="unknown_person")
        if person.registration_status != RegistrationStatus.ACTIVE:
            return ExternalAIAuthorization(
                False, reason=f"person_{person.registration_status.value}"
            )
        consent = await self.registry.get_consent(person_id)
        if consent is None or not consent.allow_processing:
            return ExternalAIAuthorization(False, reason="consent_withheld")
        if not consent.allow_external_ai:
            return ExternalAIAuthorization(False, reason="external_ai_not_granted")
        return ExternalAIAuthorization(True, reason="authorized")

    async def is_authorized(self, person_id: Optional[str]) -> bool:
        return (await self.authorize_event(person_id)).allowed

    async def filter_person_ids(
        self, person_ids: Iterable[str]
    ) -> tuple[list[str], list[str]]:
        """Split a set of person ids into ``(authorized, denied)``.

        Each id is verified individually against the external-AI grant; no id
        is trusted just because another one is authorized.
        """
        allowed: list[str] = []
        denied: list[str] = []
        for pid in sorted({p for p in person_ids if p}):
            if await self.is_authorized(pid):
                allowed.append(pid)
            else:
                denied.append(pid)
        return allowed, denied

    async def filter_utterances(self, utterances) -> list:
        """Drop history utterances whose speaker is not authorized for external AI.

        An utterance with no ``speaker_person_id`` (an unregistered speaker) is
        dropped too: it cannot be verified, so it is not sent.
        """
        kept = []
        for u in utterances:
            speaker_id = getattr(u, "speaker_person_id", None)
            if await self.is_authorized(speaker_id):
                kept.append(u)
            else:
                self.logger.info(
                    "external_context_dropped reason=unauthorized_speaker"
                    " speaker_person_id=%s",
                    speaker_id,
                )
        return kept

    async def filter_memories(
        self, memories, subject_ids_of: SubjectIdsOf
    ) -> tuple[list, list]:
        """Partition retrieved memories into ``(kept, dropped)``.

        ``subject_ids_of`` resolves one memory to its PERSON subject ids. A
        memory with no PERSON subject (e.g. PUBLIC world knowledge) is kept; a
        memory with any unauthorized PERSON subject is dropped whole.
        """
        kept: list = []
        dropped: list = []
        for m in memories:
            person_ids = await subject_ids_of(m)
            if not person_ids:
                kept.append(m)
                continue
            _, denied = await self.filter_person_ids(person_ids)
            if denied:
                self.logger.info(
                    "external_context_dropped reason=unauthorized_subject"
                    " denied=%s",
                    ",".join(denied),
                )
                dropped.append(m)
            else:
                kept.append(m)
        return kept, dropped
