"""One conservative identity decision per accepted voice turn (V0.9.9.31).

The voice pipeline answers "which voice-print cluster does this audio sound
like?". It never answers "which person is this?". This module is the only
place that turns a :class:`SpeakerResult` into a conversation identity, and it
does so with four independently checked gates:

1. the match itself is decided, non-provisional and of sufficient quality;
2. the speaker id (or one of its aliases) has a human-created binding;
3. every related binding agrees on one person;
4. that person is ACTIVE and their current consent allows processing and
   persistence.

Anything else is ``UNKNOWN``. Similarity values are evidence used by the
engine's matcher; they are never treated as a probability that a binding is
correct, and they never create, move or repair a binding.

``CONFIRMED`` means "a locally confirmed person binding applies to this
utterance". It is *not* biometric authentication for a sensitive operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from companion.domain.enums import RegistrationStatus
from companion.people.speaker_bindings import SpeakerBindingService
from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    KNOWN,
    NEW_CANDIDATE,
    QUALITY_MEDIUM,
    QUALITY_RANK,
    SpeakerResult,
)
from companion.speaker_id.store import (
    ALIAS_REASON_BROKEN_CHAIN,
    ALIAS_REASON_UNKNOWN,
    AliasClosure,
)

# Stable reason codes. They are logged and may be asserted in tests; keep them
# free of free-form text and personal data.
REASON_TRUSTED_BINDING = "trusted_binding"
REASON_SPEAKER_UNBOUND = "speaker_unbound"
REASON_PROVISIONAL_MATCH = "provisional_match"
REASON_AMBIGUOUS_MATCH = "ambiguous_match"
REASON_MODEL_UNAVAILABLE = "speaker_model_unavailable"
REASON_BINDING_CONFLICT = "binding_conflict"
REASON_CONSENT_MISSING = "consent_missing"
REASON_PERSON_INACTIVE = "person_inactive"
REASON_ALIAS_RESOLUTION_FAILED = "alias_resolution_failed"

UNKNOWN_DISPLAY_NAME = "Unknown"


class IdentityState(str, Enum):
    CONFIRMED = "CONFIRMED"
    TENTATIVE = "TENTATIVE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ResolvedIdentity:
    """Immutable identity decision for one ``(run_id, utterance_id)``."""

    run_id: Optional[str]
    utterance_id: int
    raw_speaker_id: Optional[str]
    canonical_speaker_id: Optional[str]
    person_id: Optional[str]
    display_name: str
    state: IdentityState
    reason: str

    @property
    def confirmed(self) -> bool:
        return self.state is IdentityState.CONFIRMED


@dataclass(frozen=True)
class IdentityTrustPolicy:
    """Testable policy for "reliable enough to consider a binding".

    Thresholds are deliberately *not* duplicated here. The engine already
    decides KNOWN/CANDIDATE_MATCH from ``SpeakerConfig``; this policy only adds
    the requirements that a conversation identity needs beyond a raw match.
    """

    minimum_quality: str = QUALITY_MEDIUM
    require_known: bool = True
    require_unprovisional: bool = True

    def qualifies(self, result: Optional[SpeakerResult]) -> bool:
        if result is None:
            return False
        status = str(getattr(result, "status", "") or "")
        if self.require_known and status != KNOWN:
            return False
        if self.require_unprovisional and bool(
            getattr(result, "provisional", False)
        ):
            return False
        quality = str(getattr(result, "reid_quality", "") or "").upper()
        minimum = str(self.minimum_quality or "").upper()
        if QUALITY_RANK.get(quality, 0) < QUALITY_RANK.get(minimum, 0):
            return False
        # A decided match without both values is malformed evidence. Never
        # interpret a missing score as trustworthy. The one legitimate
        # exception is a singleton match: when no second identity exists,
        # there is no runner-up to compare against, so ``margin`` is absent by
        # construction — but only the matcher may claim that, via the explicit
        # ``singleton_match`` flag.
        similarity = getattr(result, "similarity", None)
        if similarity is None:
            return False
        try:
            if not math.isfinite(float(similarity)):
                return False
        except (TypeError, ValueError):
            return False
        singleton = bool(getattr(result, "singleton_match", False))
        margin = getattr(result, "margin", None)
        if margin is None:
            return singleton
        try:
            margin_value = float(margin)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(margin_value) or margin_value < 0.0:
            return False
        if singleton:
            # A "no runner-up" claim alongside a numeric margin contradicts
            # itself; refuse rather than pick the convenient half.
            return False
        return True


class IdentityResolver:
    """Resolve one speaker snapshot to CONFIRMED/TENTATIVE/UNKNOWN."""

    def __init__(
        self,
        persons: Any,
        bindings: SpeakerBindingService,
        speaker_store: Any = None,
        *,
        policy: Optional[IdentityTrustPolicy] = None,
    ) -> None:
        self.persons = persons
        self.bindings = bindings
        self.speaker_store = speaker_store
        self.policy = policy or IdentityTrustPolicy()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _tentative_display(raw: Optional[str], canonical: Optional[str]) -> str:
        speaker_id = str(canonical or raw or "").strip()
        if not speaker_id:
            return UNKNOWN_DISPLAY_NAME
        return f"speaker {speaker_id}"

    def _alias_closure(self, speaker_id: str) -> AliasClosure:
        """Canonical id plus related aliases, with an explicit completeness bit."""
        store = self.speaker_store
        if store is None:
            return AliasClosure(speaker_id, (speaker_id,), True, None)
        resolver = getattr(store, "resolve_alias_closure", None)
        if resolver is None:
            # A store that cannot prove the alias set is complete must not be
            # used for a person decision. Never degrade to checking one id.
            return AliasClosure(
                None, (), False, ALIAS_REASON_BROKEN_CHAIN
            )
        return resolver(speaker_id)

    def _store_ids(self, speaker_id: str) -> tuple[Optional[str], list[str]]:
        """Return ``(canonical, related_ids)``; raises only on store errors."""
        closure = self._alias_closure(speaker_id)
        if closure.canonical is None:
            return None, []
        return closure.canonical, list(closure.speaker_ids)

    async def resolve(
        self,
        result: Optional[SpeakerResult],
        *,
        run_id: Optional[str],
        utterance_id: int,
    ) -> ResolvedIdentity:
        """Resolve one snapshot. Never defaults to the Owner/startup person."""
        uid = int(utterance_id)
        if result is None:
            return self._unknown(
                run_id, uid, None, None, REASON_MODEL_UNAVAILABLE
            )
        status = str(getattr(result, "status", "") or "")
        raw = str(getattr(result, "speaker_id", "") or "").strip() or None

        if status == AMBIGUOUS:
            return self._unknown(
                run_id, uid, raw, None, REASON_AMBIGUOUS_MATCH
            )
        if status == NEW_CANDIDATE:
            # A temporary cluster is displayable but is not a person.
            canonical = raw
            if raw is not None:
                try:
                    canonical = self._store_ids(raw)[0] or raw
                except Exception:  # noqa: BLE001 - display only
                    canonical = raw
            return ResolvedIdentity(
                run_id=run_id,
                utterance_id=uid,
                raw_speaker_id=raw,
                canonical_speaker_id=canonical,
                person_id=None,
                display_name=self._tentative_display(raw, canonical),
                state=IdentityState.TENTATIVE,
                reason=REASON_SPEAKER_UNBOUND,
            )
        if status == CANDIDATE_MATCH:
            canonical = raw
            if raw is not None:
                try:
                    canonical = self._store_ids(raw)[0] or raw
                except Exception:  # noqa: BLE001 - display only
                    canonical = raw
            return ResolvedIdentity(
                run_id=run_id,
                utterance_id=uid,
                raw_speaker_id=raw,
                canonical_speaker_id=canonical,
                person_id=None,
                display_name=self._tentative_display(raw, canonical),
                state=IdentityState.TENTATIVE,
                reason=REASON_PROVISIONAL_MATCH,
            )
        if status != KNOWN:
            # PENDING / ERROR / INSUFFICIENT_AUDIO / anything unrecognised.
            return self._unknown(
                run_id, uid, raw, None, REASON_MODEL_UNAVAILABLE
            )
        if not self.policy.qualifies(result):
            return self._unknown(
                run_id, uid, raw, None, REASON_PROVISIONAL_MATCH
            )
        if raw is None:
            return self._unknown(
                run_id, uid, None, None, REASON_SPEAKER_UNBOUND
            )

        try:
            closure = self._alias_closure(raw)
        except Exception:  # noqa: BLE001 - store failure is never identity
            return self._unknown(
                run_id, uid, raw, None, REASON_ALIAS_RESOLUTION_FAILED
            )
        if closure.canonical is None:
            # A deleted or never-existing speaker id has no identity to bind.
            # A broken/cyclic chain is an alias-resolution failure, not "this
            # voice is simply unbound".
            reason = (
                REASON_SPEAKER_UNBOUND
                if closure.reason == ALIAS_REASON_UNKNOWN
                else REASON_ALIAS_RESOLUTION_FAILED
            )
            return self._unknown(run_id, uid, raw, None, reason)
        if not closure.complete:
            # A truncated traversal could miss a binding on an alias, so
            # checking only the ids we found is not safe.
            return self._unknown(
                run_id,
                uid,
                raw,
                closure.canonical,
                REASON_ALIAS_RESOLUTION_FAILED,
            )
        canonical = closure.canonical
        related = [canonical] + [
            sid for sid in closure.speaker_ids if sid != canonical
        ]

        try:
            rows = await self.bindings.bindings_for_speaker_ids(related)
        except Exception:  # noqa: BLE001 - a DB failure means no identity
            return self._unknown(
                run_id, uid, raw, canonical, REASON_MODEL_UNAVAILABLE
            )
        person_ids = {
            str(row.get("person_id") or "").strip()
            for row in rows
            if str(row.get("person_id") or "").strip()
        }
        if not person_ids:
            return self._unknown(
                run_id, uid, raw, canonical, REASON_SPEAKER_UNBOUND
            )
        if len(person_ids) > 1:
            # Two voices merged into one cluster does not merge two people.
            # A human must resolve this; never pick one silently.
            return self._unknown(
                run_id, uid, raw, canonical, REASON_BINDING_CONFLICT
            )
        person_id = next(iter(person_ids))
        try:
            person = await self.persons.get(person_id)
        except Exception:  # noqa: BLE001
            return self._unknown(
                run_id, uid, raw, canonical, REASON_MODEL_UNAVAILABLE
            )
        if person is None or person.registration_status is not (
            RegistrationStatus.ACTIVE
        ):
            return self._unknown(
                run_id, uid, raw, canonical, REASON_PERSON_INACTIVE
            )
        try:
            consent = await self.persons.get_consent(person_id)
        except Exception:  # noqa: BLE001
            return self._unknown(
                run_id, uid, raw, canonical, REASON_CONSENT_MISSING
            )
        if (
            consent is None
            or not consent.allow_processing
            or not consent.allow_persistence
        ):
            return self._unknown(
                run_id, uid, raw, canonical, REASON_CONSENT_MISSING
            )

        name = str(getattr(person, "display_name", "") or "").strip()
        return ResolvedIdentity(
            run_id=run_id,
            utterance_id=uid,
            raw_speaker_id=raw,
            canonical_speaker_id=canonical,
            person_id=person_id,
            display_name=name or person_id,
            state=IdentityState.CONFIRMED,
            reason=REASON_TRUSTED_BINDING,
        )

    # ----------------------------------------------------------------- output
    def _unknown(
        self,
        run_id: Optional[str],
        utterance_id: int,
        raw: Optional[str],
        canonical: Optional[str],
        reason: str,
    ) -> ResolvedIdentity:
        return ResolvedIdentity(
            run_id=run_id,
            utterance_id=utterance_id,
            raw_speaker_id=raw,
            canonical_speaker_id=canonical,
            person_id=None,
            display_name=UNKNOWN_DISPLAY_NAME,
            state=IdentityState.UNKNOWN,
            reason=reason,
        )


__all__ = [
    "IdentityResolver",
    "IdentityState",
    "IdentityTrustPolicy",
    "REASON_ALIAS_RESOLUTION_FAILED",
    "REASON_AMBIGUOUS_MATCH",
    "REASON_BINDING_CONFLICT",
    "REASON_CONSENT_MISSING",
    "REASON_MODEL_UNAVAILABLE",
    "REASON_PERSON_INACTIVE",
    "REASON_PROVISIONAL_MATCH",
    "REASON_SPEAKER_UNBOUND",
    "REASON_TRUSTED_BINDING",
    "ResolvedIdentity",
    "UNKNOWN_DISPLAY_NAME",
]
