"""Guided local voice registration (V0.9.9.31.1 §9).

``/speaker register <PERSON_ID>`` walks a human through one explicit local
enrollment instead of asking them to combine ``enroll``, ``promote`` and
``bind`` by hand:

    IDLE -> CHECKING_CONSENT -> COLLECTING -> QUALITY_CHECK
         -> READY_FOR_CONFIRMATION -> BOUND

with ``FAILED`` / ``CANCELLED`` / ``EXPIRED`` as terminal exits.

The flow never binds on its own: ``READY_FOR_CONFIRMATION`` only means "the
captured audio is usable". A separate, explicit ``confirm`` call performs the
human decision — promotion, the Speaker -> Person link and the storage grant.
Audio from Ava's own speech output never reaches the microphone pipeline, and
``observe`` also refuses any observation explicitly marked as TTS so a future
wiring mistake cannot enroll the assistant's voice.

This module keeps its state in memory. It writes the speaker store only at
confirm time (via promotion), the main database only through the link service,
and it never enables memory learning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from companion.domain.enums import RegistrationStatus
from companion.people.speaker_profile_link import (
    SpeakerLinkError,
    SpeakerProfileLinkService,
)
from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    ERROR,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    PENDING,
    QUALITY_RANK,
)
from companion.speaker_id.promotion import (
    ALREADY_KNOWN,
    NO_SAMPLES,
    PROMOTED,
    promote_voice_profile,
)

IDLE = "IDLE"
CHECKING_CONSENT = "CHECKING_CONSENT"
COLLECTING = "COLLECTING"
QUALITY_CHECK = "QUALITY_CHECK"
READY_FOR_CONFIRMATION = "READY_FOR_CONFIRMATION"
BOUND = "BOUND"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
EXPIRED = "EXPIRED"

TERMINAL_STATES = frozenset({BOUND, FAILED, CANCELLED, EXPIRED})
ACTIVE_STATES = frozenset(
    {CHECKING_CONSENT, COLLECTING, QUALITY_CHECK, READY_FOR_CONFIRMATION}
)


class SpeakerRegistrationError(Exception):
    """The flow was used out of order; ``code`` is a stable reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class RegistrationStep:
    """One observable state of the guided flow."""

    state: str
    message: str
    person_id: Optional[str] = None
    speaker_id: Optional[str] = None
    reason: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def can_confirm(self) -> bool:
        return self.state == READY_FOR_CONFIRMATION


class SpeakerRegistrationFlow:
    """In-memory state machine for one guided registration."""

    def __init__(
        self,
        *,
        store: Any,
        persons: Any,
        link: SpeakerProfileLinkService,
        enrollment_policy: Any = None,
        min_quality: str = "HIGH",
        min_voiced_seconds: float = 2.0,
        timeout_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.persons = persons
        self.link = link
        self.enrollment_policy = enrollment_policy
        self.min_quality = str(min_quality or "HIGH").upper()
        self.min_voiced_seconds = max(0.0, float(min_voiced_seconds))
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self._clock = clock
        self.state = IDLE
        self.person_id: Optional[str] = None
        self.speaker_id: Optional[str] = None
        self.reason: Optional[str] = None
        self.expires_at: Optional[float] = None
        self.observed_samples = 0
        self.best_voiced_seconds = 0.0
        self._seen_utterances: set[int] = set()

    # ------------------------------------------------------------- helpers
    def _step(self, message: str, **overrides: Any) -> RegistrationStep:
        values = {
            "state": self.state,
            "message": message,
            "person_id": self.person_id,
            "speaker_id": self.speaker_id,
            "reason": self.reason,
        }
        values.update(overrides)
        return RegistrationStep(**values)

    def _finish(self, state: str, message: str, *, reason: str) -> RegistrationStep:
        self.state = state
        self.reason = reason
        if state in TERMINAL_STATES and self.enrollment_policy is not None:
            # Never leave a stale "next voice may be stored" authorisation
            # behind: an abandoned flow must not enroll the next stranger.
            self.enrollment_policy.disarm()
        return self._step(message)

    def _expired(self, at: Optional[float]) -> bool:
        return (
            self.expires_at is not None
            and (self._clock() if at is None else float(at)) >= self.expires_at
        )

    def expire_if_due(self, *, at: Optional[float] = None) -> RegistrationStep:
        """Turn an abandoned flow into EXPIRED and disarm enrollment."""
        if self.state in ACTIVE_STATES and self._expired(at):
            return self._finish(
                EXPIRED,
                "registration expired: no usable voice captured in time",
                reason="registration_expired",
            )
        return self._step(self._message_for_state())

    def _message_for_state(self) -> str:
        if self.state == COLLECTING:
            return (
                "speak naturally for about two seconds;"
                " say /speaker cancel to stop"
            )
        if self.state == QUALITY_CHECK:
            return "audio quality is not sufficient yet; keep speaking"
        if self.state == READY_FOR_CONFIRMATION:
            return (
                "voice captured; run /speaker register confirm to bind it"
                " (or /speaker cancel)"
            )
        if self.state == BOUND:
            return "registration complete"
        if self.state == IDLE:
            return "no registration in progress"
        return "registration stopped"

    # --------------------------------------------------------------- start
    async def start(self, person_id: str) -> RegistrationStep:
        """Check consent, then arm exactly one new voice identity."""
        if self.state in ACTIVE_STATES:
            raise SpeakerRegistrationError("registration_already_active")
        pid = str(person_id or "").strip()
        if not pid:
            raise SpeakerRegistrationError("person_id_required")
        self.state = CHECKING_CONSENT
        self.person_id = pid
        self.speaker_id = None
        self.reason = None
        self.observed_samples = 0
        self.best_voiced_seconds = 0.0
        self._seen_utterances = set()

        if self.enrollment_policy is None:
            return self._finish(
                FAILED,
                "voiceprint storage policy is unavailable",
                reason="storage_policy_unavailable",
            )
        try:
            person = await self.persons.get(pid)
        except Exception:  # noqa: BLE001 - a DB failure is a refusal
            person = None
        if person is None:
            return self._finish(
                FAILED, f"person not found: {pid}", reason="person_missing"
            )
        if person.registration_status is not RegistrationStatus.ACTIVE:
            return self._finish(
                FAILED,
                f"person not active: {pid}"
                f" ({person.registration_status.value})",
                reason="person_inactive",
            )
        try:
            consent = await self.persons.get_consent(pid)
        except Exception:  # noqa: BLE001 - missing consent means no consent
            consent = None
        if (
            consent is None
            or not consent.allow_processing
            or not consent.allow_persistence
        ):
            # Learning and external-AI grants are not required to store the
            # voiceprint, but processing and persistence are.
            return self._finish(
                FAILED,
                "person consent does not allow processing and persistence;"
                " update consent before registering a voice",
                reason="consent_missing",
            )

        self.state = COLLECTING
        self.expires_at = self._clock() + self.timeout_seconds
        self.enrollment_policy.arm_new_identity(pid)
        name = str(getattr(person, "display_name", "") or "").strip()
        return self._step(
            f"registration started for {pid}"
            f"{f' ({name})' if name else ''}: speak now",
        )

    # ------------------------------------------------------------- observe
    def observe(
        self,
        result: Any,
        *,
        source: str = "microphone",
        at: Optional[float] = None,
        utterance_id: Optional[int] = None,
    ) -> RegistrationStep:
        """Feed one speaker result into the flow.

        Only microphone audio may enroll a voice. A result explicitly marked
        as TTS is ignored; it can neither satisfy quality nor create a
        candidate. Repeated revisions of one utterance count once.
        """
        if self.state in TERMINAL_STATES:
            raise SpeakerRegistrationError("registration_finished")
        if self.state not in (COLLECTING, QUALITY_CHECK, READY_FOR_CONFIRMATION):
            raise SpeakerRegistrationError("registration_not_collecting")
        expired = self.expire_if_due(at=at)
        if expired.state == EXPIRED:
            return expired
        if utterance_id is not None:
            uid = int(utterance_id)
            if uid in self._seen_utterances:
                return self._step(self._message_for_state())
            self._seen_utterances.add(uid)
        if str(source or "").strip().lower() != "microphone":
            return self._step(
                "ignored audio that did not come from the microphone",
                reason="non_microphone_audio",
            )
        if result is None:
            self.observed_samples += 1
            return self._step(self._message_for_state())

        self.observed_samples += 1
        status = str(getattr(result, "status", "") or "")
        speaker_id = str(getattr(result, "speaker_id", "") or "").strip()
        voiced = float(getattr(result, "voiced_seconds", 0.0) or 0.0)
        quality = str(getattr(result, "reid_quality", "") or "").upper()

        if status == KNOWN:
            # An existing stable identity already owns this voice. Enrolling a
            # second cluster would fork one person into two profiles.
            return self._finish(
                FAILED,
                "this voice already matches an existing identity;"
                " use /speaker link <SPEAKER_ID> <PERSON_ID> instead",
                reason="already_known_voice",
            )
        if status == ERROR:
            return self._step(
                "speaker model is unavailable; keeping the registration"
                " open (text input still works)",
                reason="speaker_model_error",
            )
        if status in (PENDING, INSUFFICIENT_AUDIO, AMBIGUOUS):
            return self._step(
                self._message_for_state(), reason="audio_not_usable_yet"
            )
        if status in (NEW_CANDIDATE, CANDIDATE_MATCH):
            if speaker_id:
                self.speaker_id = speaker_id
            self.best_voiced_seconds = max(self.best_voiced_seconds, voiced)
            enough_quality = QUALITY_RANK.get(quality, 0) >= QUALITY_RANK.get(
                self.min_quality, 0
            )
            enough_audio = voiced >= self.min_voiced_seconds
            if not (enough_quality and enough_audio):
                self.state = QUALITY_CHECK
                return self._step(
                    "audio quality is not sufficient yet; keep speaking",
                    reason="quality_too_low",
                )
            self.state = READY_FOR_CONFIRMATION
            return self._step(
                "voice captured; run /speaker register confirm to bind it"
                " (or /speaker cancel)",
                reason="ready_for_confirmation",
            )
        # Anything unrecognised is treated as unusable evidence.
        return self._step(
            self._message_for_state(), reason="unknown_speaker_status"
        )

    # ------------------------------------------------------------- confirm
    async def confirm(
        self,
        *,
        speaker_id: Optional[str] = None,
        actor: str = "local_registration",
    ) -> RegistrationStep:
        """Perform the human decision: promote, link, then grant storage."""
        if self.state in TERMINAL_STATES:
            raise SpeakerRegistrationError("registration_finished")
        if self.state != READY_FOR_CONFIRMATION:
            raise SpeakerRegistrationError("registration_not_ready")
        candidate = str(speaker_id or self.speaker_id or "").strip()
        if not candidate or self.person_id is None:
            return self._finish(
                FAILED,
                "no captured voice to confirm",
                reason="no_candidate",
            )

        final_speaker = candidate
        promotion = promote_voice_profile(self.store, candidate)
        if promotion.outcome == PROMOTED and promotion.speaker_id:
            final_speaker = promotion.speaker_id
        elif promotion.outcome == ALREADY_KNOWN and promotion.speaker_id:
            final_speaker = promotion.speaker_id
        elif promotion.outcome == NO_SAMPLES:
            return self._finish(
                FAILED,
                "the captured voice has no usable samples yet; speak again",
                reason="no_samples",
            )
        elif promotion.outcome not in (PROMOTED, ALREADY_KNOWN):
            # Linking a broken/unsupported id would create an identity that
            # cannot resolve; refuse instead of binding it anyway.
            return self._finish(
                FAILED,
                f"cannot promote {candidate} ({promotion.outcome})",
                reason="promotion_failed",
            )

        try:
            link = await self.link.confirm_link(
                final_speaker,
                self.person_id,
                verified_by="local_registration",
                actor=actor,
            )
        except SpeakerLinkError as exc:
            return self._finish(
                FAILED,
                f"link refused: {exc.code}",
                reason=exc.code,
            )

        if self.enrollment_policy is not None:
            # Only an explicit human confirmation may persist a GRANTED
            # voiceprint row. It is written after the binding exists so a
            # failed link leaves consent untouched. Any arm that was not
            # consumed by the engine is cleared: exactly one identity.
            self.enrollment_policy.authorize(
                final_speaker, store=self.store, actor=actor
            )
            self.enrollment_policy.disarm()
        self.speaker_id = final_speaker
        self.state = BOUND
        self.reason = None
        return self._step(
            f"registered speaker {final_speaker} ->"
            f" {self.person_id} ({link.binding.get('person_id')});"
            " voiceprint storage granted for this identity"
        )

    # --------------------------------------------------------------- exits
    def cancel(self, *, reason: str = "cancelled_by_human") -> RegistrationStep:
        if self.state in TERMINAL_STATES:
            return self._step(self._message_for_state())
        return self._finish(
            CANCELLED, "registration cancelled; nothing was bound", reason=reason
        )


__all__ = [
    "ACTIVE_STATES",
    "BOUND",
    "CANCELLED",
    "CHECKING_CONSENT",
    "COLLECTING",
    "EXPIRED",
    "FAILED",
    "IDLE",
    "QUALITY_CHECK",
    "READY_FOR_CONFIRMATION",
    "RegistrationStep",
    "SpeakerRegistrationError",
    "SpeakerRegistrationFlow",
    "TERMINAL_STATES",
]
