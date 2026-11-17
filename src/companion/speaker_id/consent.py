"""Explicit authorisation for long-term voiceprint storage.

Speaker recognition and voiceprint storage are different decisions. A person
may let Ava recognise their voice for one conversation without granting
permission to keep a biometric template locally. This policy is the single
gate the engine consults before any durable write:

* existing profiles keep working, because their data was collected under the
  previous policy and the new code must not silently invalidate it;
* a *new* temporary identity is only created after ``arm_new_identity()``,
  which a human invokes explicitly (``/speaker enroll <PERSON_ID>``);
* new samples for any identity, including a newly enrolled one, need
  ``is_authorized()`` to be true;
* revoking an id stops all further learning without touching rows.

The policy is thread-safe because the engine reads it from its worker thread
while the REPL writes it on the asyncio loop.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from companion.speaker_id.store import SpeakerStore


class EnrollmentPolicy:
    """Thread-safe consent state for new identities and new samples."""

    def __init__(self, *, strict: bool = True) -> None:
        self._lock = threading.Lock()
        # ``strict=False`` is the standalone/demo path: no consent checks.
        # The formal REPL always constructs a strict policy.
        self._strict = bool(strict)
        self._authorized: set[str] = set()
        self._armed: Optional[str] = None  # person id awaiting one new voice
        self._loaded = False

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed is not None

    def load_from_store(self, store: SpeakerStore) -> None:
        """Load persisted grants and treat existing profiles as usable.

        Existing profiles are *not* new collection; they are data the user
        already has. They become authorized so their samples can still be
        consolidated, but no new profile can appear without an explicit arm.
        """
        grants = {
            str(row["speaker_id"])
            for row in store.enrollment_consents()
        }
        existing = {
            str(row["speaker_id"])
            for row in store.list_speakers()
            if int(row.get("samples") or 0) > 0
        }
        with self._lock:
            self._authorized.update(grants)
            self._authorized.update(existing)
            self._loaded = True

    def allow_new_identity(self) -> bool:
        if not self._strict:
            return True
        with self._lock:
            return self._armed is not None

    def consumption_note(self) -> Optional[str]:
        with self._lock:
            return self._armed

    def is_authorized(self, speaker_id: Optional[str]) -> bool:
        if not self._strict:
            return True
        sid = str(speaker_id or "").strip()
        if not sid:
            return False
        with self._lock:
            return sid in self._authorized

    def arm_new_identity(self, person_id: str) -> None:
        """Authorise exactly the next newly created voice identity.

        The caller must tell the human to speak now. The arm is consumed as
        soon as the engine creates the identity, so a second unknown voice in
        the same session does not inherit the grant.
        """
        pid = str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id is required")
        with self._lock:
            self._armed = pid

    def disarm(self) -> None:
        with self._lock:
            self._armed = None

    def note_created(self, speaker_id: str) -> None:
        """A new id was created under an armed consent; authorise it once."""
        sid = str(speaker_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._authorized.add(sid)
            self._armed = None

    def authorize(self, speaker_id: str, *, store: Any = None, actor: str = "local_confirmation") -> None:
        sid = str(speaker_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._authorized.add(sid)
        if store is not None:
            store.set_enrollment_consent(sid, granted_by=actor)

    def revoke(self, speaker_id: str, *, store: Any = None) -> None:
        sid = str(speaker_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._authorized.discard(sid)
        if store is not None:
            store.revoke_enrollment_consent(sid)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "strict": self._strict,
                "authorized": len(self._authorized),
                "armed": self._armed is not None,
                "loaded": self._loaded,
            }


__all__ = ["EnrollmentPolicy"]
