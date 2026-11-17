"""Explicit authorisation for long-term voiceprint storage.

Speaker recognition and voiceprint storage are different decisions. A person
may let Ava recognise their voice for one conversation without granting
permission to keep a biometric template locally. This policy is the single
gate the engine consults before any durable write:

* recognition always works: matching only reads profiles that already exist;
* a *new* temporary identity is only created after ``arm_new_identity()``,
  which a human invokes explicitly (``/speaker enroll <PERSON_ID>``);
* new samples for any identity, including a newly enrolled one, need
  ``is_authorized()`` to be true;
* only an explicit ``GRANTED`` row restores that authorisation after a
  restart. Profiles that merely have samples from an earlier version are
  ``LEGACY_READ_ONLY``: they can still be recognised, but they cannot keep
  learning until a human grants storage again;
* ``REVOKED`` is durable. It is never inferred from a missing row, and a
  restart cannot turn it back into a grant.

The policy is thread-safe because the engine reads it from its worker thread
while the REPL writes it on the asyncio loop.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from companion.speaker_id.store import (
    CONSENT_GRANTED,
    CONSENT_LEGACY_READ_ONLY,
    CONSENT_REVOKED,
    CONSENT_STATUSES,
    SpeakerStore,
)

logger = logging.getLogger(__name__)


class EnrollmentPolicy:
    """Thread-safe consent state for new identities and new samples."""

    def __init__(self, *, strict: bool = True) -> None:
        self._lock = threading.Lock()
        # ``strict=False`` is the standalone/demo path: no consent checks.
        # The formal REPL always constructs a strict policy.
        self._strict = bool(strict)
        self._authorized: set[str] = set()
        # Explicit durable state by speaker id. ``None`` means "no row".
        self._statuses: dict[str, str] = {}
        # Profiles that exist with samples but have no explicit row. They are
        # recognisable history, not a licence to keep collecting.
        self._legacy_profiles: set[str] = set()
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
        """Restore only explicitly recorded storage decisions.

        Sample count is deliberately *not* used as a grant: a person who
        revoked storage would otherwise be re-authorised by the very rows the
        revocation left in place. The store is read outside the lock; the
        result replaces the previous state atomically.
        """
        authorized: set[str] = set()
        statuses: dict[str, str] = {}
        for row in store.enrollment_consents():
            sid = str(row.get("speaker_id") or "").strip()
            if not sid:
                continue
            status = str(row.get("status") or "").strip().upper()
            if status not in CONSENT_STATUSES:
                # An unreadable/missing status is historical data at most.
                status = CONSENT_LEGACY_READ_ONLY
            statuses[sid] = status
            if status == CONSENT_GRANTED:
                authorized.add(sid)
        legacy = {
            str(row.get("speaker_id"))
            for row in store.list_speakers()
            if int(row.get("samples") or 0) > 0
        }
        with self._lock:
            self._authorized = authorized
            self._statuses = statuses
            self._legacy_profiles = legacy
            self._loaded = True

    def status_of(self, speaker_id: Optional[str]) -> Optional[str]:
        """Explicit consent status, or ``None`` when no record exists.

        A profile with samples and no consent row reports
        ``LEGACY_READ_ONLY`` so callers can display it honestly instead of
        treating the absence of a row as permission.
        """
        sid = str(speaker_id or "").strip()
        if not sid:
            return None
        with self._lock:
            status = self._statuses.get(sid)
            if status is not None:
                return status
            if sid in self._legacy_profiles:
                return CONSENT_LEGACY_READ_ONLY
            return None

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

    def note_created(
        self,
        speaker_id: str,
        *,
        store: Any = None,
        actor: str = "local_enrollment",
    ) -> None:
        """A new id was created under an armed consent; authorise it once.

        The grant is persisted when a store is supplied, so the identity
        created by an armed enrollment keeps working after a restart instead
        of silently becoming read-only. A store failure is logged and leaves
        only the in-memory grant; it never stops the caller.
        """
        sid = str(speaker_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._authorized.add(sid)
            self._statuses[sid] = CONSENT_GRANTED
            self._armed = None
        if store is not None:
            try:
                store.set_enrollment_consent(
                    sid,
                    granted_by=actor,
                    status=CONSENT_GRANTED,
                )
            except Exception:  # noqa: BLE001 - consent cache still applies
                logger.debug("speaker_consent_persist_failed speaker_id=%s", sid)

    def authorize(
        self,
        speaker_id: str,
        *,
        store: Any = None,
        actor: str = "local_confirmation",
    ) -> None:
        sid = str(speaker_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._authorized.add(sid)
            self._statuses[sid] = CONSENT_GRANTED
        if store is not None:
            store.set_enrollment_consent(
                sid, granted_by=actor, status=CONSENT_GRANTED
            )

    def revoke(self, speaker_id: str, *, store: Any = None) -> None:
        sid = str(speaker_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._authorized.discard(sid)
            self._statuses[sid] = CONSENT_REVOKED
        if store is not None:
            store.revoke_enrollment_consent(sid)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "strict": self._strict,
                "authorized": len(self._authorized),
                "armed": self._armed is not None,
                "loaded": self._loaded,
                "revoked": sum(
                    1
                    for status in self._statuses.values()
                    if status == CONSENT_REVOKED
                ),
            }


__all__ = ["EnrollmentPolicy"]
