"""Explicit person <-> voice-print bindings (V0.9.9.24).

A voice print answers "does this audio sound like speaker S?", never "is this
person P?". The only bridge between the two is a binding a human created
through an explicit confirmation or a trusted local registration. Similarity
scores are never allowed to create, move or resolve a binding.

The speaker embeddings live in a separate SQLite store; this table holds only
the trusted link. Resolution re-checks the person's current registration
status and consent every time, so revoking a person or withdrawing
``allow_processing``/``allow_persistence`` immediately stops identity
resolution without touching the speaker store.

Voice matching alone never authenticates a sensitive operation. Callers must
keep treating a resolved ``person_id`` as a conversation identity hint, not as
proof of identity.
"""

from __future__ import annotations

from typing import Any, Optional

from companion.domain.enums import RegistrationStatus
from companion.domain.models import utcnow
from companion.speaker_id.models import (
    AMBIGUOUS,
    CANDIDATE_MATCH,
    ERROR,
    INSUFFICIENT_AUDIO,
    KNOWN,
    NEW_CANDIDATE,
    PENDING,
)
from companion.storage.database import Database
from companion.storage.repositories import PersonRepository

# A bound identity may only be resolved from a decided, non-provisional
# match. ``NEW_CANDIDATE``/``AMBIGUOUS``/``PENDING`` are explicitly excluded:
# a new cluster and an ambiguous window are not an identity.
RESOLVABLE_STATUSES = frozenset({KNOWN, CANDIDATE_MATCH})
_NON_IDENTITIES = frozenset(
    {PENDING, NEW_CANDIDATE, AMBIGUOUS, INSUFFICIENT_AUDIO, ERROR}
)


class SpeakerBindingError(Exception):
    """A binding request was refused; ``code`` is a stable reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SpeakerBindingService:
    def __init__(self, db: Database, persons: PersonRepository) -> None:
        self.db = db
        self.persons = persons

    async def bind_speaker(
        self,
        person_id: str,
        speaker_id: str,
        verified_by: str,
    ) -> dict:
        """Record a human-confirmed speaker -> person binding.

        Re-binding the same speaker to the same person is idempotent. Moving a
        binding to another person is refused: it needs an explicit unbind (and
        the human decision that goes with it), never a silent overwrite.
        """
        person_id = str(person_id or "").strip()
        speaker_id = str(speaker_id or "").strip()
        verifier = str(verified_by or "").strip()
        if not person_id or not speaker_id or not verifier:
            raise SpeakerBindingError("binding_requires_person_speaker_and_verifier")
        person = await self.persons.get(person_id)
        if person is None or person.registration_status is not RegistrationStatus.ACTIVE:
            raise SpeakerBindingError("person_not_active")
        existing = await self.get_binding(speaker_id)
        now = utcnow().isoformat()
        if existing is not None:
            if existing["person_id"] != person_id:
                raise SpeakerBindingError("speaker_already_bound")
            await self.db.execute(
                "UPDATE person_speaker_bindings SET verified_by = ?,"
                " updated_at = ? WHERE speaker_id = ?",
                (verifier, now, speaker_id),
            )
            return await self.get_binding(speaker_id)  # type: ignore[return-value]
        await self.db.execute(
            "INSERT INTO person_speaker_bindings("
            " speaker_id, person_id, verified_by, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (speaker_id, person_id, verifier, now, now),
        )
        return await self.get_binding(speaker_id)  # type: ignore[return-value]

    async def unbind_speaker(self, speaker_id: str) -> bool:
        speaker_id = str(speaker_id or "").strip()
        if not speaker_id:
            return False
        cur = await self.db.execute(
            "DELETE FROM person_speaker_bindings WHERE speaker_id = ?",
            (speaker_id,),
        )
        return bool(cur.rowcount)

    async def get_binding(self, speaker_id: str) -> Optional[dict]:
        cur = await self.db.execute(
            "SELECT speaker_id, person_id, verified_by, created_at, updated_at"
            " FROM person_speaker_bindings WHERE speaker_id = ?",
            (str(speaker_id or "").strip(),),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_bindings(self, person_id: Optional[str] = None) -> list[dict]:
        sql = (
            "SELECT speaker_id, person_id, verified_by, created_at, updated_at"
            " FROM person_speaker_bindings"
        )
        params: tuple[Any, ...] = ()
        if person_id is not None:
            sql += " WHERE person_id = ?"
            params = (person_id,)
        sql += " ORDER BY created_at ASC, speaker_id ASC"
        cur = await self.db.execute(sql, params)
        return [dict(row) for row in await cur.fetchall()]

    async def resolve_speaker(self, speaker_result: Any) -> Optional[str]:
        """Return the bound person for a reliable match, or ``None``.

        ``None`` is the safe answer for every uncertain or unauthorized case:
        pending/ambiguous/new candidates, provisional samples, unknown speaker
        ids, missing bindings, a non-ACTIVE person, or consent that does not
        allow processing and persistence.
        """
        if speaker_result is None:
            return None
        status = str(getattr(speaker_result, "status", "") or "")
        if status in _NON_IDENTITIES or status not in RESOLVABLE_STATUSES:
            return None
        if bool(getattr(speaker_result, "provisional", False)):
            return None
        speaker_id = str(getattr(speaker_result, "speaker_id", "") or "").strip()
        if not speaker_id:
            return None
        binding = await self.get_binding(speaker_id)
        if binding is None:
            return None
        person_id = binding["person_id"]
        person = await self.persons.get(person_id)
        if person is None or person.registration_status is not RegistrationStatus.ACTIVE:
            return None
        consent = await self.persons.get_consent(person_id)
        if (
            consent is None
            or not consent.allow_processing
            or not consent.allow_persistence
        ):
            return None
        return person_id


__all__ = [
    "RESOLVABLE_STATUSES",
    "SpeakerBindingError",
    "SpeakerBindingService",
]
