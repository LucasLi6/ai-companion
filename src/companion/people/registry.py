"""Person registry: identity, role and explicit consent.

A person's ``role`` (owner/friend/guest) is a descriptive label and is *not*
used as a privacy access rule. Access is governed solely by the explicit
:class:`Consent` flags, which are never defaulted to ``True`` in V0.0.
"""

from __future__ import annotations

from typing import Optional

from companion.domain.enums import PersonRole, RegistrationStatus
from companion.domain.models import Consent, Person, new_id, utcnow
from companion.storage.repositories import PersonRepository


class PersonRegistry:
    def __init__(self, repo: PersonRepository) -> None:
        self.repo = repo

    async def register(
        self,
        display_name: str,
        role: PersonRole | str = PersonRole.GUEST,
        *,
        allow_processing: bool = False,
        allow_persistence: bool = False,
        allow_memory_learning: bool = False,
        person_id: str | None = None,
    ) -> Person:
        """Register a person and their explicit consent in one atomic step."""
        pid = person_id or new_id("person")
        now = utcnow()
        person = Person(
            person_id=pid,
            display_name=display_name,
            role=PersonRole(role),
            created_at=now,
            updated_at=now,
        )
        consent = Consent(
            person_id=pid,
            allow_processing=allow_processing,
            allow_persistence=allow_persistence,
            allow_memory_learning=allow_memory_learning,
            updated_at=now,
        )
        await self.repo.register(person, consent)
        return person

    async def get(self, person_id: str) -> Optional[Person]:
        return await self.repo.get(person_id)

    async def get_consent(self, person_id: str) -> Optional[Consent]:
        return await self.repo.get_consent(person_id)

    async def set_consent(self, consent: Consent) -> None:
        await self.repo.set_consent(consent)

    async def set_status(
        self, person_id: str, status: RegistrationStatus
    ) -> None:
        """Suspend or revoke a person; this never deletes existing data."""
        await self.repo.set_registration_status(person_id, status)

    async def list(self) -> list[Person]:
        return await self.repo.list()

    async def is_authorized(self, person_id: str | None) -> bool:
        """Return True only for an ACTIVE registered person granted processing.

        Role is a descriptive label and does not bypass consent or status.
        """
        if person_id is None:
            return False
        person = await self.repo.get(person_id)
        if person is None or person.registration_status != RegistrationStatus.ACTIVE:
            return False
        consent = await self.repo.get_consent(person_id)
        return consent is not None and consent.allow_processing
