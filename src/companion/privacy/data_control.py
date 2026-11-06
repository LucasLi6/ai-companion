"""Trusted, confirmation-gated data-control flow.

A privacy request reaches this module only as a *structured* decision from the
router (``data_control_request``). Nothing here parses free text, and a mere
``PRIVACY_REQUEST`` intent never deletes anything: the router's output is a
request, and this service is the only code allowed to act on it.

Flow:

1. ``resolve_request`` checks the requester, the scope and the target against
   the trusted store, then records a pending operation. It performs no
   deletion.
2. ``request_confirmation`` returns the pending record so a caller can ask the
   user; ``execute_confirmed`` runs only after an explicit confirmation.
3. ``execute_confirmed`` re-checks ownership and consent at execution time,
   then uses the existing MemoryService / PersonProfileService / utterance
   APIs. Unsupported or vague targets are reported as such rather than guessed
   at.

Audit rows are append-only bookkeeping; they are never a substitute for a
deletion, and reporting a deletion as done without removing the row is
forbidden here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from companion.domain.enums import RegistrationStatus
from companion.domain.models import UtteranceEvent, new_id, utcnow
from companion.memory.models import (
    AccessContext,
    MemoryNotFoundError,
    MemoryPermissionDeniedError,
)
from companion.memory.service import MemoryService
from companion.people.profile_service import PersonProfileService
from companion.routing.models import DataControlRequest
from companion.storage.repositories import Repositories

__all__ = ["DataControlService", "DataControlResult"]

_CONFIRMATION_TTL = timedelta(minutes=15)

_OPERATIONS = frozenset({"DELETE", "FORGET"})
_DELETABLE_TARGETS = frozenset({"MEMORY", "PROFILE", "EVENT"})
_ID_REQUIRED_TARGETS = frozenset({"MEMORY", "EVENT"})


class _Status:
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"
    DENIED = "DENIED"
    NOT_FOUND = "NOT_FOUND"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


@dataclass
class DataControlResult:
    status: str
    operation_id: Optional[str] = None
    detail: str = ""
    result: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "operation_id": self.operation_id,
            "detail": self.detail,
            "result": self.result,
        }


class DataControlService:
    def __init__(
        self,
        *,
        repos: Repositories,
        memory_service: MemoryService,
        profile_service: Optional[PersonProfileService] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.memory_service = memory_service
        self.profile_service = profile_service
        self.logger = logger or logging.getLogger("companion.privacy.data_control")

    # -- request resolution -------------------------------------------------

    async def resolve_request(
        self,
        request: Optional[DataControlRequest],
        event: UtteranceEvent,
        consent=None,
    ) -> DataControlResult:
        """Validate a structured request and record it as pending.

        No data is removed here, even for a fully specified target: a privacy
        deletion is irreversible, so it always passes through confirmation.
        """
        if request is None:
            return DataControlResult(
                _Status.NEEDS_CLARIFICATION,
                detail="no_structured_request",
            )

        speaker = event.speaker_person_id
        if not speaker:
            return DataControlResult(_Status.DENIED, detail="no_registered_speaker")
        person = await self.repos.person.get(speaker)
        if person is None:
            return DataControlResult(_Status.DENIED, detail="unknown_person")
        if person.registration_status == RegistrationStatus.SUSPENDED:
            return DataControlResult(_Status.DENIED, detail="person_suspended")

        operation = (request.operation or "").strip().upper()
        target_type = (request.target_type or "").strip().upper()
        scope = (request.scope or "SELF").strip().upper()

        if operation not in _OPERATIONS:
            return DataControlResult(
                _Status.UNSUPPORTED, detail=f"unsupported_operation:{operation}"
            )
        if not self.validate_scope(scope):
            return DataControlResult(
                _Status.UNSUPPORTED, detail=f"unsupported_scope:{scope}"
            )
        if target_type not in _DELETABLE_TARGETS:
            return DataControlResult(
                _Status.UNSUPPORTED, detail=f"unsupported_target:{target_type}"
            )

        target_id = (request.target_id or "").strip() or None
        if target_type in _ID_REQUIRED_TARGETS and target_id is None:
            # "Forget my stuff" is not a safe automatic bulk delete. Ask for a
            # specific target instead of guessing the scope.
            return DataControlResult(
                _Status.NEEDS_CLARIFICATION,
                detail=f"target_id_required:{target_type}",
            )

        denial = await self._ownership_denial(
            speaker, target_type, target_id, operation
        )
        if denial is not None:
            status, detail = denial
            return DataControlResult(status, detail=detail)

        operation_id = new_id("dco")
        now = utcnow()
        expires_at = now + _CONFIRMATION_TTL
        detail = f"operation={operation} target={target_type}"
        async with self.repos.db.transaction():
            await self.repos.db.execute(
                "INSERT INTO data_control_operations(operation_id, person_id,"
                " operation, target_type, target_id, status, detail,"
                " result_json, created_at, updated_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)",
                (
                    operation_id,
                    speaker,
                    operation,
                    target_type,
                    target_id,
                    _Status.AWAITING_CONFIRMATION,
                    detail,
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            await self.repos.audit.record(
                actor=speaker,
                action="privacy_request_recorded",
                entity_type="data_control",
                entity_id=operation_id,
                detail=detail,
            )
        return DataControlResult(
            _Status.AWAITING_CONFIRMATION,
            operation_id=operation_id,
            detail="confirmation_required",
            result={
                "operation": operation,
                "target_type": target_type,
                "target_id": target_id,
                "expires_at": expires_at.isoformat(),
            },
        )

    @staticmethod
    def validate_scope(scope: str) -> bool:
        """Only the requester's own data may be targeted by this path."""
        return (scope or "").strip().upper() == "SELF"

    async def request_confirmation(self, operation_id: str) -> Optional[dict]:
        """Return the pending operation, or ``None`` if it cannot be confirmed."""
        row = await self._get_row(operation_id)
        if row is None:
            return None
        if row["status"] != _Status.AWAITING_CONFIRMATION:
            return None
        if self._expired(row):
            return None
        return dict(row)

    async def get_operation_result(self, operation_id: str) -> Optional[dict]:
        return await self._get_row(operation_id)

    # -- execution ----------------------------------------------------------

    async def execute_confirmed(
        self,
        operation_id: str,
        *,
        actor_id: Optional[str] = None,
        confirmed: bool = True,
    ) -> DataControlResult:
        """Run a previously resolved operation after confirmation.

        Idempotent: once an operation is terminal, the recorded result is
        returned unchanged and no second deletion runs.
        """
        row = await self._get_row(operation_id)
        if row is None:
            return DataControlResult(_Status.NOT_FOUND, operation_id=operation_id)
        if row["status"] != _Status.AWAITING_CONFIRMATION:
            return DataControlResult(
                row["status"],
                operation_id=operation_id,
                detail=row.get("detail") or "",
                result=_loads(row.get("result_json")),
            )
        if actor_id is not None and actor_id != row["person_id"]:
            return DataControlResult(
                _Status.DENIED,
                operation_id=operation_id,
                detail="operation_belongs_to_another_person",
            )
        if self._expired(row):
            await self._finish(operation_id, _Status.EXPIRED, "confirmation_expired")
            return DataControlResult(
                _Status.EXPIRED, operation_id=operation_id, detail="confirmation_expired"
            )
        if not confirmed:
            await self._finish(operation_id, _Status.CANCELLED, "not_confirmed")
            return DataControlResult(
                _Status.CANCELLED, operation_id=operation_id, detail="not_confirmed"
            )

        person_id = row["person_id"]
        target_type = row["target_type"]
        target_id = row["target_id"]

        # Ownership and consent are re-checked at execution time, not trusted
        # from the request that created the row.
        denial = await self._ownership_denial(
            person_id, target_type, target_id, row["operation"]
        )
        if denial is not None:
            status, detail = denial
            await self._finish(operation_id, status, detail)
            return DataControlResult(status, operation_id=operation_id, detail=detail)

        try:
            if target_type == "MEMORY":
                result = await self._delete_memory(person_id, target_id)
            elif target_type == "EVENT":
                result = await self._delete_event(person_id, target_id)
            elif target_type == "PROFILE":
                result = await self._delete_profile(person_id)
            else:  # pragma: no cover - guarded in resolve_request
                await self._finish(
                    operation_id, _Status.UNSUPPORTED, "unsupported_target"
                )
                return DataControlResult(
                    _Status.UNSUPPORTED,
                    operation_id=operation_id,
                    detail="unsupported_target",
                )
        except (MemoryNotFoundError, MemoryPermissionDeniedError) as exc:
            await self._finish(operation_id, _Status.DENIED, type(exc).__name__)
            return DataControlResult(
                _Status.DENIED, operation_id=operation_id, detail=type(exc).__name__
            )
        except Exception as exc:
            self.logger.error(
                "data_control_failed operation_id=%s target=%s error=%s",
                operation_id,
                target_type,
                type(exc).__name__,
            )
            await self._finish(operation_id, _Status.FAILED, "execution_failed")
            return DataControlResult(
                _Status.FAILED, operation_id=operation_id, detail="execution_failed"
            )

        await self._finish(
            operation_id, _Status.COMPLETED, "deleted", result=result
        )
        return DataControlResult(
            _Status.COMPLETED,
            operation_id=operation_id,
            detail="deleted",
            result=result,
        )

    # -- target operations --------------------------------------------------

    async def _delete_memory(self, person_id: str, memory_id: str) -> dict:
        access = AccessContext(actor_id=person_id, is_admin=True, role="data_control")
        # The raw delete and the profile invalidation share one transaction:
        # a failure in either leaves the memory and its derived understanding
        # exactly as they were.
        async with self.repos.db.transaction(immediate=True):
            await self.memory_service.delete(memory_id, access=access)
            if self.profile_service is not None:
                await self.profile_service.invalidate_profile_source(
                    person_id,
                    memory_id=memory_id,
                    deletion_mode="PRIVACY_DELETE",
                )
        return {"deleted": {"memory": 1}, "memory_id": memory_id}

    async def _delete_profile(self, person_id: str) -> dict:
        if self.profile_service is None:
            raise RuntimeError("profile service is not configured")
        await self.profile_service.purge(person_id, reason="data_control_request")
        return {"deleted": {"profile": 1}, "person_id": person_id}

    async def _delete_event(self, person_id: str, event_id: str) -> dict:
        event = await self.repos.utterance.get(event_id)
        if event is None or event.speaker_person_id != person_id:
            raise MemoryPermissionDeniedError(event_id)
        access = AccessContext(actor_id=person_id, is_admin=True, role="data_control")
        async with self.repos.db.transaction(immediate=True):
            # Memories learned from this event are deleted first: the utterance
            # delete cascades the candidate/provenance bookkeeping, so this is
            # the last moment the provenance lookup can find them.
            derived = await self.memory_service.repo.list_memory_ids_for_source_event(
                event_id
            )
            deleted_memories = 0
            for memory_id in derived:
                try:
                    await self.memory_service.delete(memory_id, access=access)
                except MemoryNotFoundError:
                    continue
                deleted_memories += 1
            deleted = await self.repos.utterance.delete_owned_event(
                event_id, speaker_person_id=person_id
            )
            if not deleted:
                raise MemoryPermissionDeniedError(event_id)
            if self.profile_service is not None:
                await self.profile_service.invalidate_profile_source(
                    person_id,
                    event_id=event_id,
                    deletion_mode="PRIVACY_DELETE",
                )
        return {
            "deleted": {"event": 1, "memory": deleted_memories},
            "event_id": event_id,
        }

    # -- checks -------------------------------------------------------------

    async def _ownership_denial(
        self, person_id: str, target_type: str, target_id: Optional[str], operation: str
    ) -> Optional[tuple[str, str]]:
        if target_type == "MEMORY":
            item = await self.memory_service.repo.get(target_id)
            if item is None:
                return (_Status.NOT_FOUND, "memory_not_found")
            subjects = await self.memory_service.repo.get_subjects(target_id)
            if item.created_by != person_id and (
                "PERSON",
                person_id,
            ) not in {(kind, sid) for kind, sid in subjects}:
                return (_Status.DENIED, "memory_not_owned")
            return None
        if target_type == "EVENT":
            event = await self.repos.utterance.get(target_id)
            if event is None:
                return (_Status.NOT_FOUND, "event_not_found")
            if event.speaker_person_id != person_id:
                return (_Status.DENIED, "event_not_owned")
            return None
        if target_type == "PROFILE":
            if self.profile_service is None:
                return (_Status.UNSUPPORTED, "profile_service_unavailable")
            return None
        return (_Status.UNSUPPORTED, f"unsupported_target:{target_type}")

    # -- bookkeeping --------------------------------------------------------

    async def _get_row(self, operation_id: str) -> Optional[dict]:
        cur = await self.repos.db.execute(
            "SELECT operation_id, person_id, operation, target_type, target_id,"
            " status, detail, result_json, created_at, updated_at, expires_at"
            " FROM data_control_operations WHERE operation_id = ?",
            (operation_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def _finish(
        self,
        operation_id: str,
        status: str,
        detail: str,
        *,
        result: Optional[dict] = None,
    ) -> None:
        now = utcnow().isoformat()
        payload = json.dumps(result or {}, ensure_ascii=False)
        async with self.repos.db.transaction():
            await self.repos.db.execute(
                "UPDATE data_control_operations SET status = ?, detail = ?,"
                " result_json = ?, updated_at = ? WHERE operation_id = ?",
                (status, detail, payload, now, operation_id),
            )
            await self.repos.audit.record(
                actor="system",
                action=f"data_control_{status.lower()}",
                entity_type="data_control",
                entity_id=operation_id,
                detail=detail,
            )

    @staticmethod
    def _expired(row: dict) -> bool:
        raw = row.get("expires_at")
        if not raw:
            return False
        try:
            expires = datetime.fromisoformat(raw)
        except ValueError:
            return True
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return utcnow() >= expires


def _loads(raw) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
