"""The unified memory service: the single, policy-enforcing gateway.

All memory reads and writes go through here. It enforces, in order:

- **Requester identity** — an :class:`AccessContext` supplied by the trusted
  application (never a value returned by an LLM). Non-admin requesters only see
  public items or items where they are a PERSON subject.
- **Subject privacy** — creating personal/private items about a person requires
  that person to be currently authorised for learning (checked via the injected
  ``authorize_subject`` callback).
- **Optimistic locking** — updates must present the current ``version``; a stale
  version raises :class:`MemoryVersionConflictError`.
- **Human-confirmed priority** — AI proposals (:meth:`propose`) never overwrite
  a human-confirmed fact; they create a ``NEEDS_REVIEW`` record that a human
  must approve. Only :meth:`update` (admin/creator) mutates a live record.
- **Retract vs delete** — :meth:`retract` marks an item ``RETRACTED`` (kept, out
  of default search); :meth:`delete` removes the record, its private content and
  its index entries for good.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from companion.domain.enums import (
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
    RegistrationStatus,
)
from companion.domain.models import new_id, utcnow

from .models import (
    AccessContext,
    MemoryEvidence,
    MemoryItem,
    MemoryNotFoundError,
    MemoryPermissionDeniedError,
    MemoryRevision,
    MemorySource,
    MemoryVersionConflictError,
)
from .repository import MemoryRepository

if TYPE_CHECKING:  # pragma: no cover - typing only
    from companion.storage.repositories import AuditRepository, Repositories

AuthorizeSubject = Callable[[str], Awaitable[bool]]
PersonExists = Callable[[str], Awaitable[bool]]


class MemoryService:
    def __init__(
        self,
        repo: MemoryRepository,
        *,
        authorize_subject: Optional[AuthorizeSubject] = None,
        person_exists: Optional[PersonExists] = None,
        audit: Optional["AuditRepository"] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repo = repo
        self.authorize_subject = authorize_subject
        self.person_exists = person_exists
        self.audit = audit
        self.logger = logger or logging.getLogger("companion.memory")

    async def _audit(
        self, access: AccessContext, action: str, entity_id: str, detail: str | None = None
    ) -> None:
        if self.audit is None:
            return
        await self.audit.record(
            actor=access.actor_id,
            action=action,
            entity_type="memory",
            entity_id=entity_id,
            detail=detail,
        )

    # -- create ------------------------------------------------------------

    async def create(
        self,
        *,
        kind: MemoryKind | str,
        content: str,
        access: AccessContext,
        status: MemoryStatus | str = MemoryStatus.ACTIVE,
        visibility: MemoryVisibility | str = MemoryVisibility.PRIVATE,
        created_by: str | None = None,
        subjects: list[tuple[str, str]] | None = None,
        sources: list[MemorySource] | None = None,
        evidence: list[MemoryEvidence] | None = None,
        human_confirmed: bool = False,
        source_supported: bool = False,
        self_reported: bool = False,
        third_party_reported: bool = False,
        model_inferred: bool = False,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        memory_id: str | None = None,
    ) -> MemoryItem:
        subjects = subjects or []
        sources = sources or []
        evidence = evidence or []
        kind = MemoryKind(kind)
        status = MemoryStatus(status)
        visibility = MemoryVisibility(visibility)

        # Subject privacy: a non-admin must not write personal records about a
        # person who is not currently authorised for learning.
        person_subjects = [sid for (st, sid) in subjects if st == "PERSON"]
        if person_subjects and not access.is_admin:
            await self._require_subjects_authorized(person_subjects)

        now = utcnow()
        item = MemoryItem(
            memory_id=memory_id or new_id("mem"),
            kind=kind,
            content=content,
            status=status,
            visibility=visibility,
            created_by=created_by or access.actor_id,
            created_at=now,
            updated_at=now,
            version=1,
            valid_from=valid_from,
            valid_until=valid_until,
            human_confirmed=human_confirmed,
            source_supported=source_supported,
            self_reported=self_reported,
            third_party_reported=third_party_reported,
            model_inferred=model_inferred,
        )
        async with self.repo.db.transaction():
            await self.repo.create(item, subjects, sources, evidence)
            await self.repo.insert_revision(
                MemoryRevision(
                    revision_id=new_id("rev"),
                    memory_id=item.memory_id,
                    version=1,
                    kind=item.kind,
                    content=item.content,
                    status=item.status,
                    changed_by=item.created_by,
                    changed_at=now,
                    reason="created",
                ),
                self._snapshot(item),
            )
        self.logger.info(
            "memory_created memory_id=%s kind=%s status=%s by=%s",
            item.memory_id,
            kind.value,
            status.value,
            item.created_by,
        )
        await self._audit(
            access, "create", item.memory_id, f"kind={kind.value} status={status.value}"
        )
        return item

    # -- read --------------------------------------------------------------

    async def get_item(self, memory_id: str, access: AccessContext) -> MemoryItem | None:
        item = await self.repo.get(memory_id)
        if item is None or not await self._visible(item, access):
            return None
        return item

    async def get_view(self, memory_id: str, access: AccessContext) -> dict | None:
        item = await self.get_item(memory_id, access)
        if item is None:
            return None
        subjects = await self.repo.get_subjects(memory_id)
        sources = await self.repo.get_sources(memory_id)
        links = await self.repo.get_links(memory_id)
        revisions = await self.repo.list_revisions(memory_id)
        provenance = await self.repo.get_provenance(memory_id)
        assignments = await self.repo.list_assignments(memory_id)
        return {
            "item": self._item_dict(item),
            "subjects": [{"subject_type": t, "subject_id": i} for t, i in subjects],
            "sources": sources,
            "links": links,
            "assignments": assignments,
            "provenance": provenance,
            "revisions": [
                {
                    "revision_id": r.revision_id,
                    "version": r.version,
                    "content": r.content,
                    "status": r.status.value,
                    "changed_by": r.changed_by,
                    "changed_at": r.changed_at.isoformat() if r.changed_at else None,
                    "reason": r.reason,
                }
                for r in revisions
            ],
        }

    async def search(
        self,
        *,
        access: AccessContext,
        query: str | None = None,
        kinds: list[str] | None = None,
        statuses: list[str] | None = None,
        visibilities: list[str] | None = None,
        subject_type: str | None = None,
        subject_id: str | None = None,
        source_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        updated_after: datetime | None = None,
        updated_before: datetime | None = None,
        include_retracted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        visible_to = None if access.is_admin else access.actor_id
        return await self.repo.search(
            query=query,
            kinds=kinds,
            statuses=statuses,
            visibilities=visibilities,
            subject_type=subject_type,
            subject_id=subject_id,
            source_id=source_id,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            visible_to_subject_id=visible_to,
            include_retracted=include_retracted,
            limit=limit,
            offset=offset,
        )

    # -- update ------------------------------------------------------------

    async def update(
        self,
        memory_id: str,
        *,
        access: AccessContext,
        expected_version: int,
        content: str | None = None,
        kind: MemoryKind | str | None = None,
        status: MemoryStatus | str | None = None,
        visibility: MemoryVisibility | str | None = None,
        reason: str | None = None,
    ) -> MemoryItem:
        item = await self.repo.get(memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        if not self._can_modify(item, access):
            raise MemoryPermissionDeniedError(memory_id)
        if expected_version != item.version:
            raise MemoryVersionConflictError(
                f"{memory_id}: expected version {expected_version}, found {item.version}"
            )

        new_item = item.model_copy(
            update={
                "content": content if content is not None else item.content,
                "kind": MemoryKind(kind) if kind is not None else item.kind,
                "status": MemoryStatus(status) if status is not None else item.status,
                "visibility": MemoryVisibility(visibility) if visibility is not None else item.visibility,
                "updated_at": utcnow(),
                "version": item.version + 1,
            }
        )
        content_changed = new_item.content != item.content
        async with self.repo.db.transaction():
            ok = await self.repo.replace_item(new_item, item.version)
            if not ok:
                raise MemoryVersionConflictError(
                    f"{memory_id}: version changed concurrently"
                )
            if content_changed:
                await self.repo.sync_fts(memory_id, new_item.content)
            await self.repo.insert_revision(
                MemoryRevision(
                    revision_id=new_id("rev"),
                    memory_id=memory_id,
                    version=new_item.version,
                    kind=new_item.kind,
                    content=new_item.content,
                    status=new_item.status,
                    changed_by=access.actor_id,
                    changed_at=new_item.updated_at,
                    reason=reason,
                ),
                self._snapshot(new_item),
            )
        self.logger.info(
            "memory_updated memory_id=%s version=%d by=%s",
            memory_id,
            new_item.version,
            access.actor_id,
        )
        await self._audit(access, "update", memory_id, f"version={new_item.version}")
        return new_item

    # -- human actions -----------------------------------------------------

    async def confirm(
        self, memory_id: str, *, access: AccessContext, reason: str | None = None
    ) -> MemoryItem:
        """A human confirms a candidate: it becomes ACTIVE and human-confirmed."""
        if not access.is_admin:
            raise MemoryPermissionDeniedError("confirm requires admin")
        item = await self.repo.get(memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        new_item = item.model_copy(
            update={
                "status": MemoryStatus.ACTIVE,
                "human_confirmed": True,
                "updated_at": utcnow(),
                "version": item.version + 1,
            }
        )
        async with self.repo.db.transaction():
            await self.repo.replace_item(new_item, item.version)
            await self.repo.insert_revision(
                MemoryRevision(
                    revision_id=new_id("rev"),
                    memory_id=memory_id,
                    version=new_item.version,
                    kind=new_item.kind,
                    content=new_item.content,
                    status=new_item.status,
                    changed_by=access.actor_id,
                    changed_at=new_item.updated_at,
                    reason=reason or "human_confirmed",
                ),
                self._snapshot(new_item),
            )
        await self._audit(access, "confirm", memory_id, "human_confirmed")
        return new_item

    async def retract(
        self, memory_id: str, *, access: AccessContext, reason: str | None = None
    ) -> MemoryItem:
        """Mark an item RETRACTED: kept for audit, excluded from default search."""
        item = await self.repo.get(memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        if not self._can_modify(item, access):
            raise MemoryPermissionDeniedError(memory_id)
        await self._invalidate_derived(memory_id, access)
        new_item = item.model_copy(
            update={
                "status": MemoryStatus.RETRACTED,
                "updated_at": utcnow(),
                "version": item.version + 1,
            }
        )
        async with self.repo.db.transaction():
            await self.repo.replace_item(new_item, item.version)
            await self.repo.insert_revision(
                MemoryRevision(
                    revision_id=new_id("rev"),
                    memory_id=memory_id,
                    version=new_item.version,
                    kind=new_item.kind,
                    content=new_item.content,
                    status=new_item.status,
                    changed_by=access.actor_id,
                    changed_at=new_item.updated_at,
                    reason=reason or "retracted",
                ),
                self._snapshot(new_item),
            )
        self.logger.info("memory_retracted memory_id=%s by=%s", memory_id, access.actor_id)
        await self._audit(access, "retract", memory_id, reason)
        return new_item

    async def delete(self, memory_id: str, *, access: AccessContext) -> None:
        """Permanently remove an item, its private content and index entries.

        Memories derived from this item are invalidated (retracted) first, so
        no reflection is left silently pointing at a deleted source.
        """
        if not access.is_admin:
            raise MemoryPermissionDeniedError("delete requires admin")
        item = await self.repo.get(memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        await self._invalidate_derived(memory_id, access)
        await self.repo.delete(memory_id)
        self.logger.info("memory_deleted memory_id=%s by=%s", memory_id, access.actor_id)
        await self._audit(access, "delete", memory_id)

    # -- AI proposal -------------------------------------------------------

    async def propose(
        self,
        *,
        access: AccessContext,
        content: str,
        conflicts_with: str | None = None,
        kind: MemoryKind | str = MemoryKind.REFLECTION,
        source_ids: list[str] | None = None,
        source_access: AccessContext | None = None,
        reason: str | None = None,
    ) -> MemoryItem:
        """Record an AI-proposed change without overwriting anything.

        ``source_ids`` are validated (they must exist and be visible under
        ``source_access``, which defaults to ``access``) and recorded as
        ``derived_from`` links, so a proposal always names the memories it was
        derived from — never fabricated sources. ``source_access`` lets a
        trusted internal reader that acts as a non-admin *writer* cite memories
        it legitimately read.

        If ``conflicts_with`` names a human-confirmed item, the proposal is
        stored as a ``NEEDS_REVIEW`` record linked to it — never applied to the
        original. The original remains untouched until a human approves it.
        """
        kind = MemoryKind(kind)
        # An AI proposal may never auto-classify itself as a fact about a
        # person; only a human can create a PERSONAL_FACT.
        if kind == MemoryKind.PERSONAL_FACT and not access.is_admin:
            self.logger.warning(
                "propose_personalfact_denied actor=%s downgraded_to=REFLECTION",
                access.actor_id,
            )
            kind = MemoryKind.REFLECTION

        validator = source_access or access
        derived_from: list[str] = []
        for sid in source_ids or []:
            src = await self.repo.get(sid)
            if src is None:
                raise MemoryNotFoundError(sid)
            if not await self._visible(src, validator):
                raise MemoryPermissionDeniedError(sid)
            derived_from.append(sid)

        if conflicts_with is not None:
            original = await self.repo.get(conflicts_with)
            if original is None:
                raise MemoryNotFoundError(conflicts_with)
            if original.human_confirmed and not access.is_admin:
                proposal = await self.create(
                    kind=kind,
                    content=content,
                    access=access,
                    status=MemoryStatus.NEEDS_REVIEW,
                    model_inferred=True,
                    created_by=access.actor_id,
                )
                await self.repo.add_link(
                    proposal.memory_id, conflicts_with, "conflicts_with"
                )
                for sid in derived_from:
                    await self.repo.add_link(proposal.memory_id, sid, "derived_from")
                self.logger.info(
                    "memory_conflict_proposed memory_id=%s conflicts_with=%s by=%s",
                    proposal.memory_id,
                    conflicts_with,
                    access.actor_id,
                )
                await self._audit(
                    access,
                    "propose_conflict",
                    proposal.memory_id,
                    f"conflicts_with={conflicts_with} sources={len(derived_from)}",
                )
                return proposal

        # No protected conflict: record the proposal as a plain candidate.
        proposal = await self.create(
            kind=kind,
            content=content,
            access=access,
            status=MemoryStatus.CANDIDATE,
            model_inferred=True,
            created_by=access.actor_id,
        )
        for sid in derived_from:
            await self.repo.add_link(proposal.memory_id, sid, "derived_from")
        await self._audit(
            access,
            "propose",
            proposal.memory_id,
            f"kind={kind.value} sources={len(derived_from)}",
        )
        return proposal

    async def history(
        self, memory_id: str, access: AccessContext
    ) -> list[MemoryRevision]:
        item = await self.get_item(memory_id, access)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        return await self.repo.list_revisions(memory_id)

    async def link(
        self, memory_id: str, linked_memory_id: str, link_type: str, access: AccessContext
    ) -> None:
        if not access.is_admin:
            raise MemoryPermissionDeniedError("link requires admin")
        if await self.repo.get(memory_id) is None:
            raise MemoryNotFoundError(memory_id)
        if await self.repo.get(linked_memory_id) is None:
            raise MemoryNotFoundError(linked_memory_id)
        await self.repo.add_link(memory_id, linked_memory_id, link_type)
        await self._audit(
            access, "link", memory_id, f"{linked_memory_id}:{link_type}"
        )

    # -- assignments -------------------------------------------------------
    #
    # ``memory_subjects`` says what/who a memory *describes*; an assignment
    # says which person the memory is *associated with for retrieval*. It is
    # metadata only: an assignment never changes content, status or
    # visibility, and it never grants a person access to a private memory.

    async def assign_memory(
        self, memory_id: str, person_id: str, access: AccessContext
    ) -> bool:
        """Associate a memory with a person. Returns False if already assigned.

        A non-admin may only assign a memory they can already see, and only to
        themselves. A PRIVATE memory may never be cross-person assigned unless
        an admin does it explicitly; future sharing needs a separate grant.
        """
        item = await self.repo.get(memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        if not await self._person_exists(person_id):
            raise MemoryNotFoundError(f"person {person_id} not found")
        if not access.is_admin:
            if not await self._visible(item, access):
                raise MemoryPermissionDeniedError(memory_id)
            if person_id != access.actor_id:
                raise MemoryPermissionDeniedError(
                    "cross-person assignment requires admin"
                )
        async with self.repo.db.transaction():
            created = await self.repo.assign(
                memory_id, person_id, created_by=access.actor_id
            )
            await self._audit(
                access,
                "assign_memory",
                memory_id,
                f"person_id={person_id} new={int(created)}",
            )
        self.logger.info(
            "memory_assigned memory_id=%s person_id=%s new=%d by=%s",
            memory_id,
            person_id,
            int(created),
            access.actor_id,
        )
        return created

    async def unassign_memory(
        self, memory_id: str, person_id: str, access: AccessContext
    ) -> bool:
        item = await self.repo.get(memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        if not await self._person_exists(person_id):
            raise MemoryNotFoundError(f"person {person_id} not found")
        if not access.is_admin:
            if not await self._visible(item, access):
                raise MemoryPermissionDeniedError(memory_id)
            if person_id != access.actor_id:
                raise MemoryPermissionDeniedError(
                    "removing another person's assignment requires admin"
                )
        async with self.repo.db.transaction():
            removed = await self.repo.unassign(memory_id, person_id)
            if removed:
                await self._audit(
                    access, "unassign_memory", memory_id, f"person_id={person_id}"
                )
        self.logger.info(
            "memory_unassigned memory_id=%s person_id=%s removed=%d by=%s",
            memory_id,
            person_id,
            int(removed),
            access.actor_id,
        )
        return removed

    async def list_assigned_memories(
        self, person_id: str, access: AccessContext
    ) -> list[dict]:
        """Memories associated with ``person_id`` and visible to ``access``.

        Non-admin callers may only list their own assignments. Visibility is
        still enforced per item, so an assignment is not an access grant.
        """
        if not await self._person_exists(person_id):
            raise MemoryNotFoundError(f"person {person_id} not found")
        if not access.is_admin and access.actor_id != person_id:
            raise MemoryPermissionDeniedError(
                "listing another person's assignments requires admin"
            )
        ids = await self.repo.list_assigned_memory_ids(person_id)
        items = await self.repo.get_many(ids)
        out: list[dict] = []
        for item in items:
            if await self._visible(item, access):
                out.append(self._item_dict(item))
        return out

    async def list_memory_assignments(
        self, memory_id: str, access: AccessContext
    ) -> list[dict]:
        item = await self.repo.get(memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id)
        if not access.is_admin and not await self._visible(item, access):
            raise MemoryPermissionDeniedError(memory_id)
        return await self.repo.list_assignments(memory_id)

    # -- helpers -----------------------------------------------------------

    async def _invalidate_derived(self, source_id: str, access: AccessContext) -> None:
        """Retract every memory derived from ``source_id``.

        Called when a source is deleted or retracted, so a reflection never
        stays live while its evidence has gone. Invalidation keeps the derived
        record (for audit) but excludes it from default search.
        """
        derived = await self.repo.find_derived(source_id)
        for mid in derived:
            item = await self.repo.get(mid)
            if item is None or item.status == MemoryStatus.RETRACTED:
                continue
            new_item = item.model_copy(
                update={
                    "status": MemoryStatus.RETRACTED,
                    "updated_at": utcnow(),
                    "version": item.version + 1,
                }
            )
            async with self.repo.db.transaction():
                await self.repo.replace_item(new_item, item.version)
                await self.repo.insert_revision(
                    MemoryRevision(
                        revision_id=new_id("rev"),
                        memory_id=mid,
                        version=new_item.version,
                        kind=new_item.kind,
                        content=new_item.content,
                        status=new_item.status,
                        changed_by=access.actor_id,
                        changed_at=new_item.updated_at,
                        reason="derived_source_invalidated",
                    ),
                    self._snapshot(new_item),
                )
            self.logger.info(
                "memory_derived_invalidated memory_id=%s source=%s", mid, source_id
            )
            await self._audit(access, "retract", mid, f"derived_source={source_id}")

    async def _require_subjects_authorized(self, person_ids: list[str]) -> None:
        if self.authorize_subject is None:
            raise MemoryPermissionDeniedError("subject authorization not available")
        for pid in person_ids:
            if not await self.authorize_subject(pid):
                raise MemoryPermissionDeniedError(
                    f"person {pid} is not authorised for memory learning"
                )

    async def _person_exists(self, person_id: str) -> bool:
        if self.person_exists is not None:
            return await self.person_exists(person_id)
        # No callback wired: fall back to the FK-checkable database itself.
        cur = await self.repo.db.execute(
            "SELECT 1 FROM persons WHERE person_id = ?", (person_id,)
        )
        return await cur.fetchone() is not None

    async def _visible(self, item: MemoryItem, access: AccessContext) -> bool:
        if access.is_admin:
            return True
        if item.visibility == MemoryVisibility.PUBLIC:
            return True
        # PRIVATE / SHARED items are visible only to a PERSON subject.
        subjects = await self.repo.get_subjects(item.memory_id)
        return any(
            st == "PERSON" and sid == access.actor_id for st, sid in subjects
        )

    def _can_modify(self, item: MemoryItem, access: AccessContext) -> bool:
        return access.is_admin or item.created_by == access.actor_id

    def _snapshot(self, item: MemoryItem) -> dict:
        return {
            "content": item.content,
            "kind": item.kind.value,
            "status": item.status.value,
            "visibility": item.visibility.value,
            "human_confirmed": item.human_confirmed,
        }

    def _item_dict(self, item: MemoryItem) -> dict:
        return {
            "memory_id": item.memory_id,
            "kind": item.kind.value,
            "content": item.content,
            "status": item.status.value,
            "visibility": item.visibility.value,
            "created_by": item.created_by,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            "version": item.version,
            "valid_from": item.valid_from.isoformat() if item.valid_from else None,
            "valid_until": item.valid_until.isoformat() if item.valid_until else None,
            "human_confirmed": item.human_confirmed,
            "source_supported": item.source_supported,
            "self_reported": item.self_reported,
            "third_party_reported": item.third_party_reported,
            "model_inferred": item.model_inferred,
        }


def build_memory_service(repos: "Repositories") -> MemoryService:
    """Build the standard service wired to the runtime repositories.

    The authorization callback is the trusted local one: a person may be the
    subject of a learned memory only while they are ACTIVE and have consented
    to memory learning. Consent is re-checked on every write, so revoking it
    blocks new memories immediately.
    """

    async def authorize(person_id: str) -> bool:
        person = await repos.person.get(person_id)
        if person is None or person.registration_status != RegistrationStatus.ACTIVE:
            return False
        consent = await repos.person.get_consent(person_id)
        return consent is not None and consent.allow_memory_learning

    async def person_exists(person_id: str) -> bool:
        return await repos.person.get(person_id) is not None

    return MemoryService(
        MemoryRepository(repos.db),
        authorize_subject=authorize,
        person_exists=person_exists,
        audit=repos.audit,
    )
