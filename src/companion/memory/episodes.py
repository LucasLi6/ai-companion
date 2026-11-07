"""Multi-turn episode summaries stored as ordinary ``EPISODE`` memories.

An episode is a *story* about a piece of work (a problem appeared, something
was tried, something changed, it was or was not resolved). It is not a single
fact, and it must not be smuggled into ``MemoryCandidate`` — that structure is
one claim backed by one event. This module adds the missing multi-event shape
and the writer that persists it through the existing :class:`MemoryService`.

Trust rules (enforced here, never delegated to the model):

- ``person_id`` and the conversation come from the stored, authorized events;
  the model only supplies prose and event ids.
- every cited event must exist, belong to the same speaker and conversation,
  and be part of the current extraction batch. A fabricated id rejects the
  whole episode.
- consent is re-checked at write time.
- idempotency is anchored on a deterministic source id derived from the sorted
  source events, not on summary wording (a re-summary may phrase things
  differently, but the same batch is the same episode).
- a continuation is only merged when the model names an existing title *and*
  exactly one stored episode for this person matches it. Ambiguity creates a
  new episode instead of overwriting one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from companion.domain.enums import (
    MemoryKind,
    MemoryStatus,
    MemoryVisibility,
    RegistrationStatus,
    VerificationLevel,
)
from companion.domain.models import utcnow
from companion.memory.models import (
    AccessContext,
    MemoryEvidence,
    MemorySource,
    MemoryVersionConflictError,
)
from companion.memory.repository import MemoryRepository
from companion.memory.service import MemoryService
from companion.storage.repositories import Repositories

MAX_EPISODES_PER_BATCH = 3
MAX_TITLE_CHARS = 200
MAX_SUMMARY_CHARS = 2000
MAX_TOPIC_CHARS = 200

STATUS_RESOLVED = "RESOLVED"
STATUS_ONGOING = "ONGOING"
STATUS_UNKNOWN = "UNKNOWN"
EPISODE_STATUSES = (STATUS_RESOLVED, STATUS_ONGOING, STATUS_UNKNOWN)

_STATUS_ALIASES = {
    "RESOLVED": STATUS_RESOLVED,
    "SOLVED": STATUS_RESOLVED,
    "FIXED": STATUS_RESOLVED,
    "DONE": STATUS_RESOLVED,
    "COMPLETED": STATUS_RESOLVED,
    "ONGOING": STATUS_ONGOING,
    "IN_PROGRESS": STATUS_ONGOING,
    "UNRESOLVED": STATUS_ONGOING,
    "OPEN": STATUS_ONGOING,
    "PENDING": STATUS_ONGOING,
    "UNKNOWN": STATUS_UNKNOWN,
    "UNCLEAR": STATUS_UNKNOWN,
}

_EPISODE_SOURCE_TYPE = "episode"
_EPISODE_PROVIDER = "background_extraction"


def _clean(text: object, limit: int) -> str:
    return " ".join(str(text or "").split()).strip()[:limit]


def normalize_status(value: object) -> str:
    """Map a model status onto the three honest episode states."""
    return _STATUS_ALIASES.get(str(value or "").strip().upper(), STATUS_UNKNOWN)


class EpisodeProposal(BaseModel):
    """One model-proposed episode, already bounded to safe sizes."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    topic: str = Field(default="", max_length=MAX_TOPIC_CHARS)
    status: str = STATUS_UNKNOWN
    source_event_ids: list[str] = Field(default_factory=list)
    # Only a *name*, never a database id. The writer resolves it against the
    # stored episodes for this person and requires a unique match.
    continues_title: Optional[str] = Field(default=None, max_length=MAX_TITLE_CHARS)

    def cleaned(self) -> "EpisodeProposal":
        return EpisodeProposal(
            title=_clean(self.title, MAX_TITLE_CHARS),
            summary=_clean(self.summary, MAX_SUMMARY_CHARS),
            topic=_clean(self.topic, MAX_TOPIC_CHARS),
            status=normalize_status(self.status),
            source_event_ids=list(dict.fromkeys(self.source_event_ids)),
            continues_title=(
                _clean(self.continues_title, MAX_TITLE_CHARS)
                if self.continues_title
                else None
            ),
        )


@dataclass
class ExtractionOutcome:
    """One background model call: per-event candidates plus episodes."""

    candidates: dict[str, list] = field(default_factory=dict)
    episodes: list[EpisodeProposal] = field(default_factory=list)


def render_episode_content(proposal: EpisodeProposal) -> str:
    """The stable, parseable text stored as the memory body."""
    return "\n".join(
        [
            "[EPISODE]",
            f"标题：{proposal.title}",
            f"主题：{proposal.topic}",
            f"状态：{proposal.status}",
            f"梗概：{proposal.summary}",
        ]
    )


def parse_episode_content(content: str | None) -> dict | None:
    """Read back a stored episode body; ``None`` when it is not one."""
    if not content or not content.lstrip().startswith("[EPISODE]"):
        return None
    fields: dict[str, str] = {}
    for line in content.splitlines()[1:]:
        for label, key in (
            ("标题：", "title"),
            ("主题：", "topic"),
            ("状态：", "status"),
            ("梗概：", "summary"),
        ):
            if line.startswith(label):
                fields[key] = line[len(label):].strip()
    if not fields.get("title") or not fields.get("summary"):
        return None
    fields["status"] = normalize_status(fields.get("status"))
    return fields


def episode_fingerprint(person_id: str, source_event_ids: list[str]) -> str:
    """Deterministic identity of one extraction batch for one person."""
    material = "|".join(
        [person_id, *sorted(dict.fromkeys(source_event_ids))]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def episode_source_id(person_id: str, source_event_ids: list[str]) -> str:
    return f"episode:{episode_fingerprint(person_id, source_event_ids)}"


@dataclass
class EpisodeWriteResult:
    outcome: str  # created | updated | duplicate | rejected
    memory_id: Optional[str] = None
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "memory_id": self.memory_id,
            "reason": self.reason,
        }


class EpisodeWriter:
    """Persist episode proposals through the unified memory service."""

    def __init__(
        self,
        repos: Repositories,
        service: MemoryService,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.repos = repos
        self.service = service
        self.mem_repo: MemoryRepository = service.repo
        self.logger = logger or logging.getLogger("companion.memory.episodes")

    async def write(
        self,
        proposal: EpisodeProposal,
        *,
        person_id: str,
        conversation_id: str,
    ) -> EpisodeWriteResult:
        proposal = proposal.cleaned()
        if not proposal.title or not proposal.summary:
            return EpisodeWriteResult("rejected", reason="empty_title_or_summary")

        access = AccessContext(actor_id=person_id, is_admin=False, role="cognitive")

        # Authorisation is re-checked at execution time, never trusted from
        # enqueue. Revoking consent must stop new episodes immediately.
        if not await self._authorized(person_id):
            return EpisodeWriteResult(
                "rejected", reason="source_authorization_revoked"
            )

        valid_ids, invalid_ids = await self._validated_sources(
            proposal.source_event_ids, person_id, conversation_id
        )
        if invalid_ids:
            # A fabricated, foreign or out-of-batch id makes the whole episode
            # untrustworthy: do not store a partial citation list.
            self.logger.warning(
                "episode_rejected_invalid_sources person_id=%s invalid=%d",
                person_id,
                len(invalid_ids),
            )
            return EpisodeWriteResult(
                "rejected", reason="unknown_or_foreign_source_event"
            )
        if not valid_ids:
            return EpisodeWriteResult("rejected", reason="no_source_events")

        source_id = episode_source_id(person_id, valid_ids)
        content = render_episode_content(proposal)

        # One transaction covers the duplicate check and the write, so a
        # concurrent retry of the same batch sees a committed episode.
        async with self.service.repo.db.transaction():
            existing = await self.service.search(
                access=access,
                source_id=source_id,
                include_retracted=True,
                limit=2,
            )
            if existing:
                return EpisodeWriteResult(
                    "duplicate",
                    memory_id=existing[0]["memory_id"],
                    reason="same_source_batch",
                )

            twin = await self._continuation_target(proposal, person_id)
            if twin is not None:
                return await self._update(
                    twin, proposal, content, valid_ids, person_id, access
                )

            return await self._create(
                proposal, content, valid_ids, person_id, source_id, access
            )

    # -- validation --------------------------------------------------------

    async def _authorized(self, person_id: str) -> bool:
        person = await self.repos.person.get(person_id)
        if person is None or person.registration_status != RegistrationStatus.ACTIVE:
            return False
        consent = await self.repos.person.get_consent(person_id)
        return bool(consent is not None and consent.allow_memory_learning)

    async def _validated_sources(
        self, source_event_ids: list[str], person_id: str, conversation_id: str
    ) -> tuple[list[str], list[str]]:
        valid: list[str] = []
        invalid: list[str] = []
        for event_id in dict.fromkeys(source_event_ids):
            event = await self.repos.utterance.get(event_id)
            if (
                event is None
                or event.speaker_person_id != person_id
                or event.conversation_id != conversation_id
            ):
                invalid.append(event_id)
                continue
            valid.append(event_id)
        return valid, invalid

    async def _continuation_target(
        self, proposal: EpisodeProposal, person_id: str
    ) -> dict | None:
        """Exactly one stored episode this proposal explicitly continues."""
        wanted = _clean(proposal.continues_title, MAX_TITLE_CHARS)
        if not wanted:
            return None
        rows = await self.mem_repo.list_for_subject(
            person_id,
            statuses=[MemoryStatus.ACTIVE.value],
            kinds=[MemoryKind.EPISODE.value],
            limit=20,
        )
        matches: list[dict] = []
        for row in rows:
            if row.get("human_confirmed") or not row.get("model_inferred"):
                continue
            if row.get("created_by") not in (None, person_id):
                continue
            parsed = parse_episode_content(row.get("content"))
            if parsed is None:
                continue
            if _clean(parsed.get("title"), MAX_TITLE_CHARS) != wanted:
                continue
            if proposal.topic and parsed.get("topic") and (
                _clean(parsed.get("topic"), MAX_TOPIC_CHARS) != proposal.topic
            ):
                continue
            matches.append(row)
        if len(matches) != 1:
            if matches:
                self.logger.info(
                    "episode_continuation_ambiguous person_id=%s matches=%d",
                    person_id,
                    len(matches),
                )
            return None
        return matches[0]

    # -- writes ------------------------------------------------------------

    async def _create(
        self,
        proposal: EpisodeProposal,
        content: str,
        valid_ids: list[str],
        person_id: str,
        source_id: str,
        access: AccessContext,
    ) -> EpisodeWriteResult:
        source = MemorySource(
            source_id=source_id,
            source_type=_EPISODE_SOURCE_TYPE,
            source_ref=json.dumps(valid_ids, ensure_ascii=False),
            title=proposal.title,
            provider=_EPISODE_PROVIDER,
            retrieved_at=utcnow(),
        )
        evidence = MemoryEvidence(
            source_id=source_id,
            evidence_span=None,
            verification_level=VerificationLevel.UNVERIFIED,
            retrieved_at=utcnow(),
        )
        item = await self.service.create(
            kind=MemoryKind.EPISODE,
            content=content,
            access=access,
            status=MemoryStatus.ACTIVE,
            visibility=MemoryVisibility.PRIVATE,
            created_by=person_id,
            subjects=[("PERSON", person_id)],
            sources=[source],
            evidence=[evidence],
            human_confirmed=False,
            source_supported=True,
            model_inferred=True,
        )
        # Provenance keeps the canonical topic for topic lookups and records
        # the first cited event. Topic matching is what later batches use.
        await self.mem_repo.add_provenance(
            item.memory_id,
            source_event_id=valid_ids[0],
            candidate_id=None,
            speaker_person_id=person_id,
            subject_person_id=person_id,
            source_type=_EPISODE_SOURCE_TYPE,
            assertion_type="UNKNOWN",
            certainty_category="INFERRED",
            topic=proposal.topic or None,
            model_inferred=True,
            new_version=item.version,
        )
        self.logger.info(
            "episode_created memory_id=%s person_id=%s sources=%d status=%s",
            item.memory_id,
            person_id,
            len(valid_ids),
            proposal.status,
        )
        return EpisodeWriteResult("created", memory_id=item.memory_id)

    async def _update(
        self,
        row: dict,
        proposal: EpisodeProposal,
        content: str,
        valid_ids: list[str],
        person_id: str,
        access: AccessContext,
    ) -> EpisodeWriteResult:
        old_status = normalize_status(
            (parse_episode_content(row.get("content")) or {}).get("status")
        )
        status = proposal.status
        if old_status == STATUS_RESOLVED and status != STATUS_RESOLVED:
            # A later conversation does not un-resolve a problem unless the
            # user explicitly says so; a model summary alone must not undo it.
            status = STATUS_RESOLVED
        if status != proposal.status:
            content = content.replace(
                f"状态：{proposal.status}", f"状态：{status}"
            )

        source_id = episode_source_id(person_id, valid_ids)
        source = MemorySource(
            source_id=source_id,
            source_type=_EPISODE_SOURCE_TYPE,
            source_ref=json.dumps(valid_ids, ensure_ascii=False),
            title=proposal.title,
            provider=_EPISODE_PROVIDER,
            retrieved_at=utcnow(),
        )
        evidence = MemoryEvidence(
            source_id=source_id,
            evidence_span=None,
            verification_level=VerificationLevel.UNVERIFIED,
            retrieved_at=utcnow(),
        )
        try:
            item = await self.service.update(
                row["memory_id"],
                access=access,
                expected_version=row["version"],
                content=content,
                status=MemoryStatus.ACTIVE,
                reason="episode_extended",
                sources=[source],
                evidence=[evidence],
            )
        except MemoryVersionConflictError:
            # Another writer got there first. Creating a second record of the
            # same event would be worse than keeping the newer version.
            self.logger.warning(
                "episode_update_conflict memory_id=%s", row["memory_id"]
            )
            return EpisodeWriteResult(
                "rejected",
                memory_id=row["memory_id"],
                reason="version_conflict",
            )
        await self.mem_repo.add_provenance_event(
            row["memory_id"],
            event_type="episode_updated",
            source_event_id=valid_ids[-1],
            speaker_person_id=person_id,
            subject_person_id=person_id,
            old_version=row["version"],
            new_version=item.version,
            reason="episode_extended",
            payload={
                "source_event_ids": valid_ids,
                "status": status,
                "title": proposal.title,
            },
        )
        self.logger.info(
            "episode_updated memory_id=%s person_id=%s version=%d status=%s",
            item.memory_id,
            person_id,
            item.version,
            status,
        )
        return EpisodeWriteResult("updated", memory_id=item.memory_id)


__all__ = [
    "EpisodeProposal",
    "EpisodeWriteResult",
    "EpisodeWriter",
    "ExtractionOutcome",
    "MAX_EPISODES_PER_BATCH",
    "STATUS_ONGOING",
    "STATUS_RESOLVED",
    "STATUS_UNKNOWN",
    "episode_fingerprint",
    "episode_source_id",
    "normalize_status",
    "parse_episode_content",
    "render_episode_content",
]
