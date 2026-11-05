"""Data-access repositories over :class:`Database`.

Each repository is a thin, parameterised-SQL layer. No business rules live
here — those belong in the registry, orchestrator and policy layers. All
timestamps are stored as timezone-aware UTC ISO-8601 strings.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from companion.domain.enums import (
    AddresseeHint,
    AssertionType,
    CertaintyCategory,
    CognitiveAction,
    EventSource,
    JobStatus,
    JobType,
    MemoryCandidateStatus,
    MemoryKind,
    MemoryOperation,
    PersonRole,
    ProcessingStatus,
    ReasonCode,
    RegistrationStatus,
    ReplyNecessity,
)
from companion.domain.models import (
    AffectiveSignal,
    AgentAction,
    CognitiveResult,
    Consent,
    ConversationContext,
    DialogueImpulse,
    MemoryCandidate,
    NonresponseMotive,
    PragmaticAnalysis,
    Person,
    ProcessingJob,
    UtteranceEvent,
    utcnow,
)

from companion.cognition.nonresponse import NonresponseTrace
from companion.core.fast_path import LOCAL_PROVIDER
from companion.learning.state import LearningRepository
from companion.people.profile_repository import PersonProfileRepository
from companion.research.state import ResearchStateRepository
from companion.goals.state import GoalRepository
from companion.routing.repository import RoutingRepository
from companion.runtime_context.repository import RuntimeSnapshotRepository

from .database import Database


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


class JobCapacityError(RuntimeError):
    """An autonomous-research capacity check rejected an enqueue.

    Raised *inside* the enqueue transaction so a concurrent submitter can
    never slip past the same capacity check.
    """

    def __init__(self, pending: int, limit: int) -> None:
        super().__init__(f"autonomous research queue is full ({pending}/{limit})")
        self.pending = pending
        self.limit = limit


class ResearchCapacityError(RuntimeError):
    """Too many *unfinished* autonomous research detail rows.

    Unlike a job-count limit, this counts the research records themselves, so
    a task that has no job (for example after a rejected enqueue) still
    consumes a slot until it is completed, failed or cancelled.
    """

    def __init__(self, active: int, limit: int) -> None:
        super().__init__(
            f"too many unfinished autonomous research tasks ({active}/{limit})"
        )
        self.active = active
        self.limit = limit


_QUESTION_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_QUESTION_WHITESPACE = re.compile(r"\s+", re.UNICODE)


def normalize_research_question(question: str) -> str:
    """Canonical form used only for dedupe, never for display or search.

    NFKC folds compatibility variants (full-width characters), casefold folds
    case, punctuation is dropped and whitespace collapsed. This makes
    ``"OTA 为什么不稳定？"`` and ``"ota 为什么不稳定?"`` the same key without an
    LLM call or a second question classifier.
    """
    text = unicodedata.normalize("NFKC", str(question or ""))
    text = text.casefold()
    text = _QUESTION_PUNCTUATION.sub(" ", text)
    return _QUESTION_WHITESPACE.sub(" ", text).strip()


def research_dedupe_key(
    question: str,
    source_memory_ids: tuple[str, ...] | list[str] = (),
    *,
    origin: str = "curiosity",
) -> str:
    """Stable identity of a research need within one source/permission scope."""
    normalized = normalize_research_question(question)
    sources = sorted({str(item) for item in source_memory_ids if str(item)})
    payload = json.dumps(
        {"q": normalized, "sources": sources, "origin": str(origin)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PersonRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def register(self, person: Person, consent: Consent | None = None) -> Person:
        async with self.db.transaction():
            await self.db.conn.execute(
                "INSERT INTO persons(person_id, display_name, role, registration_status,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    person.person_id,
                    person.display_name,
                    person.role.value,
                    person.registration_status.value,
                    _iso(person.created_at),
                    _iso(person.updated_at),
                ),
            )
            if consent is not None:
                await self.db.conn.execute(
                    "INSERT INTO consents(person_id, allow_processing, allow_persistence,"
                    " allow_memory_learning, allow_external_ai, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        consent.person_id,
                        int(consent.allow_processing),
                        int(consent.allow_persistence),
                        int(consent.allow_memory_learning),
                        int(consent.allow_external_ai),
                        _iso(consent.updated_at),
                    ),
                )
        return person

    async def get(self, person_id: str) -> Person | None:
        cur = await self.db.execute(
            "SELECT person_id, display_name, role, registration_status, created_at,"
            " updated_at FROM persons WHERE person_id = ?",
            (person_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return Person(
            person_id=row["person_id"],
            display_name=row["display_name"],
            role=PersonRole(row["role"]),
            registration_status=RegistrationStatus(row["registration_status"]),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    async def list(self) -> list[Person]:
        cur = await self.db.execute(
            "SELECT person_id, display_name, role, registration_status, created_at,"
            " updated_at FROM persons ORDER BY created_at ASC"
        )
        rows = await cur.fetchall()
        return [
            Person(
                person_id=r["person_id"],
                display_name=r["display_name"],
                role=PersonRole(r["role"]),
                registration_status=RegistrationStatus(r["registration_status"]),
                created_at=_dt(r["created_at"]),
                updated_at=_dt(r["updated_at"]),
            )
            for r in rows
        ]

    async def set_consent(self, consent: Consent) -> None:
        await self.db.execute(
            "INSERT INTO consents(person_id, allow_processing, allow_persistence,"
            " allow_memory_learning, allow_external_ai, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(person_id) DO UPDATE SET"
            " allow_processing=excluded.allow_processing,"
            " allow_persistence=excluded.allow_persistence,"
            " allow_memory_learning=excluded.allow_memory_learning,"
            " allow_external_ai=excluded.allow_external_ai,"
            " updated_at=excluded.updated_at",
            (
                consent.person_id,
                int(consent.allow_processing),
                int(consent.allow_persistence),
                int(consent.allow_memory_learning),
                int(consent.allow_external_ai),
                _iso(consent.updated_at),
            ),
        )

    async def set_registration_status(
        self, person_id: str, status: RegistrationStatus
    ) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE persons SET registration_status = ?, updated_at = ?"
            " WHERE person_id = ?",
            (status.value, _iso(utcnow()), person_id),
        )

    async def get_consent(self, person_id: str) -> Consent | None:
        cur = await self.db.execute(
            "SELECT person_id, allow_processing, allow_persistence,"
            " allow_memory_learning, allow_external_ai, updated_at FROM consents"
            " WHERE person_id = ?",
            (person_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return Consent(
            person_id=row["person_id"],
            allow_processing=bool(row["allow_processing"]),
            allow_persistence=bool(row["allow_persistence"]),
            allow_memory_learning=bool(row["allow_memory_learning"]),
            allow_external_ai=bool(row["allow_external_ai"]),
            updated_at=_dt(row["updated_at"]),
        )


class ConversationRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def get(self, conversation_id: str) -> ConversationContext | None:
        cur = await self.db.execute(
            "SELECT conversation_id, quiet_mode, last_action, updated_at"
            " FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return ConversationContext(
            conversation_id=row["conversation_id"],
            quiet_mode=bool(row["quiet_mode"]),
            last_action=row["last_action"],
            updated_at=_dt(row["updated_at"]),
        )

    async def get_or_create(self, conversation_id: str) -> ConversationContext:
        existing = await self.get(conversation_id)
        if existing is not None:
            return existing
        from companion.domain.models import utcnow

        now = utcnow()
        await self.db.execute(
            "INSERT INTO conversations(conversation_id, quiet_mode, last_action,"
            " created_at, updated_at) VALUES (?, 0, NULL, ?, ?)",
            (conversation_id, _iso(now), _iso(now)),
        )
        return ConversationContext(
            conversation_id=conversation_id,
            quiet_mode=False,
            updated_at=now,
        )

    async def set_quiet_mode(self, conversation_id: str, enabled: bool) -> None:
        from companion.domain.models import utcnow

        await self.get_or_create(conversation_id)
        await self.db.execute(
            "UPDATE conversations SET quiet_mode = ?, updated_at = ?"
            " WHERE conversation_id = ?",
            (int(enabled), _iso(utcnow()), conversation_id),
        )

    async def update_last_action(self, conversation_id: str, action: str | None) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE conversations SET last_action = ?, updated_at = ?"
            " WHERE conversation_id = ?",
            (action, _iso(utcnow()), conversation_id),
        )


class UtteranceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(self, event: UtteranceEvent) -> None:
        await self.db.execute(
            "INSERT INTO utterances(event_id, conversation_id, speaker_person_id,"
            " speaker_label, text, source, addressee_hint, occurred_at,"
            " processing_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.conversation_id,
                event.speaker_person_id,
                event.speaker_label,
                event.text,
                event.source.value,
                event.addressee_hint.value,
                _iso(event.occurred_at),
                event.processing_status.value,
                _iso(utcnow()),
            ),
        )

    async def get(self, event_id: str) -> UtteranceEvent | None:
        cur = await self.db.execute(
            "SELECT event_id, conversation_id, speaker_person_id, speaker_label, text,"
            " source, addressee_hint, occurred_at, processing_status FROM utterances"
            " WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return UtteranceEvent(
            event_id=row["event_id"],
            conversation_id=row["conversation_id"],
            speaker_person_id=row["speaker_person_id"],
            speaker_label=row["speaker_label"],
            text=row["text"],
            source=EventSource(row["source"]),
            addressee_hint=AddresseeHint(row["addressee_hint"]),
            occurred_at=_dt(row["occurred_at"]),
            processing_status=ProcessingStatus(row["processing_status"]),
        )

    async def update_status(self, event_id: str, status: ProcessingStatus) -> None:
        await self.db.execute(
            "UPDATE utterances SET processing_status = ? WHERE event_id = ?",
            (status.value, event_id),
        )

    async def get_many(self, event_ids: list[str]) -> dict[str, UtteranceEvent]:
        """Return the requested events in one query, keyed by event id."""
        if not event_ids:
            return {}
        placeholders = ",".join("?" for _ in event_ids)
        cur = await self.db.execute(
            "SELECT event_id, conversation_id, speaker_person_id, speaker_label,"
            " text, source, addressee_hint, occurred_at, processing_status"
            f" FROM utterances WHERE event_id IN ({placeholders})",
            tuple(event_ids),
        )
        rows = await cur.fetchall()
        return {
            r["event_id"]: UtteranceEvent(
                event_id=r["event_id"],
                conversation_id=r["conversation_id"],
                speaker_person_id=r["speaker_person_id"],
                speaker_label=r["speaker_label"],
                text=r["text"],
                source=EventSource(r["source"]),
                addressee_hint=AddresseeHint(r["addressee_hint"]),
                occurred_at=_dt(r["occurred_at"]),
                processing_status=ProcessingStatus(r["processing_status"]),
            )
            for r in rows
        }

    async def recent(self, conversation_id: str, limit: int = 20) -> list[str]:
        cur = await self.db.execute(
            "SELECT event_id FROM utterances WHERE conversation_id = ?"
            " ORDER BY occurred_at DESC, event_id DESC LIMIT ?",
            (conversation_id, limit),
        )
        rows = await cur.fetchall()
        return [r["event_id"] for r in rows]

    async def conversation_stats(self, person_id: str) -> dict:
        """Read-only counts for one speaker's own records.

        Scope is exactly ``utterances.speaker_person_id = person_id``: a
        conversation the speaker shares with other people contributes only
        their own messages, and another person's messages/replies are never
        attributed to them. Replies are counted by joining the action to its
        source utterance, so only visible RESPOND turns are included.

        This is a bounded stat query, not a general history dump: no message
        text is read and no caller-supplied SQL is involved.
        """
        cur = await self.db.execute(
            "SELECT COUNT(*) AS messages,"
            " COUNT(DISTINCT conversation_id) AS sessions"
            " FROM utterances WHERE speaker_person_id = ?",
            (person_id,),
        )
        row = await cur.fetchone()
        cur = await self.db.execute(
            "SELECT COUNT(*) AS replies FROM agent_actions a"
            " JOIN utterances u ON u.event_id = a.event_id"
            " WHERE u.speaker_person_id = ?"
            " AND a.action_type = 'RESPOND'"
            " AND a.status IN ('COMPLETED', 'TRUNCATED')",
            (person_id,),
        )
        replies = await cur.fetchone()
        return {
            "user_messages": int(row["messages"] if row else 0),
            "assistant_replies": int(replies["replies"] if replies else 0),
            "sessions": int(row["sessions"] if row else 0),
        }

    async def recent_context(
        self,
        conversation_id: str,
        exclude_event_id: str | None,
        limit: int = 20,
        *,
        before_event: "UtteranceEvent | None" = None,
    ) -> list[UtteranceEvent]:
        """Return the *most recent* ``limit`` utterances, oldest first.

        The inner query selects the newest ``limit`` rows (``occurred_at DESC``),
        then the outer query flips them back to ascending order so a chat-history
        consumer reads them chronologically. ``exclude_event_id`` — the event
        already persisted for the current input — is always dropped, and only the
        given ``conversation_id`` is considered, never another conversation.

        Ordering semantics for ties and late events: the sort key is the pair
        ``(occurred_at, event_id)``. ``occurred_at`` is the event's *claimed*
        time, so an event that arrives late but carries an older timestamp still
        sorts where its timestamp says it belongs; ``event_id`` is a
        deterministic tie-break for events that share a timestamp, so the same
        data always returns in the same order.
        """
        where = "conversation_id = ?"
        params: list = [conversation_id]
        if exclude_event_id is not None:
            where += " AND event_id != ?"
            params.append(exclude_event_id)
        if before_event is not None:
            # Background analysis of an already-answered turn must see only
            # what the speaker could see at that moment: exclude this event and
            # anything ordered after it (occurred_at, event_id tie-break).
            where += " AND (occurred_at < ? OR (occurred_at = ? AND event_id < ?))"
            occurred = before_event.occurred_at.isoformat()
            params.extend([occurred, occurred, before_event.event_id])
        sql = (
            "SELECT * FROM ("
            "  SELECT event_id, conversation_id, speaker_person_id, speaker_label,"
            "         text, source, addressee_hint, occurred_at, processing_status"
            f"  FROM utterances WHERE {where}"
            "  ORDER BY occurred_at DESC, event_id DESC LIMIT ?"
            ") ORDER BY occurred_at ASC, event_id ASC"
        )
        params.append(limit)
        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        return [
            UtteranceEvent(
                event_id=r["event_id"],
                conversation_id=r["conversation_id"],
                speaker_person_id=r["speaker_person_id"],
                speaker_label=r["speaker_label"],
                text=r["text"],
                source=EventSource(r["source"]),
                addressee_hint=AddresseeHint(r["addressee_hint"]),
                occurred_at=_dt(r["occurred_at"]),
                processing_status=ProcessingStatus(r["processing_status"]),
            )
            for r in rows
        ]

    async def last_before_speaker(
        self,
        person_id: str,
        *,
        before: datetime,
        exclude_event_id: str | None = None,
    ) -> UtteranceEvent | None:
        """The most recent utterance by ``person_id`` strictly before ``before``.

        Used for "Last Heard" continuity. The comparison is strict so the
        current event can never be returned as its own history, and the
        exclusion is applied again for an event that shares its timestamp.
        """
        where = "speaker_person_id = ? AND occurred_at < ?"
        params: list = [person_id, _iso(before)]
        if exclude_event_id is not None:
            where += " AND event_id != ?"
            params.append(exclude_event_id)
        cur = await self.db.execute(
            "SELECT event_id, conversation_id, speaker_person_id, speaker_label,"
            " text, source, addressee_hint, occurred_at, processing_status"
            f" FROM utterances WHERE {where}"
            " ORDER BY occurred_at DESC, event_id DESC LIMIT 1",
            tuple(params),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return UtteranceEvent(
            event_id=row["event_id"],
            conversation_id=row["conversation_id"],
            speaker_person_id=row["speaker_person_id"],
            speaker_label=row["speaker_label"],
            text=row["text"],
            source=EventSource(row["source"]),
            addressee_hint=AddresseeHint(row["addressee_hint"]),
            occurred_at=_dt(row["occurred_at"]),
            processing_status=ProcessingStatus(row["processing_status"]),
        )

    async def earliest_for_speaker(
        self,
        person_id: str,
        conversation_id: str | None = None,
        *,
        exclude_event_id: str | None = None,
        limit: int = 3,
    ) -> list[UtteranceEvent]:
        """The oldest utterances by ``person_id``, oldest first.

        Used when the speaker explicitly asks what they said first. It is
        scoped to one conversation when given, so a request in a current
        conversation never silently pulls text from another one.
        """
        where = "speaker_person_id = ?"
        params: list = [person_id]
        if conversation_id is not None:
            where += " AND conversation_id = ?"
            params.append(conversation_id)
        if exclude_event_id is not None:
            where += " AND event_id != ?"
            params.append(exclude_event_id)
        sql = (
            "SELECT event_id, conversation_id, speaker_person_id, speaker_label,"
            " text, source, addressee_hint, occurred_at, processing_status"
            f" FROM utterances WHERE {where}"
            " ORDER BY occurred_at ASC, event_id ASC LIMIT ?"
        )
        params.append(limit)
        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        return [
            UtteranceEvent(
                event_id=r["event_id"],
                conversation_id=r["conversation_id"],
                speaker_person_id=r["speaker_person_id"],
                speaker_label=r["speaker_label"],
                text=r["text"],
                source=EventSource(r["source"]),
                addressee_hint=AddresseeHint(r["addressee_hint"]),
                occurred_at=_dt(r["occurred_at"]),
                processing_status=ProcessingStatus(r["processing_status"]),
            )
            for r in rows
        ]

    async def list_events(
        self, person_id: str | None = None, limit: int = 100
    ) -> list[UtteranceEvent]:
        if person_id is not None:
            cur = await self.db.execute(
                "SELECT event_id, conversation_id, speaker_person_id, speaker_label,"
                " text, source, addressee_hint, occurred_at, processing_status"
                " FROM utterances WHERE speaker_person_id = ?"
                " ORDER BY occurred_at DESC LIMIT ?",
                (person_id, limit),
            )
        else:
            cur = await self.db.execute(
                "SELECT event_id, conversation_id, speaker_person_id, speaker_label,"
                " text, source, addressee_hint, occurred_at, processing_status"
                " FROM utterances ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cur.fetchall()
        return [
            UtteranceEvent(
                event_id=r["event_id"],
                conversation_id=r["conversation_id"],
                speaker_person_id=r["speaker_person_id"],
                speaker_label=r["speaker_label"],
                text=r["text"],
                source=EventSource(r["source"]),
                addressee_hint=AddresseeHint(r["addressee_hint"]),
                occurred_at=_dt(r["occurred_at"]),
                processing_status=ProcessingStatus(r["processing_status"]),
            )
            for r in rows
        ]

    async def list_by_status(
        self, status: ProcessingStatus, limit: int = 1000
    ) -> list[UtteranceEvent]:
        """Return events in a given lifecycle status, oldest first."""
        cur = await self.db.execute(
            "SELECT event_id, conversation_id, speaker_person_id, speaker_label,"
            " text, source, addressee_hint, occurred_at, processing_status"
            " FROM utterances WHERE processing_status = ?"
            " ORDER BY occurred_at ASC LIMIT ?",
            (status.value, limit),
        )
        rows = await cur.fetchall()
        return [
            UtteranceEvent(
                event_id=r["event_id"],
                conversation_id=r["conversation_id"],
                speaker_person_id=r["speaker_person_id"],
                speaker_label=r["speaker_label"],
                text=r["text"],
                source=EventSource(r["source"]),
                addressee_hint=AddresseeHint(r["addressee_hint"]),
                occurred_at=_dt(r["occurred_at"]),
                processing_status=ProcessingStatus(r["processing_status"]),
            )
            for r in rows
        ]

    async def delete_owned_event(
        self, event_id: str, *, speaker_person_id: str
    ) -> bool:
        """Delete one event the speaker owns, cascading its derived rows.

        The speaker condition is part of the SQL statement, so a mistargeted id
        can never delete another person's utterance. Foreign keys (analysis,
        action, candidates, runtime snapshot, scene receipt, extraction
        bookkeeping, route decisions/tool calls) cascade from this row.
        Returns whether a row was actually deleted.
        """
        cur = await self.db.execute(
            "DELETE FROM utterances WHERE event_id = ? AND speaker_person_id = ?",
            (event_id, speaker_person_id),
        )
        return cur.rowcount > 0


class AnalysisRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def delete_analysis(self, event_id: str) -> None:
        """Remove a placeholder analysis before storing the deferred result."""
        await self.db.execute(
            "DELETE FROM cognitive_analyses WHERE event_id = ?", (event_id,)
        )

    async def save_analysis(
        self,
        analysis: CognitiveResult,
        analysis_id: str,
        *,
        persona: dict | None = None,
    ) -> None:
        signal_json = (
            json.dumps(analysis.affective_signal.model_dump(), ensure_ascii=False)
            if analysis.affective_signal is not None
            else None
        )
        motive_json = (
            json.dumps(
                analysis.nonresponse_motive.model_dump(), ensure_ascii=False
            )
            if analysis.nonresponse_motive is not None
            else None
        )
        scene_json = (
            json.dumps(
                analysis.scene_proposal.model_dump(mode="json"),
                ensure_ascii=False,
            )
            if analysis.scene_proposal is not None
            else None
        )
        impulse_json = (
            json.dumps(analysis.dialogue_impulse.model_dump(), ensure_ascii=False)
            if analysis.dialogue_impulse is not None
            else None
        )
        pragmatic_json = (
            json.dumps(analysis.pragmatic.model_dump(), ensure_ascii=False)
            if analysis.pragmatic is not None
            else None
        )
        persona_agent_id = None
        persona_revision = None
        persona_snapshot_json = None
        if isinstance(persona, dict):
            agent_id = persona.get("agent_id")
            if agent_id:
                persona_agent_id = str(agent_id)[:128]
            try:
                revision = int(persona.get("revision"))
            except (TypeError, ValueError):
                revision = None
            if revision is not None:
                persona_revision = revision
            snapshot = persona.get("snapshot")
            if isinstance(snapshot, dict):
                persona_snapshot_json = json.dumps(snapshot, ensure_ascii=False)
        await self.db.execute(
            "INSERT INTO cognitive_analyses(analysis_id, event_id, meaning, action,"
            " reason_code, provider, referenced_memory_ids, response_style, created_at,"
            " intent, addressed_to, silence_requested, reply_necessity, conversation_state,"
            " affective_signal_json, nonresponse_motive_json, scene_proposal_json,"
            " dialogue_impulse_json, pragmatic_json, persona_agent_id, persona_revision,"
            " persona_snapshot_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis_id,
                analysis.event_id,
                analysis.meaning,
                analysis.action.value,
                analysis.reason_code.value,
                analysis.provider,
                json.dumps(analysis.referenced_memory_ids),
                analysis.response_style,
                _iso(utcnow()),
                analysis.intent,
                analysis.addressed_to,
                int(analysis.silence_requested),
                analysis.reply_necessity.value,
                analysis.conversation_state,
                signal_json,
                motive_json,
                scene_json,
                impulse_json,
                pragmatic_json,
                persona_agent_id,
                persona_revision,
                persona_snapshot_json,
            ),
        )

    async def get_analysis(self, event_id: str) -> CognitiveResult | None:
        cur = await self.db.execute(
            "SELECT analysis_id, event_id, meaning, action, reason_code, provider,"
            " referenced_memory_ids, response_style, intent, addressed_to,"
            " silence_requested, reply_necessity, conversation_state,"
            " affective_signal_json, nonresponse_motive_json, scene_proposal_json,"
            " dialogue_impulse_json, pragmatic_json"
            " FROM cognitive_analyses WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        signal = None
        if row["affective_signal_json"]:
            try:
                signal = AffectiveSignal.model_validate(
                    json.loads(row["affective_signal_json"])
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                signal = None
        motive = None
        if row["nonresponse_motive_json"]:
            try:
                motive = NonresponseMotive.model_validate(
                    json.loads(row["nonresponse_motive_json"])
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                motive = None
        scene_proposal = None
        if row["scene_proposal_json"]:
            from companion.domain.models import SceneProposal

            try:
                scene_proposal = SceneProposal.model_validate(
                    json.loads(row["scene_proposal_json"])
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                scene_proposal = None
        impulse = None
        if row["dialogue_impulse_json"]:
            try:
                impulse = DialogueImpulse.model_validate(
                    json.loads(row["dialogue_impulse_json"])
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                impulse = None
        pragmatic = None
        if row["pragmatic_json"]:
            try:
                pragmatic = PragmaticAnalysis.model_validate(
                    json.loads(row["pragmatic_json"])
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                pragmatic = None
        return CognitiveResult(
            event_id=row["event_id"],
            meaning=row["meaning"],
            action=CognitiveAction(row["action"]),
            reason_code=ReasonCode(row["reason_code"]),
            referenced_memory_ids=json.loads(row["referenced_memory_ids"]),
            response_style=row["response_style"],
            provider=row["provider"],
            intent=row["intent"] or "unknown",
            addressed_to=row["addressed_to"],
            silence_requested=bool(row["silence_requested"]),
            reply_necessity=ReplyNecessity(row["reply_necessity"] or "OPTIONAL"),
            conversation_state=row["conversation_state"] or "unknown",
            affective_signal=signal,
            nonresponse_motive=motive,
            scene_proposal=scene_proposal,
            dialogue_impulse=impulse,
            pragmatic=pragmatic,
        )

    async def get_analysis_id(self, event_id: str) -> Optional[str]:
        cur = await self.db.execute(
            "SELECT analysis_id FROM cognitive_analyses WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        return row["analysis_id"] if row is not None else None

    async def get_analysis_persona(self, event_id: str) -> dict | None:
        """The trusted character revision an analysis was produced with.

        Returns ``None`` when no analysis exists. A row that predates schema 17
        has NULL metadata; callers must treat that as "unknown", not as a
        matching revision.
        """
        cur = await self.db.execute(
            "SELECT persona_agent_id, persona_revision, persona_snapshot_json"
            " FROM cognitive_analyses WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        snapshot = None
        if row["persona_snapshot_json"]:
            try:
                parsed = json.loads(row["persona_snapshot_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, dict):
                snapshot = parsed
        return {
            "agent_id": row["persona_agent_id"],
            "revision": row["persona_revision"],
            "snapshot": snapshot,
        }

    async def save_candidate(self, candidate: MemoryCandidate) -> None:
        # INSERT OR IGNORE + idx_candidates_event_content (schema v7) makes a
        # repeated proposal for the same event a no-op, so retries and
        # duplicate model output can never multiply candidate rows.
        await self.db.execute(
            "INSERT OR IGNORE INTO memory_candidates(candidate_id, source_event_id,"
            " subject_person_id, content, operation, status, created_at,"
            " assertion_type, certainty_category, explicit_remember_request, topic,"
            " kind, source_evidence_span)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.candidate_id,
                candidate.source_event_id,
                candidate.subject_person_id,
                candidate.content,
                candidate.operation.value,
                candidate.status.value,
                _iso(utcnow()),
                candidate.assertion_type.value,
                candidate.certainty_category.value,
                int(candidate.explicit_remember_request),
                candidate.topic,
                candidate.kind.value if candidate.kind is not None else None,
                candidate.source_evidence_span,
            ),
        )

    async def get_candidates(self, event_id: str) -> list[MemoryCandidate]:
        cur = await self.db.execute(
            "SELECT candidate_id, source_event_id, subject_person_id, content,"
            " operation, status, assertion_type, certainty_category,"
            " explicit_remember_request, topic, kind, source_evidence_span"
            " FROM memory_candidates WHERE source_event_id = ?"
            " ORDER BY rowid ASC",
            (event_id,),
        )
        rows = await cur.fetchall()
        return [
            MemoryCandidate(
                candidate_id=r["candidate_id"],
                source_event_id=r["source_event_id"],
                subject_person_id=r["subject_person_id"],
                content=r["content"],
                operation=MemoryOperation(r["operation"]),
                status=MemoryCandidateStatus(r["status"]),
                assertion_type=AssertionType(r["assertion_type"] or "UNKNOWN"),
                certainty_category=CertaintyCategory(
                    r["certainty_category"] or "UNKNOWN"
                ),
                explicit_remember_request=bool(r["explicit_remember_request"]),
                topic=r["topic"],
                kind=MemoryKind(r["kind"]) if r["kind"] else None,
                source_evidence_span=r["source_evidence_span"],
            )
            for r in rows
        ]

    async def get_candidate(self, candidate_id: str) -> MemoryCandidate | None:
        cur = await self.db.execute(
            "SELECT candidate_id, source_event_id, subject_person_id, content,"
            " operation, status, assertion_type, certainty_category,"
            " explicit_remember_request, topic, kind, source_evidence_span"
            " FROM memory_candidates WHERE candidate_id = ?",
            (candidate_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return MemoryCandidate(
            candidate_id=row["candidate_id"],
            source_event_id=row["source_event_id"],
            subject_person_id=row["subject_person_id"],
            content=row["content"],
            operation=MemoryOperation(row["operation"]),
            status=MemoryCandidateStatus(row["status"]),
            assertion_type=AssertionType(row["assertion_type"] or "UNKNOWN"),
            certainty_category=CertaintyCategory(
                row["certainty_category"] or "UNKNOWN"
            ),
            explicit_remember_request=bool(row["explicit_remember_request"]),
            topic=row["topic"],
            kind=MemoryKind(row["kind"]) if row["kind"] else None,
            source_evidence_span=row["source_evidence_span"],
        )

    async def update_candidate_status(
        self, candidate_id: str, status: MemoryCandidateStatus
    ) -> None:
        await self.db.execute(
            "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
            (status.value, candidate_id),
        )

    async def mark_candidates_explicit(self, event_id: str) -> None:
        """Treat this event's candidates as explicitly requested.

        The route, not the wording, decided that the turn asked for a durable
        memory. Any candidate already extracted for this event (background pass
        or an earlier attempt) belongs to that same explicit request, so it may
        take part in the synchronous write. Already-decided candidates keep
        their recorded outcome; this only stops them being filtered out by
        ``only_explicit``.
        """
        await self.db.execute(
            "UPDATE memory_candidates SET explicit_remember_request = 1"
            " WHERE source_event_id = ?",
            (event_id,),
        )

    # -- candidate -> memory write mapping ---------------------------------

    async def get_candidate_write(self, candidate_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT candidate_id, memory_id, source_event_id, outcome, detail,"
            " created_at, updated_at FROM memory_candidate_writes"
            " WHERE candidate_id = ?",
            (candidate_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def record_candidate_write(
        self,
        candidate_id: str,
        *,
        source_event_id: str,
        outcome: str,
        memory_id: str | None = None,
        detail: str | None = None,
    ) -> dict:
        """Insert the immutable candidate -> memory result (idempotent).

        The first row wins: a retry reads the recorded outcome instead of
        re-writing a second memory. A DB failure happens before this insert,
        so a failure is never recorded as success.
        """
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT OR IGNORE INTO memory_candidate_writes(candidate_id, memory_id,"
            " source_event_id, outcome, detail, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (candidate_id, memory_id, source_event_id, outcome, detail, now, now),
        )
        existing = await self.get_candidate_write(candidate_id)
        return existing if existing is not None else {
            "candidate_id": candidate_id,
            "memory_id": memory_id,
            "source_event_id": source_event_id,
            "outcome": outcome,
            "detail": detail,
        }

    async def list_candidate_writes(self, event_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT w.candidate_id, w.memory_id, w.source_event_id, w.outcome,"
            " w.detail, w.created_at, w.updated_at"
            " FROM memory_candidate_writes w"
            " JOIN memory_candidates c ON c.candidate_id = w.candidate_id"
            " WHERE c.source_event_id = ? ORDER BY c.rowid ASC",
            (event_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def last_action_for_speaker(
        self,
        person_id: str,
        *,
        before: datetime,
        action_type: str | None = None,
    ) -> tuple[AgentAction, datetime] | None:
        """Latest completed action for anything this speaker said.

        Returns ``(action, occurred_at)`` where ``occurred_at`` is the source
        utterance's time. ``action_type="RESPOND"`` answers "when did the
        companion last actually say something to this speaker?"; ``None``
        includes WAIT/IGNORE, i.e. "when did we last have a turn?".
        """
        where = (
            "u.speaker_person_id = ? AND u.occurred_at < ?"
            " AND a.status IN ('COMPLETED', 'TRUNCATED')"
        )
        params: list = [person_id, _iso(before)]
        if action_type is not None:
            where += " AND a.action_type = ?"
            params.append(action_type)
        cur = await self.db.execute(
            "SELECT a.action_id, a.event_id, a.action_type, a.response_text,"
            " a.reason_code, a.status, a.created_at, a.model_suggested_action,"
            " a.policy_final_action, a.override_reason, a.nonresponse_trace_json,"
            " u.occurred_at"
            " FROM agent_actions a JOIN utterances u ON u.event_id = a.event_id"
            f" WHERE {where}"
            " ORDER BY u.occurred_at DESC, a.rowid DESC LIMIT 1",
            tuple(params),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._to_action(row), _dt(row["occurred_at"])

    async def delete_action(self, event_id: str) -> None:
        """Remove a partial (CANCELLED/FAILED) action before a retry.

        A completed or truncated action is never deleted: it is the durable
        record of what was actually emitted.
        """
        await self.db.execute(
            "DELETE FROM agent_actions WHERE event_id = ?", (event_id,)
        )

    async def save_action(self, action: AgentAction) -> None:
        trace_json = (
            json.dumps(action.nonresponse_trace, ensure_ascii=False)
            if action.nonresponse_trace is not None
            else None
        )
        await self.db.execute(
            "INSERT INTO agent_actions(action_id, event_id, action_type, response_text,"
            " reason_code, status, created_at, model_suggested_action,"
            " policy_final_action, override_reason, nonresponse_trace_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action.action_id,
                action.event_id,
                action.action_type.value,
                action.response_text,
                action.reason_code.value,
                action.status,
                _iso(action.created_at),
                action.model_suggested_action.value
                if action.model_suggested_action is not None
                else None,
                action.policy_final_action.value
                if action.policy_final_action is not None
                else None,
                action.override_reason,
                trace_json,
            ),
        )

    async def get_action(self, event_id: str) -> AgentAction | None:
        cur = await self.db.execute(
            "SELECT action_id, event_id, action_type, response_text, reason_code,"
            " status, created_at, model_suggested_action, policy_final_action,"
            " override_reason, nonresponse_trace_json"
            " FROM agent_actions WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._to_action(row)

    async def get_actions_for_events(
        self, event_ids: list[str]
    ) -> dict[str, AgentAction]:
        """Return one action per event in a single query (batched for history).

        Empty input returns an empty mapping. Callers (the context builder) use
        this to interleave the agent's responses into the recent conversation
        without issuing one query per event.
        """
        if not event_ids:
            return {}
        placeholders = ",".join("?" for _ in event_ids)
        cur = await self.db.execute(
            "SELECT action_id, event_id, action_type, response_text, reason_code,"
            " status, created_at, model_suggested_action, policy_final_action,"
            " override_reason, nonresponse_trace_json"
            f" FROM agent_actions WHERE event_id IN ({placeholders})",
            tuple(event_ids),
        )
        rows = await cur.fetchall()
        return {r["event_id"]: self._to_action(r) for r in rows}

    def _to_action(self, row) -> AgentAction:
        trace = None
        raw_trace = row["nonresponse_trace_json"]
        if raw_trace:
            try:
                trace = NonresponseTrace.model_validate(
                    json.loads(raw_trace)
                ).model_dump()
            except (ValueError, TypeError, json.JSONDecodeError):
                trace = None
        return AgentAction(
            action_id=row["action_id"],
            event_id=row["event_id"],
            action_type=CognitiveAction(row["action_type"]),
            response_text=row["response_text"],
            reason_code=ReasonCode(row["reason_code"]),
            status=row["status"],
            created_at=_dt(row["created_at"]),
            model_suggested_action=(
                CognitiveAction(row["model_suggested_action"])
                if row["model_suggested_action"]
                else None
            ),
            policy_final_action=(
                CognitiveAction(row["policy_final_action"])
                if row["policy_final_action"]
                else None
            ),
            override_reason=row["override_reason"],
            nonresponse_trace=trace,
        )


class MemoryFactRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def insert(
        self, fact_id: str, content: str, subject_person_id: str | None
    ) -> None:
        from companion.domain.models import utcnow

        now = utcnow()
        await self.db.execute(
            "INSERT INTO memory_facts(fact_id, subject_person_id, content, status,"
            " created_at, updated_at) VALUES (?, ?, ?, 'manual', ?, ?)",
            (fact_id, subject_person_id, content, _iso(now), _iso(now)),
        )

    async def retrieve(
        self, query: str | None, person_id: str | None
    ) -> list[dict]:
        """Retrieve hand-authored facts.

        This is an exact/substring match over stored facts, *not* semantic
        search. ``query=None`` returns all matching facts for ``person_id``.
        """
        sql = "SELECT fact_id, subject_person_id, content, status, created_at FROM memory_facts"
        params: list = []
        conditions: list[str] = []
        if person_id is not None:
            conditions.append("subject_person_id = ?")
            params.append(person_id)
        if query:
            conditions.append("content LIKE ?")
            params.append(f"%{query}%")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at ASC LIMIT 50"
        cur = await self.db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


class JobRepository:
    _COLUMNS = (
        "job_id, job_key, job_type, payload, status, attempts, max_attempts,"
        " last_error, created_at, started_at, completed_at, owner_id,"
        " lease_expires_at, heartbeat_at, next_run_at, priority, source_kind,"
        " source_ref, execution_mode, checkpoint_json, cancel_requested,"
        " result_json"
    )

    def __init__(self, db: Database) -> None:
        self.db = db

    async def enqueue(
        self,
        job_type: JobType,
        job_key: str,
        payload: dict,
        max_attempts: int = 3,
        next_run_at: Optional[datetime] = None,
        *,
        priority: str = "NORMAL",
        source_kind: str = "SYSTEM",
        source_ref: Optional[str] = None,
        execution_mode: str = "BACKGROUND",
        checkpoint: Optional[dict] = None,
        autonomous_limit: Optional[int] = None,
        autonomous_types: tuple[str, ...] = (),
    ) -> str:
        """Enqueue a job, deduplicating on ``job_key``.

        Returns the *existing* job id if ``job_key`` is already present, so
        re-submitting the same event never creates a second job. Dedupe, the
        optional autonomous-research capacity check and insertion all happen
        under one ``BEGIN IMMEDIATE`` transaction: two concurrent submissions
        cannot both insert the same key, and they cannot both pass a capacity
        check that only one slot can satisfy. ``autonomous_limit`` applies only
        to ``autonomous_types`` from a non-USER source; maintenance work such
        as TURN_ANALYSIS, MEMORY_CONSOLIDATION and EMBEDDING is never counted
        or rejected here.
        """
        from companion.domain.models import new_id, utcnow

        type_value = job_type.value if isinstance(job_type, JobType) else str(job_type)
        job_id = new_id("job")
        is_autonomous = bool(
            autonomous_limit is not None
            and type_value in autonomous_types
            and str(source_kind).upper() != "USER"
        )
        async with self.db.transaction(immediate=True):
            cur = await self.db.execute(
                "SELECT job_id FROM processing_jobs WHERE job_key = ?",
                (job_key,),
            )
            row = await cur.fetchone()
            if row is not None:
                return row["job_id"]

            if is_autonomous:
                placeholders = ",".join("?" for _ in autonomous_types)
                cur = await self.db.execute(
                    "SELECT COUNT(*) AS n FROM processing_jobs"
                    " WHERE status = 'PENDING'"
                    f" AND job_type IN ({placeholders})"
                    " AND source_kind != 'USER'",
                    tuple(autonomous_types),
                )
                count_row = await cur.fetchone()
                pending = int(count_row["n"]) if count_row is not None else 0
                if pending >= int(autonomous_limit):
                    raise JobCapacityError(pending, int(autonomous_limit))

            cur = await self.db.execute(
                "INSERT OR IGNORE INTO processing_jobs(job_id, job_key, job_type,"
                " payload, status, attempts, max_attempts, created_at, next_run_at,"
                " priority, source_kind, source_ref, execution_mode,"
                " checkpoint_json, cancel_requested)"
                " VALUES (?, ?, ?, ?, 'PENDING', 0, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    job_id,
                    job_key,
                    type_value,
                    json.dumps(payload),
                    max_attempts,
                    _iso(utcnow()),
                    _iso(next_run_at) if next_run_at is not None else None,
                    priority,
                    source_kind,
                    source_ref,
                    execution_mode,
                    json.dumps(checkpoint) if checkpoint else None,
                ),
            )
            if (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1:
                return job_id
            cur = await self.db.execute(
                "SELECT job_id FROM processing_jobs WHERE job_key = ?",
                (job_key,),
            )
            row = await cur.fetchone()
            if row is None:  # Extremely defensive: a concurrent delete.
                raise RuntimeError(f"job {job_key!r} vanished during enqueue")
            return row["job_id"]

    async def get(self, job_id: str) -> ProcessingJob | None:
        cur = await self.db.execute(
            f"SELECT {self._COLUMNS} FROM processing_jobs WHERE job_id = ?",
            (job_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._to_job(row)

    async def get_by_key(self, job_key: str) -> ProcessingJob | None:
        cur = await self.db.execute(
            f"SELECT {self._COLUMNS} FROM processing_jobs WHERE job_key = ?",
            (job_key,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._to_job(row)

    async def pending(
        self,
        limit: int = 100,
        *,
        include_foreground: bool = False,
        now: datetime | None = None,
    ) -> list[ProcessingJob]:
        """Jobs that are due now, ordered by priority then creation time.

        ``next_run_at`` delays a job (used by the batched background
        extraction and by a cooperative pause). A delayed job is still
        ``PENDING`` in the database — it is simply not claimable yet, so a
        restart or a manual ``jobs list`` still sees it.

        Foreground tasks are executed by ``TaskService.execute_now`` and are
        excluded by default: the background poller must never race a caller
        that is already waiting for the result. ``include_foreground=True`` is
        available for inspection/manual tooling.
        """
        from companion.domain.models import utcnow

        if now is None:
            now = utcnow()
        mode_clause = "" if include_foreground else " AND execution_mode = 'BACKGROUND'"
        cur = await self.db.execute(
            f"SELECT {self._COLUMNS} FROM processing_jobs"
            " WHERE status = 'PENDING'"
            "   AND cancel_requested = 0"
            "   AND (next_run_at IS NULL OR next_run_at <= ?)"
            f"{mode_clause}"
            " ORDER BY CASE priority"
            "            WHEN 'HIGH' THEN 0 WHEN 'NORMAL' THEN 1"
            "            WHEN 'LOW' THEN 2 ELSE 1 END,"
            "          created_at ASC LIMIT ?",
            (_iso(now), limit),
        )
        rows = await cur.fetchall()
        return [self._to_job(r) for r in rows]

    async def count_pending(self, *, autonomous_only: bool = False) -> int:
        sql = "SELECT COUNT(*) AS n FROM processing_jobs WHERE status = 'PENDING'"
        params: tuple = ()
        if autonomous_only:
            sql += " AND source_kind != 'USER'"
        cur = await self.db.execute(sql, params)
        row = await cur.fetchone()
        return int(row["n"]) if row is not None else 0

    async def count_running(self, job_types: tuple[str, ...]) -> int:
        if not job_types:
            return 0
        placeholders = ",".join("?" for _ in job_types)
        cur = await self.db.execute(
            f"SELECT COUNT(*) AS n FROM processing_jobs"
            f" WHERE status = 'RUNNING' AND job_type IN ({placeholders})",
            tuple(job_types),
        )
        row = await cur.fetchone()
        return int(row["n"]) if row is not None else 0

    async def promote_foreground(self, job_id: str) -> bool:
        """Turn a queued background job into a foreground execution.

        Only a PENDING job may be promoted. This lets an explicit user request
        reuse an already-queued background task instead of duplicating it,
        while never stealing a job another process is running.
        """
        cur = await self.db.execute(
            "UPDATE processing_jobs SET execution_mode = 'FOREGROUND',"
            " priority = 'HIGH', next_run_at = NULL"
            " WHERE job_id = ? AND status = 'PENDING'",
            (job_id,),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def list_jobs(self, status: JobStatus | None = None) -> list[ProcessingJob]:
        sql = f"SELECT {self._COLUMNS} FROM processing_jobs"
        params: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status.value,)
        sql += " ORDER BY created_at ASC"
        cur = await self.db.execute(sql, params)
        rows = await cur.fetchall()
        return [self._to_job(r) for r in rows]

    async def claim(
        self, job_id: str, owner_id: str = "worker", lease_seconds: int = 300
    ) -> bool:
        """Atomically claim a ``PENDING`` job, returning ``True`` on success.

        The ``WHERE status = 'PENDING'`` guard plus SQLite's write lock means
        that, of any number of concurrent workers (even on separate
        connections), at most one sees ``rowcount == 1``.

        The claim records an ``owner_id`` and a time-limited lease so a crashed
        worker's in-flight job can be reclaimed once its lease lapses, without
        ever resetting another *live* worker's job.
        """
        from companion.domain.models import utcnow

        now = utcnow()
        lease = now + timedelta(seconds=lease_seconds) if lease_seconds else None
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'RUNNING', attempts = attempts + 1,"
            " owner_id = ?, lease_expires_at = ?, heartbeat_at = ?,"
            " started_at = COALESCE(started_at, ?), last_error = NULL"
            " WHERE job_id = ? AND status = 'PENDING' AND attempts < max_attempts"
            " AND cancel_requested = 0",
            (
                owner_id,
                _iso(lease) if lease is not None else None,
                _iso(now),
                _iso(now),
                job_id,
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    async def heartbeat(
        self, job_id: str, owner_id: str, lease_seconds: int = 300
    ) -> bool:
        """Extend the lease of a job we still own, proving we are alive.

        Returns ``True`` only if this ``owner_id`` still holds the ``RUNNING``
        job — a worker that has lost its lease cannot extend it.
        """
        from companion.domain.models import utcnow

        now = utcnow()
        lease = now + timedelta(seconds=lease_seconds)
        cur = await self.db.execute(
            "UPDATE processing_jobs SET heartbeat_at = ?, lease_expires_at = ?"
            " WHERE job_id = ? AND status = 'RUNNING' AND owner_id = ?",
            (_iso(now), _iso(lease), job_id, owner_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    async def complete(self, job_id: str, result: Any = None) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE processing_jobs SET status = 'COMPLETED', completed_at = ?,"
            " result_json = COALESCE(?, result_json),"
            " owner_id = NULL, lease_expires_at = NULL WHERE job_id = ?",
            (
                _iso(utcnow()),
                json.dumps(result, ensure_ascii=False, default=str)
                if result is not None
                else None,
                job_id,
            ),
        )

    async def complete_owned(
        self, job_id: str, owner_id: str, result: Any = None
    ) -> bool:
        """Commit a result only while this worker still owns a live lease.

        A worker that lost its lease, whose lease lapsed, or whose job was
        cancelled/reclaimed must not mark the work completed. The caller
        reports failure rather than claiming success.
        """
        from companion.domain.models import utcnow

        now = utcnow()
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'COMPLETED', completed_at = ?,"
            " result_json = COALESCE(?, result_json),"
            " owner_id = NULL, lease_expires_at = NULL"
            " WHERE job_id = ? AND status = 'RUNNING' AND owner_id = ?"
            " AND cancel_requested = 0"
            " AND lease_expires_at IS NOT NULL AND lease_expires_at > ?",
            (
                _iso(now),
                json.dumps(result, ensure_ascii=False, default=str)
                if result is not None
                else None,
                job_id,
                owner_id,
                _iso(now),
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def lease_owned(
        self,
        job_id: str,
        owner_id: str,
        *,
        lease_seconds: Optional[int] = None,
    ) -> bool:
        """Return whether this owner still holds a live, uncancelled lease.

        With ``lease_seconds`` the lease is renewed first (a heartbeat), so a
        long multi-stage task can prove liveness at safe boundaries without a
        separate timer.
        """
        from companion.domain.models import utcnow

        if lease_seconds is not None:
            return await self.heartbeat(job_id, owner_id, lease_seconds)
        cur = await self.db.execute(
            "SELECT status, owner_id, cancel_requested, lease_expires_at"
            " FROM processing_jobs WHERE job_id = ?",
            (job_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return False
        if row["status"] != "RUNNING" or row["owner_id"] != owner_id:
            return False
        if bool(row["cancel_requested"]):
            return False
        expires = _dt(row["lease_expires_at"])
        return expires is not None and expires > utcnow()

    async def cancel(self, job_id: str) -> bool:
        """Request cancellation, terminal immediately for a PENDING job.

        A RUNNING job cannot be killed inside a database transaction or an
        in-flight external request; the flag is cooperative. The worker checks
        it before claiming and after the handler returns. Returns ``True`` when
        the job is already terminal as CANCELLED or the request was recorded.
        """
        from companion.domain.models import utcnow

        now = _iso(utcnow())
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'CANCELLED', cancel_requested = 1,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE job_id = ? AND status = 'PENDING'",
            (now, job_id),
        )
        if (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0:
            return True
        cur = await self.db.execute(
            "UPDATE processing_jobs SET cancel_requested = 1"
            " WHERE job_id = ? AND status = 'RUNNING'",
            (job_id,),
        )
        if (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0:
            return True
        job = await self.get(job_id)
        return job is not None and job.status == JobStatus.CANCELLED

    async def cancel_pending_for_source(
        self, source_kind: str, source_ref: str
    ) -> int:
        """Trusted, queue-independent cancellation for a revoked source."""
        from companion.domain.models import utcnow

        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'CANCELLED', cancel_requested = 1,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE status = 'PENDING' AND source_kind = ? AND source_ref = ?",
            (_iso(utcnow()), source_kind, source_ref),
        )
        return cur.rowcount if hasattr(cur, "rowcount") else 0

    async def cancellation_requested(self, job_id: str) -> bool:
        cur = await self.db.execute(
            "SELECT cancel_requested FROM processing_jobs WHERE job_id = ?",
            (job_id,),
        )
        row = await cur.fetchone()
        return bool(row["cancel_requested"]) if row is not None else False

    async def save_checkpoint(
        self, job_id: str, checkpoint: dict, *, owner_id: Optional[str] = None
    ) -> bool:
        """Persist a checkpoint by merging it into the existing document.

        Stage outputs are stored by stable reference; callers must not put
        unbounded raw text here. The merge means a handler can save one field
        at a time without losing earlier stages.

        With ``owner_id`` the write is conditional on this worker still owning
        a live ``RUNNING`` lease: a stale worker can neither overwrite a newer
        worker's checkpoint nor revive a cancelled/completed job.
        """
        from companion.domain.models import utcnow

        job = await self.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False
        now = utcnow()
        if owner_id is not None:
            if job.owner_id != owner_id or job.cancel_requested:
                return False
            if job.lease_expires_at is None or job.lease_expires_at <= now:
                return False
        merged = dict(job.checkpoint or {})
        merged.update(checkpoint or {})
        sql = (
            "UPDATE processing_jobs SET checkpoint_json = ?"
            " WHERE job_id = ? AND status = 'RUNNING' AND cancel_requested = 0"
        )
        params: tuple = (
            json.dumps(merged, ensure_ascii=False, default=str),
            job_id,
        )
        if owner_id is not None:
            sql += (
                " AND owner_id = ? AND lease_expires_at IS NOT NULL"
                " AND lease_expires_at > ?"
            )
            params = (*params, owner_id, _iso(now))
        cur = await self.db.execute(sql, params)
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def mark_cancelled(self, job_id: str) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE processing_jobs SET status = 'CANCELLED', cancel_requested = 1,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE job_id = ?",
            (_iso(utcnow()), job_id),
        )

    async def mark_cancelled_pending(self, job_id: str) -> bool:
        """Cancel a still-PENDING row without touching a claimed execution."""
        from companion.domain.models import utcnow

        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'CANCELLED', cancel_requested = 1,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE job_id = ? AND status = 'PENDING'",
            (_iso(utcnow()), job_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def mark_cancelled_owned(self, job_id: str, owner_id: str) -> bool:
        """Cancel only the RUNNING job this caller still owns."""
        from companion.domain.models import utcnow

        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'CANCELLED', cancel_requested = 1,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE job_id = ? AND status = 'RUNNING' AND owner_id = ?",
            (_iso(utcnow()), job_id, owner_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def postpone(self, job_id: str, next_run_at: datetime) -> None:
        """Delay a still-PENDING job without changing its attempt count."""
        await self.db.execute(
            "UPDATE processing_jobs SET next_run_at = ?"
            " WHERE job_id = ? AND status = 'PENDING'",
            (_iso(next_run_at), job_id),
        )

    async def yield_for_foreground(
        self,
        job_id: str,
        checkpoint: Optional[dict] = None,
        *,
        resume_after: datetime | None = None,
        owner_id: Optional[str] = None,
    ) -> bool:
        """Cooperatively pause a claimed job without consuming a retry.

        The claim incremented ``attempts``; the pause decrements it, clears the
        lease/owner and returns the job to PENDING. The checkpoint is merged,
        so the next attempt resumes instead of repeating expensive work. This
        is deliberately *not* a failure and is not counted as an automatic
        retry.

        With ``owner_id`` both the checkpoint and the state transition are
        conditional on a live lease held by this worker. A stale worker cannot
        turn a cancelled, completed, or newly-reclaimed job back into PENDING.
        """
        from companion.domain.models import utcnow

        now = utcnow()
        async with self.db.transaction(immediate=True):
            if checkpoint:
                saved = await self.save_checkpoint(
                    job_id, checkpoint, owner_id=owner_id
                )
                if not saved:
                    return False
            sql = (
                "UPDATE processing_jobs SET status = 'PENDING',"
                " attempts = CASE WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,"
                " started_at = NULL, owner_id = NULL, lease_expires_at = NULL,"
                " heartbeat_at = NULL, last_error = NULL, next_run_at = ?"
                " WHERE job_id = ? AND status = 'RUNNING' AND cancel_requested = 0"
            )
            params: tuple = (
                _iso(resume_after) if resume_after is not None else None,
                job_id,
            )
            if owner_id is not None:
                sql += (
                    " AND owner_id = ? AND lease_expires_at IS NOT NULL"
                    " AND lease_expires_at > ?"
                )
                params = (*params, owner_id, _iso(now))
            cur = await self.db.execute(sql, params)
            return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def fail(self, job_id: str, error: str, result: Any = None) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE processing_jobs SET status = 'FAILED', last_error = ?,"
            " result_json = COALESCE(?, result_json),"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE job_id = ?",
            (
                error,
                json.dumps(result, ensure_ascii=False, default=str)
                if result is not None
                else None,
                _iso(utcnow()),
                job_id,
            ),
        )

    async def fail_owned(
        self, job_id: str, owner_id: str, error: str, result: Any = None
    ) -> bool:
        """Record a failure only for a RUNNING job we still own."""
        from companion.domain.models import utcnow

        now = utcnow()
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'FAILED', last_error = ?,"
            " result_json = COALESCE(?, result_json),"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE job_id = ? AND status = 'RUNNING' AND owner_id = ?"
            " AND lease_expires_at IS NOT NULL AND lease_expires_at > ?",
            (
                error,
                json.dumps(result, ensure_ascii=False, default=str)
                if result is not None
                else None,
                _iso(now),
                job_id,
                owner_id,
                _iso(now),
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def recover_running(self, now: datetime | None = None) -> int:
        """Reset *expired* ``RUNNING`` jobs to ``PENDING``.

        Only jobs whose lease has lapsed (or was never set, i.e. jobs claimed
        by an older build) are reset. A live worker holding an unexpired lease
        is left untouched, so starting one worker never steals another
        worker's in-flight job. Idempotent handlers guarantee this never
        duplicates side effects.
        """
        from companion.domain.models import utcnow

        if now is None:
            now = utcnow()
        # A cancellation requested while a worker was running is final once the
        # lease has lapsed: do not resurrect it as PENDING.
        await self.db.execute(
            "UPDATE processing_jobs SET status = 'CANCELLED', completed_at = ?,"
            " owner_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL"
            " WHERE status = 'RUNNING' AND cancel_requested = 1"
            " AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
            (_iso(now), _iso(now)),
        )
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'PENDING', owner_id = NULL,"
            " lease_expires_at = NULL, heartbeat_at = NULL, started_at = NULL"
            " WHERE status = 'RUNNING'"
            " AND cancel_requested = 0"
            " AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
            (_iso(now),),
        )
        return cur.rowcount if hasattr(cur, "rowcount") else 0

    async def requeue(self, job_id: str, error: str) -> None:
        """Requeue a job for automatic retry, keeping its attempt count."""
        await self.db.execute(
            "UPDATE processing_jobs SET status = 'PENDING', last_error = ?,"
            " started_at = NULL, owner_id = NULL, lease_expires_at = NULL,"
            " heartbeat_at = NULL WHERE job_id = ?",
            (error, job_id),
        )

    async def requeue_owned(self, job_id: str, owner_id: str, error: str) -> bool:
        """Requeue only a RUNNING job still owned under a live lease."""
        from companion.domain.models import utcnow

        now = utcnow()
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'PENDING', last_error = ?,"
            " started_at = NULL, owner_id = NULL, lease_expires_at = NULL,"
            " heartbeat_at = NULL WHERE job_id = ? AND status = 'RUNNING'"
            " AND owner_id = ? AND cancel_requested = 0"
            " AND lease_expires_at IS NOT NULL AND lease_expires_at > ?",
            (error, job_id, owner_id, _iso(now)),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def retry(self, job_id: str) -> bool:
        """Requeue a *failed* job for a clean manual retry.

        Only ``FAILED`` jobs may be retried: a ``RUNNING`` job may be owned by
        another worker, and a ``COMPLETED`` job may already have produced
        external side effects, so neither is blindly reset.
        """
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'PENDING', last_error = NULL,"
            " started_at = NULL, completed_at = NULL, attempts = 0,"
            " owner_id = NULL, lease_expires_at = NULL, heartbeat_at = NULL"
            " WHERE job_id = ? AND status = 'FAILED'",
            (job_id,),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    def _to_job(self, row) -> ProcessingJob:
        raw_type = row["job_type"]
        try:
            job_type = JobType(raw_type)
        except ValueError:  # A registered V0.9 task type not in the enum yet.
            job_type = raw_type
        return ProcessingJob(
            job_id=row["job_id"],
            job_key=row["job_key"],
            job_type=job_type,
            payload=json.loads(row["payload"]),
            status=JobStatus(row["status"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            last_error=row["last_error"],
            created_at=_dt(row["created_at"]),
            started_at=_dt(row["started_at"]),
            completed_at=_dt(row["completed_at"]),
            owner_id=row["owner_id"],
            lease_expires_at=_dt(row["lease_expires_at"]),
            heartbeat_at=_dt(row["heartbeat_at"]),
            next_run_at=_dt(row["next_run_at"]),
            priority=row["priority"] or "NORMAL",
            source_kind=row["source_kind"] or "SYSTEM",
            source_ref=row["source_ref"],
            execution_mode=row["execution_mode"] or "BACKGROUND",
            checkpoint=json.loads(row["checkpoint_json"])
            if row["checkpoint_json"]
            else {},
            cancel_requested=bool(row["cancel_requested"]),
            result=json.loads(row["result_json"])
            if row["result_json"] is not None
            else None,
        )


class ExtractionRepository:
    """Bookkeeping for the lightweight background event-extraction job.

    A row exists once an event has been handed to the extraction pass. The row
    is the idempotency marker: a retried or restarted job sees it and does not
    call the model again. Only ``local_fast_path`` events are candidates;
    deliberation turns already have a full, locally validated analysis.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    async def get(self, event_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT event_id, status, candidate_count, created_at, processed_at"
            " FROM turn_extractions WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def pending_events(
        self,
        conversation_id: str,
        person_id: str,
        *,
        limit: int = 5,
    ) -> list[str]:
        """Eligible events for one speaker, oldest first.

        The speaker scope is deliberate: one batch must never mix different
        people's private material.
        """
        cur = await self.db.execute(
            "SELECT u.event_id FROM utterances u"
            " JOIN cognitive_analyses a ON a.event_id = u.event_id"
            " LEFT JOIN turn_extractions x ON x.event_id = u.event_id"
            " WHERE u.conversation_id = ? AND u.speaker_person_id = ?"
            "   AND u.processing_status = ?"
            "   AND a.provider = ?"
            "   AND x.event_id IS NULL"
            " ORDER BY u.occurred_at ASC, u.event_id ASC"
            " LIMIT ?",
            (
                conversation_id,
                person_id,
                ProcessingStatus.COMPLETED.value,
                LOCAL_PROVIDER,
                limit,
            ),
        )
        rows = await cur.fetchall()
        return [row["event_id"] for row in rows]

    async def mark(
        self, event_id: str, status: str, candidate_count: int = 0
    ) -> None:
        now = utcnow().isoformat()
        await self.db.execute(
            "INSERT OR IGNORE INTO turn_extractions("
            " event_id, status, candidate_count, created_at, processed_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (event_id, status, candidate_count, now, now),
        )
        # A retry may mark a previously skipped row as processed; the first
        # terminal status wins, so the row is never re-opened.
        await self.db.execute(
            "UPDATE turn_extractions SET candidate_count = ?, processed_at = ?"
            " WHERE event_id = ? AND status = ?",
            (candidate_count, now, event_id, status),
        )


@dataclass
class Repositories:
    """Convenience container holding every repository for a single database."""

    db: Database
    person: PersonRepository
    conversation: ConversationRepository
    utterance: UtteranceRepository
    analysis: AnalysisRepository
    memory_fact: MemoryFactRepository
    job: JobRepository
    settings: "SettingsRepository"
    audit: "AuditRepository"
    research: "ResearchRepository"
    research_state: ResearchStateRepository
    learning: LearningRepository
    goals: GoalRepository
    runtime_snapshot: RuntimeSnapshotRepository
    extraction: ExtractionRepository
    profile: PersonProfileRepository
    routing: RoutingRepository

    async def persist_cognitive_outcome(
        self,
        analysis: CognitiveResult,
        analysis_id: str,
        *,
        candidates: list[MemoryCandidate],
        job_key: str | None,
        job_payload: dict,
        persona: dict | None = None,
    ) -> str | None:
        """Atomically persist an analysis, its memory candidates and the job.

        This is one short, local transaction: either the analysis, all the
        given candidates and the deduplicated memory job are all committed, or
        none are. It never spans an external provider call.
        """
        from companion.domain.enums import JobType

        async with self.db.transaction():
            await self.analysis.save_analysis(
                analysis, analysis_id, persona=persona
            )
            for cand in candidates:
                await self.analysis.save_candidate(cand)
            if job_key is None:
                return None
            return await self.job.enqueue(
                JobType.MEMORY_CONSOLIDATION, job_key, job_payload
            )


def make_repositories(db: Database) -> Repositories:
    return Repositories(
        db=db,
        person=PersonRepository(db),
        conversation=ConversationRepository(db),
        utterance=UtteranceRepository(db),
        analysis=AnalysisRepository(db),
        memory_fact=MemoryFactRepository(db),
        job=JobRepository(db),
        settings=SettingsRepository(db),
        audit=AuditRepository(db),
        research=ResearchRepository(db),
        research_state=ResearchStateRepository(db),
        learning=LearningRepository(db),
        goals=GoalRepository(db),
        runtime_snapshot=RuntimeSnapshotRepository(db),
        extraction=ExtractionRepository(db),
        profile=PersonProfileRepository(db),
        routing=RoutingRepository(db),
    )


class SettingsRepository:
    """Key/value application settings (idle, research, web search toggles)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def get(self, key: str, default: str | None = None) -> str | None:
        cur = await self.db.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        )
        row = await cur.fetchone()
        return row["value"] if row is not None else default

    async def get_bool(self, key: str, default: bool = False) -> bool:
        value = await self.get(key)
        if value is None:
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    async def set(self, key: str, value: str) -> None:
        await self.db.execute(
            "INSERT INTO app_settings(key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, value, _iso(utcnow())),
        )

    async def all(self) -> dict[str, str]:
        cur = await self.db.execute(
            "SELECT key, value FROM app_settings ORDER BY key ASC"
        )
        return {r["key"]: r["value"] for r in await cur.fetchall()}


class AuditRepository:
    """Append-only audit trail for cross-entity activity.

    Rows are written (never updated or deleted) by the memory service and the
    person registry, and surfaced on the Activity/Audit Log page.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    async def record(
        self,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        from companion.domain.models import new_id

        await self.db.execute(
            "INSERT INTO audit_log(audit_id, actor, action, entity_type, entity_id,"
            " detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                new_id("audit"),
                actor,
                action,
                entity_type,
                entity_id,
                detail,
                _iso(utcnow()),
            ),
        )

    async def list(
        self, limit: int = 200, entity_type: str | None = None, actor: str | None = None
    ) -> list[dict]:
        sql = (
            "SELECT audit_id, actor, action, entity_type, entity_id, detail, created_at"
            " FROM audit_log"
        )
        conditions: list[str] = []
        params: list = []
        if entity_type is not None:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        if actor is not None:
            conditions.append("actor = ?")
            params.append(actor)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cur = await self.db.execute(sql, tuple(params))
        return [dict(r) for r in await cur.fetchall()]


class ResearchRepository:
    """Durable store for idle research tasks (populated by the Phase D engine).

    Tasks flow ``PENDING -> RUNNING -> COMPLETED`` or ``PENDING -> RUNNING ->
    FAILED``. Claiming is atomic and lease-gated (mirroring ``JobRepository``)
    so any number of workers may compete safely; a crashed worker's expired
    lease is reclaimed by :meth:`recover_running`. ``memory_id`` is the
    idempotency key of the consolidated memory: a retry that finds it already
    set re-uses the existing memory instead of creating a duplicate.
    """

    _COLUMNS = (
        "task_id, question, status, priority, source_kind, source_ref, result,"
        " error, created_at, started_at, completed_at, owner_id, lease_expires_at,"
        " attempts, max_attempts, memory_id, project_id, run_id, dedupe_key,"
        " source_memory_ids, origin"
    )

    def __init__(self, db: Database) -> None:
        self.db = db

    async def create(
        self,
        question: str,
        *,
        source_kind: str = "curiosity",
        source_ref: str | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        origin: str = "legacy",
        source_memory_ids: list[str] | tuple[str, ...] = (),
    ) -> str:
        """Insert one research row unconditionally.

        This low-level path is retained for tests and data migration. New
        scheduling code uses :meth:`create_unique`, which also normalizes the
        question and records the dedupe key/provenance.
        """
        from companion.domain.models import new_id

        task_id = new_id("research")
        await self.db.execute(
            "INSERT INTO research_tasks(task_id, question, status, priority,"
            " source_kind, source_ref, max_attempts, created_at, origin,"
            " source_memory_ids)"
            " VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                question,
                priority,
                source_kind,
                source_ref,
                max_attempts,
                _iso(utcnow()),
                origin,
                json.dumps(sorted({str(i) for i in source_memory_ids}), ensure_ascii=False),
            ),
        )
        return task_id

    async def create_unique(
        self,
        question: str,
        *,
        origin: str,
        source_kind: str = "curiosity",
        source_memory_ids: list[str] | tuple[str, ...] = (),
        source_ref: str | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        max_active: int | None = None,
        recheck_seconds: int | None = None,
    ) -> tuple[str, bool, str]:
        """Create a research record unless the same need is already covered.

        Returns ``(task_id, created, reason)``. A duplicate active row (or a
        recently completed row, unless the caller asks for re-research) is
        returned with ``created=False`` and the matching reason; the caller
        must not enqueue another job for it. No LLM is involved.

        ``max_active`` bounds *unfinished autonomous research rows*, not jobs:
        a research detail row without a job still occupies its slot. User
        submitted research (``origin='user'``) is authoritative and is never
        rejected by this autonomous limit.
        """
        from companion.domain.models import new_id

        source_ids = sorted({str(item) for item in source_memory_ids if str(item)})
        key = research_dedupe_key(question, source_ids, origin=origin)
        autonomous = str(origin) != "user"

        cur = await self.db.execute(
            "SELECT task_id FROM research_tasks"
            " WHERE dedupe_key = ? AND status IN ('PENDING', 'RUNNING')"
            " ORDER BY created_at ASC LIMIT 1",
            (key,),
        )
        existing = await cur.fetchone()
        if existing is not None:
            return str(existing["task_id"]), False, "active_duplicate"

        if recheck_seconds is not None and recheck_seconds > 0 and autonomous:
            cutoff = _iso(utcnow() - timedelta(seconds=int(recheck_seconds)))
            cur = await self.db.execute(
                "SELECT task_id FROM research_tasks"
                " WHERE dedupe_key = ? AND status = 'COMPLETED'"
                " AND completed_at IS NOT NULL AND completed_at >= ?"
                " ORDER BY completed_at DESC LIMIT 1",
                (key, cutoff),
            )
            fresh = await cur.fetchone()
            if fresh is not None:
                return str(fresh["task_id"]), False, "recently_completed"

        # Capacity is checked only when this call would create a *new*
        # unfinished research row; an idempotent duplicate above is free.
        if autonomous and max_active is not None:
            cur = await self.db.execute(
                "SELECT COUNT(*) AS n FROM research_tasks"
                " WHERE status IN ('PENDING', 'RUNNING') AND origin != 'user'"
            )
            active = int((await cur.fetchone())["n"])
            if active >= int(max_active):
                raise ResearchCapacityError(active, int(max_active))

        task_id = new_id("research")
        try:
            await self.db.execute(
                "INSERT INTO research_tasks(task_id, question, status, priority,"
                " source_kind, source_ref, max_attempts, created_at, dedupe_key,"
                " source_memory_ids, origin)"
                " VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    question,
                    priority,
                    source_kind,
                    source_ref,
                    max_attempts,
                    _iso(utcnow()),
                    key,
                    json.dumps(source_ids, ensure_ascii=False),
                    str(origin),
                ),
            )
        except sqlite3.IntegrityError:
            # A concurrent creator won the partial-unique race. Re-read its
            # row instead of surfacing a constraint error.
            cur = await self.db.execute(
                "SELECT task_id FROM research_tasks"
                " WHERE dedupe_key = ? AND status IN ('PENDING', 'RUNNING')"
                " LIMIT 1",
                (key,),
            )
            row = await cur.fetchone()
            if row is not None:
                return str(row["task_id"]), False, "active_duplicate"
            raise
        return task_id, True, "created"

    async def active_count(self, *, autonomous_only: bool = True) -> int:
        sql = "SELECT COUNT(*) AS n FROM research_tasks WHERE status IN ('PENDING','RUNNING')"
        if autonomous_only:
            sql += " AND origin != 'user'"
        cur = await self.db.execute(sql)
        return int((await cur.fetchone())["n"])

    async def get(self, task_id: str) -> dict | None:
        cur = await self.db.execute(
            f"SELECT {self._COLUMNS} FROM research_tasks WHERE task_id = ?",
            (task_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        sql = f"SELECT {self._COLUMNS} FROM research_tasks"
        params: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY priority DESC, created_at ASC LIMIT ?"
        params = (*params, limit)
        cur = await self.db.execute(sql, params)
        return [dict(r) for r in await cur.fetchall()]

    async def claim(
        self, task_id: str, owner_id: str = "idle", lease_seconds: int = 300
    ) -> bool:
        """Atomically claim a ``PENDING`` task, returning ``True`` on success.

        The ``WHERE status = 'PENDING' AND attempts < max_attempts`` guard plus
        SQLite's write lock mean at most one of any number of concurrent
        workers observes ``rowcount == 1``. The claim also records a time-limited
        lease so a crashed worker's in-flight task can be reclaimed.
        """
        from companion.domain.models import utcnow

        now = utcnow()
        lease = now + timedelta(seconds=lease_seconds) if lease_seconds else None
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'RUNNING', attempts = attempts + 1,"
            " owner_id = ?, lease_expires_at = ?, started_at = COALESCE(started_at, ?),"
            " error = NULL WHERE task_id = ? AND status = 'PENDING'"
            " AND attempts < max_attempts",
            (
                owner_id,
                _iso(lease) if lease is not None else None,
                _iso(now),
                task_id,
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    async def recover_running(self, now: datetime | None = None) -> int:
        """Reset *expired* ``RUNNING`` tasks back to ``PENDING``.

        A live worker holding an unexpired lease is left untouched. Idempotent
        consolidation (via ``memory_id``) guarantees this never duplicates a
        memory side effect.
        """
        from companion.domain.models import utcnow

        if now is None:
            now = utcnow()
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'PENDING', owner_id = NULL,"
            " lease_expires_at = NULL, started_at = NULL"
            " WHERE status = 'RUNNING'"
            " AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
            (_iso(now),),
        )
        return cur.rowcount if hasattr(cur, "rowcount") else 0

    async def recover_task(
        self, task_id: str, now: datetime | None = None
    ) -> bool:
        """Reclaim one expired ``RUNNING`` task without touching its peers."""
        from companion.domain.models import utcnow

        if now is None:
            now = utcnow()
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'PENDING', owner_id = NULL,"
            " lease_expires_at = NULL, started_at = NULL"
            " WHERE task_id = ? AND status = 'RUNNING'"
            " AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
            (task_id, _iso(now)),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def lease_live(self, task_id: str, now: datetime | None = None) -> bool:
        """Whether a ``RUNNING`` task is protected by an unexpired lease."""
        from companion.domain.models import utcnow

        if now is None:
            now = utcnow()
        cur = await self.db.execute(
            "SELECT 1 FROM research_tasks WHERE task_id = ? AND status = 'RUNNING'"
            " AND lease_expires_at IS NOT NULL AND lease_expires_at > ?",
            (task_id, _iso(now)),
        )
        return (await cur.fetchone()) is not None

    async def renew(
        self, task_id: str, owner_id: str, lease_seconds: int = 300
    ) -> bool:
        """Extend the research detail lease only for its current owner."""
        from companion.domain.models import utcnow

        now = utcnow()
        lease = now + timedelta(seconds=lease_seconds) if lease_seconds else None
        cur = await self.db.execute(
            "UPDATE research_tasks SET lease_expires_at = ?"
            " WHERE task_id = ? AND status = 'RUNNING' AND owner_id = ?",
            (_iso(lease) if lease is not None else None, task_id, owner_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def fail(self, task_id: str, error: str) -> None:
        await self.db.execute(
            "UPDATE research_tasks SET status = 'FAILED', error = ?, completed_at = ?,"
            " owner_id = NULL, lease_expires_at = NULL WHERE task_id = ?",
            (error, _iso(utcnow()), task_id),
        )

    async def fail_owned(
        self, task_id: str, owner_id: str, error: str
    ) -> bool:
        """Mark FAILED only for the RUNNING task this execution still owns."""
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'FAILED', error = ?,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE task_id = ? AND status = 'RUNNING' AND owner_id = ?",
            (error, _iso(utcnow()), task_id, owner_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def requeue(self, task_id: str, error: str) -> None:
        """Requeue a task for automatic retry, keeping its attempt count."""
        await self.db.execute(
            "UPDATE research_tasks SET status = 'PENDING', error = ?, started_at = NULL,"
            " owner_id = NULL, lease_expires_at = NULL WHERE task_id = ?",
            (error, task_id),
        )

    async def requeue_owned(
        self, task_id: str, owner_id: str, error: str
    ) -> bool:
        """Requeue only the RUNNING task this execution still owns."""
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'PENDING', error = ?,"
            " started_at = NULL, owner_id = NULL, lease_expires_at = NULL"
            " WHERE task_id = ? AND status = 'RUNNING' AND owner_id = ?",
            (error, task_id, owner_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def retry(self, task_id: str) -> bool:
        """Reset a *failed* task for a clean manual retry."""
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'PENDING', error = NULL,"
            " started_at = NULL, completed_at = NULL, attempts = 0, owner_id = NULL,"
            " lease_expires_at = NULL WHERE task_id = ? AND status = 'FAILED'",
            (task_id,),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def set_status(self, task_id: str, status: str, error: str | None = None) -> None:
        await self.db.execute(
            "UPDATE research_tasks SET status = ?, error = ?, started_at ="
            " COALESCE(started_at, ?) WHERE task_id = ?",
            (status, error, _iso(utcnow()), task_id),
        )

    async def complete(self, task_id: str, result: str, memory_id: str | None = None) -> None:
        await self.db.execute(
            "UPDATE research_tasks SET status = 'COMPLETED', result = ?,"
            " memory_id = COALESCE(?, memory_id), completed_at = ? WHERE task_id = ?",
            (result, memory_id, _iso(utcnow()), task_id),
        )

    async def complete_owned(
        self,
        task_id: str,
        owner_id: str,
        result: str,
        memory_id: str | None = None,
    ) -> bool:
        """Mark COMPLETED only while this execution still owns the task.

        This is the detail-row half of an atomic research commit: if the token
        changed, the caller's memory write is rolled back with its transaction.
        """
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'COMPLETED', result = ?,"
            " memory_id = COALESCE(?, memory_id), completed_at = ?,"
            " owner_id = NULL, lease_expires_at = NULL"
            " WHERE task_id = ? AND status = 'RUNNING' AND owner_id = ?",
            (result, memory_id, _iso(utcnow()), task_id, owner_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def cancel_owned(self, task_id: str, owner_id: str) -> bool:
        """Mark a task CANCELLED only for its current execution owner."""
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'CANCELLED', error = ?,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE task_id = ? AND status = 'RUNNING' AND owner_id = ?",
            ("cancelled", _iso(utcnow()), task_id, owner_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

    async def cancel_for_job(self, task_id: str, owner_id: str) -> bool:
        """Cancel a research detail row owned by (or not yet claimed by) a job.

        A job may be cancelled before its handler ever claims the research
        task, so PENDING is included. A RUNNING row owned by a *different*
        live executor is left alone: that executor decides its own fate.
        """
        cur = await self.db.execute(
            "UPDATE research_tasks SET status = 'CANCELLED', error = ?,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE task_id = ?"
            " AND (status = 'PENDING' OR (status = 'RUNNING' AND owner_id = ?))",
            ("cancelled", _iso(utcnow()), task_id, owner_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) > 0

