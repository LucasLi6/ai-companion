"""Data-access repositories over :class:`Database`.

Each repository is a thin, parameterised-SQL layer. No business rules live
here — those belong in the registry, orchestrator and policy layers. All
timestamps are stored as timezone-aware UTC ISO-8601 strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

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
    MemoryCandidate,
    NonresponseMotive,
    Person,
    ProcessingJob,
    UtteranceEvent,
    utcnow,
)

from companion.cognition.nonresponse import NonresponseTrace
from companion.learning.state import LearningRepository
from companion.research.state import ResearchStateRepository
from companion.goals.state import GoalRepository

from .database import Database


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


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

    async def recent_context(
        self,
        conversation_id: str,
        exclude_event_id: str | None,
        limit: int = 20,
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


class AnalysisRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def save_analysis(self, analysis: CognitiveResult, analysis_id: str) -> None:
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
        await self.db.execute(
            "INSERT INTO cognitive_analyses(analysis_id, event_id, meaning, action,"
            " reason_code, provider, referenced_memory_ids, response_style, created_at,"
            " intent, addressed_to, silence_requested, reply_necessity, conversation_state,"
            " affective_signal_json, nonresponse_motive_json, scene_proposal_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )

    async def get_analysis(self, event_id: str) -> CognitiveResult | None:
        cur = await self.db.execute(
            "SELECT analysis_id, event_id, meaning, action, reason_code, provider,"
            " referenced_memory_ids, response_style, intent, addressed_to,"
            " silence_requested, reply_necessity, conversation_state,"
            " affective_signal_json, nonresponse_motive_json, scene_proposal_json"
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
        )

    async def get_analysis_id(self, event_id: str) -> Optional[str]:
        cur = await self.db.execute(
            "SELECT analysis_id FROM cognitive_analyses WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        return row["analysis_id"] if row is not None else None

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
        " lease_expires_at, heartbeat_at, next_run_at"
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
    ) -> str:
        """Enqueue a job, deduplicating on ``job_key``.

        Returns the *existing* job id if ``job_key`` is already present, so
        re-submitting the same event never creates a second job.
        """
        existing = await self.get_by_key(job_key)
        if existing is not None:
            return existing.job_id
        from companion.domain.models import new_id, utcnow

        job_id = new_id("job")
        await self.db.execute(
            "INSERT INTO processing_jobs(job_id, job_key, job_type, payload, status,"
            " attempts, max_attempts, created_at, next_run_at)"
            " VALUES (?, ?, ?, ?, 'PENDING', 0, ?, ?, ?)",
            (
                job_id,
                job_key,
                job_type.value,
                json.dumps(payload),
                max_attempts,
                _iso(utcnow()),
                _iso(next_run_at) if next_run_at is not None else None,
            ),
        )
        return job_id

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

    async def pending(self, limit: int = 100) -> list[ProcessingJob]:
        cur = await self.db.execute(
            f"SELECT {self._COLUMNS} FROM processing_jobs"
            " WHERE status = 'PENDING' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [self._to_job(r) for r in rows]

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
            " WHERE job_id = ? AND status = 'PENDING' AND attempts < max_attempts",
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

    async def complete(self, job_id: str) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE processing_jobs SET status = 'COMPLETED', completed_at = ?,"
            " owner_id = NULL, lease_expires_at = NULL WHERE job_id = ?",
            (_iso(utcnow()), job_id),
        )

    async def fail(self, job_id: str, error: str) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE processing_jobs SET status = 'FAILED', last_error = ?,"
            " completed_at = ?, owner_id = NULL, lease_expires_at = NULL"
            " WHERE job_id = ?",
            (error, _iso(utcnow()), job_id),
        )

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
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'PENDING', owner_id = NULL,"
            " lease_expires_at = NULL, heartbeat_at = NULL, started_at = NULL"
            " WHERE status = 'RUNNING'"
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
        return ProcessingJob(
            job_id=row["job_id"],
            job_key=row["job_key"],
            job_type=JobType(row["job_type"]),
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

    async def persist_cognitive_outcome(
        self,
        analysis: CognitiveResult,
        analysis_id: str,
        *,
        candidates: list[MemoryCandidate],
        job_key: str | None,
        job_payload: dict,
    ) -> str | None:
        """Atomically persist an analysis, its memory candidates and the job.

        This is one short, local transaction: either the analysis, all the
        given candidates and the deduplicated memory job are all committed, or
        none are. It never spans an external provider call.
        """
        from companion.domain.enums import JobType

        async with self.db.transaction():
            await self.analysis.save_analysis(analysis, analysis_id)
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
        " attempts, max_attempts, memory_id, project_id, run_id"
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
    ) -> str:
        from companion.domain.models import new_id

        task_id = new_id("research")
        await self.db.execute(
            "INSERT INTO research_tasks(task_id, question, status, priority,"
            " source_kind, source_ref, max_attempts, created_at)"
            " VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?)",
            (task_id, question, priority, source_kind, source_ref, max_attempts, _iso(utcnow())),
        )
        return task_id

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

    async def fail(self, task_id: str, error: str) -> None:
        await self.db.execute(
            "UPDATE research_tasks SET status = 'FAILED', error = ?, completed_at = ?,"
            " owner_id = NULL, lease_expires_at = NULL WHERE task_id = ?",
            (error, _iso(utcnow()), task_id),
        )

    async def requeue(self, task_id: str, error: str) -> None:
        """Requeue a task for automatic retry, keeping its attempt count."""
        await self.db.execute(
            "UPDATE research_tasks SET status = 'PENDING', error = ?, started_at = NULL,"
            " owner_id = NULL, lease_expires_at = NULL WHERE task_id = ?",
            (error, task_id),
        )

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
