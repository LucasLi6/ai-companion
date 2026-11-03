"""Data-access repositories over :class:`Database`.

Each repository is a thin, parameterised-SQL layer. No business rules live
here — those belong in the registry, orchestrator and policy layers. All
timestamps are stored as timezone-aware UTC ISO-8601 strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from companion.domain.enums import (
    AddresseeHint,
    CognitiveAction,
    EventSource,
    JobStatus,
    JobType,
    MemoryCandidateStatus,
    MemoryOperation,
    PersonRole,
    ProcessingStatus,
    ReasonCode,
    RegistrationStatus,
)
from companion.domain.models import (
    AgentAction,
    CognitiveResult,
    Consent,
    ConversationContext,
    MemoryCandidate,
    Person,
    ProcessingJob,
    UtteranceEvent,
    utcnow,
)

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
                    " allow_memory_learning, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        consent.person_id,
                        int(consent.allow_processing),
                        int(consent.allow_persistence),
                        int(consent.allow_memory_learning),
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
            " allow_memory_learning, updated_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(person_id) DO UPDATE SET"
            " allow_processing=excluded.allow_processing,"
            " allow_persistence=excluded.allow_persistence,"
            " allow_memory_learning=excluded.allow_memory_learning,"
            " updated_at=excluded.updated_at",
            (
                consent.person_id,
                int(consent.allow_processing),
                int(consent.allow_persistence),
                int(consent.allow_memory_learning),
                _iso(consent.updated_at),
            ),
        )

    async def get_consent(self, person_id: str) -> Consent | None:
        cur = await self.db.execute(
            "SELECT person_id, allow_processing, allow_persistence,"
            " allow_memory_learning, updated_at FROM consents WHERE person_id = ?",
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

    async def recent(self, conversation_id: str, limit: int = 20) -> list[str]:
        cur = await self.db.execute(
            "SELECT event_id FROM utterances WHERE conversation_id = ?"
            " ORDER BY occurred_at DESC, event_id DESC LIMIT ?",
            (conversation_id, limit),
        )
        rows = await cur.fetchall()
        return [r["event_id"] for r in rows]

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


class AnalysisRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def save_analysis(self, analysis: CognitiveResult, analysis_id: str) -> None:
        await self.db.execute(
            "INSERT INTO cognitive_analyses(analysis_id, event_id, meaning, action,"
            " reason_code, provider, referenced_memory_ids, response_style, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )

    async def get_analysis(self, event_id: str) -> CognitiveResult | None:
        cur = await self.db.execute(
            "SELECT analysis_id, event_id, meaning, action, reason_code, provider,"
            " referenced_memory_ids, response_style FROM cognitive_analyses"
            " WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return CognitiveResult(
            event_id=row["event_id"],
            meaning=row["meaning"],
            action=CognitiveAction(row["action"]),
            reason_code=ReasonCode(row["reason_code"]),
            referenced_memory_ids=json.loads(row["referenced_memory_ids"]),
            response_style=row["response_style"],
            provider=row["provider"],
        )

    async def save_candidate(self, candidate: MemoryCandidate) -> None:
        await self.db.execute(
            "INSERT INTO memory_candidates(candidate_id, source_event_id,"
            " subject_person_id, content, operation, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.candidate_id,
                candidate.source_event_id,
                candidate.subject_person_id,
                candidate.content,
                candidate.operation.value,
                candidate.status.value,
                _iso(utcnow()),
            ),
        )

    async def get_candidates(self, event_id: str) -> list[MemoryCandidate]:
        cur = await self.db.execute(
            "SELECT candidate_id, source_event_id, subject_person_id, content,"
            " operation, status FROM memory_candidates WHERE source_event_id = ?",
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
            )
            for r in rows
        ]

    async def update_candidate_status(
        self, candidate_id: str, status: MemoryCandidateStatus
    ) -> None:
        await self.db.execute(
            "UPDATE memory_candidates SET status = ? WHERE candidate_id = ?",
            (status.value, candidate_id),
        )

    async def save_action(self, action: AgentAction) -> None:
        await self.db.execute(
            "INSERT INTO agent_actions(action_id, event_id, action_type, response_text,"
            " reason_code, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                action.action_id,
                action.event_id,
                action.action_type.value,
                action.response_text,
                action.reason_code.value,
                action.status,
                _iso(action.created_at),
            ),
        )

    async def get_action(self, event_id: str) -> AgentAction | None:
        cur = await self.db.execute(
            "SELECT action_id, event_id, action_type, response_text, reason_code,"
            " status, created_at FROM agent_actions WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return AgentAction(
            action_id=row["action_id"],
            event_id=row["event_id"],
            action_type=CognitiveAction(row["action_type"]),
            response_text=row["response_text"],
            reason_code=ReasonCode(row["reason_code"]),
            status=row["status"],
            created_at=_dt(row["created_at"]),
        )

    async def clear_outputs(self, event_id: str) -> None:
        """Remove prior partial cognitive outputs before a retry.

        Only ever called for an event that previously failed (same id, same
        content); never for a conflicting, different payload.
        """
        async with self.db.transaction():
            await self.db.conn.execute(
                "DELETE FROM memory_candidates WHERE source_event_id = ?", (event_id,)
            )
            await self.db.conn.execute(
                "DELETE FROM agent_actions WHERE event_id = ?", (event_id,)
            )
            await self.db.conn.execute(
                "DELETE FROM cognitive_analyses WHERE event_id = ?", (event_id,)
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
    def __init__(self, db: Database) -> None:
        self.db = db

    async def enqueue(
        self, job_type: JobType, job_key: str, payload: dict, max_attempts: int = 3
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
            " attempts, max_attempts, created_at) VALUES (?, ?, ?, ?, 'PENDING', 0, ?, ?)",
            (
                job_id,
                job_key,
                job_type.value,
                json.dumps(payload),
                max_attempts,
                _iso(utcnow()),
            ),
        )
        return job_id

    async def get(self, job_id: str) -> ProcessingJob | None:
        cur = await self.db.execute(
            "SELECT job_id, job_key, job_type, payload, status, attempts, max_attempts,"
            " last_error, created_at, started_at, completed_at FROM processing_jobs"
            " WHERE job_id = ?",
            (job_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._to_job(row)

    async def get_by_key(self, job_key: str) -> ProcessingJob | None:
        cur = await self.db.execute(
            "SELECT job_id, job_key, job_type, payload, status, attempts, max_attempts,"
            " last_error, created_at, started_at, completed_at FROM processing_jobs"
            " WHERE job_key = ?",
            (job_key,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return self._to_job(row)

    async def pending(self, limit: int = 100) -> list[ProcessingJob]:
        cur = await self.db.execute(
            "SELECT job_id, job_key, job_type, payload, status, attempts, max_attempts,"
            " last_error, created_at, started_at, completed_at FROM processing_jobs"
            " WHERE status = 'PENDING' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [self._to_job(r) for r in rows]

    async def list_jobs(self, status: JobStatus | None = None) -> list[ProcessingJob]:
        sql = (
            "SELECT job_id, job_key, job_type, payload, status, attempts, max_attempts,"
            " last_error, created_at, started_at, completed_at FROM processing_jobs"
        )
        params: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status.value,)
        sql += " ORDER BY created_at ASC"
        cur = await self.db.execute(sql, params)
        rows = await cur.fetchall()
        return [self._to_job(r) for r in rows]

    async def claim(self, job_id: str) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE processing_jobs SET status = 'RUNNING', attempts = attempts + 1,"
            " started_at = ?, last_error = NULL WHERE job_id = ?",
            (_iso(utcnow()), job_id),
        )

    async def complete(self, job_id: str) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE processing_jobs SET status = 'COMPLETED', completed_at = ?"
            " WHERE job_id = ?",
            (_iso(utcnow()), job_id),
        )

    async def fail(self, job_id: str, error: str) -> None:
        from companion.domain.models import utcnow

        await self.db.execute(
            "UPDATE processing_jobs SET status = 'FAILED', last_error = ?,"
            " completed_at = ? WHERE job_id = ?",
            (error, _iso(utcnow()), job_id),
        )

    async def recover_running(self) -> int:
        """Reset leftover ``RUNNING`` jobs to ``PENDING`` so they are retried.

        Idempotent handlers guarantee this never duplicates side effects.
        """
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'PENDING', started_at = NULL"
            " WHERE status = 'RUNNING'"
        )
        return cur.rowcount if hasattr(cur, "rowcount") else 0

    async def requeue(self, job_id: str, error: str) -> None:
        """Requeue a job for automatic retry, keeping its attempt count."""
        await self.db.execute(
            "UPDATE processing_jobs SET status = 'PENDING', last_error = ?,"
            " started_at = NULL WHERE job_id = ?",
            (error, job_id),
        )

    async def retry(self, job_id: str) -> bool:
        """Requeue a failed or completed-able job; reset attempts for a clean run."""
        cur = await self.db.execute(
            "UPDATE processing_jobs SET status = 'PENDING', last_error = NULL,"
            " started_at = NULL, completed_at = NULL, attempts = 0 WHERE job_id = ?",
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
        )


@dataclass
class Repositories:
    """Convenience container holding every repository for a single database."""

    person: PersonRepository
    conversation: ConversationRepository
    utterance: UtteranceRepository
    analysis: AnalysisRepository
    memory_fact: MemoryFactRepository
    job: JobRepository


def make_repositories(db: Database) -> Repositories:
    return Repositories(
        person=PersonRepository(db),
        conversation=ConversationRepository(db),
        utterance=UtteranceRepository(db),
        analysis=AnalysisRepository(db),
        memory_fact=MemoryFactRepository(db),
        job=JobRepository(db),
    )
