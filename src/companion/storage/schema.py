"""SQLite schema (V0.0).

The schema is idempotent: every ``CREATE ... IF NOT EXISTS`` statement is safe
to run against an existing database and never drops or truncates data.

``memory_facts`` is intentionally minimal — V0.0 stores only hand-authored
facts. Future multi-subject associations (e.g. a fact relating several people)
will be added as a join table rather than by widening ``subject_person_id``;
see ``docs/DATA_MODEL.md``.
"""

from __future__ import annotations

SCHEMA_VERSION = 4

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    person_id       TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    role            TEXT NOT NULL,
    registration_status TEXT NOT NULL DEFAULT 'active',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS consents (
    person_id              TEXT PRIMARY KEY REFERENCES persons(person_id) ON DELETE CASCADE,
    allow_processing       INTEGER NOT NULL,
    allow_persistence      INTEGER NOT NULL,
    allow_memory_learning  INTEGER NOT NULL,
    updated_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  TEXT PRIMARY KEY,
    quiet_mode       INTEGER NOT NULL DEFAULT 0,
    last_action      TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS utterances (
    event_id          TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL REFERENCES conversations(conversation_id),
    speaker_person_id TEXT REFERENCES persons(person_id),
    speaker_label     TEXT,
    text              TEXT NOT NULL,
    source            TEXT NOT NULL,
    addressee_hint    TEXT NOT NULL,
    occurred_at       TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

-- One analysis per event, enforced by UNIQUE(event_id).
CREATE TABLE IF NOT EXISTS cognitive_analyses (
    analysis_id           TEXT PRIMARY KEY,
    event_id              TEXT NOT NULL UNIQUE REFERENCES utterances(event_id) ON DELETE CASCADE,
    meaning               TEXT NOT NULL,
    action                TEXT NOT NULL,
    reason_code           TEXT NOT NULL,
    provider              TEXT NOT NULL,
    referenced_memory_ids TEXT NOT NULL DEFAULT '[]',
    response_style        TEXT NOT NULL,
    created_at            TEXT NOT NULL
);

-- One final action per event, enforced by UNIQUE(event_id).
CREATE TABLE IF NOT EXISTS agent_actions (
    action_id     TEXT PRIMARY KEY,
    event_id      TEXT NOT NULL UNIQUE REFERENCES utterances(event_id) ON DELETE CASCADE,
    action_type   TEXT NOT NULL,
    response_text TEXT,
    reason_code   TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_candidates (
    candidate_id      TEXT PRIMARY KEY,
    source_event_id   TEXT NOT NULL REFERENCES utterances(event_id) ON DELETE CASCADE,
    subject_person_id TEXT REFERENCES persons(person_id),
    content           TEXT NOT NULL,
    operation         TEXT NOT NULL,
    status            TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

-- Reserved for future long-term facts; V0.0 never auto-populates it.
CREATE TABLE IF NOT EXISTS memory_facts (
    fact_id           TEXT PRIMARY KEY,
    subject_person_id TEXT REFERENCES persons(person_id),
    content           TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'manual',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- Durable, deduplicated, recoverable background jobs.
CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id       TEXT PRIMARY KEY,
    job_key      TEXT NOT NULL UNIQUE,
    job_type     TEXT NOT NULL,
    payload      TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    last_error   TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_utterances_conversation ON utterances(conversation_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_utterances_speaker ON utterances(speaker_person_id);
CREATE INDEX IF NOT EXISTS idx_candidates_event ON memory_candidates(source_event_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON processing_jobs(status);
"""

# Ordered, append-only migrations. Each entry is ``(version, sql)`` and is
# applied once, in order, by ``Database.initialize``. Never edit an existing
# entry — append a new one with the next version number. V0.0 = version 1.
#
# Each migration's SQL must be plain DDL/DML (no ``PRAGMA`` and no string
# literal containing a semicolon) so ``Database.initialize`` can split it into
# statements and run them inside a single transaction (see A2: migration
# atomicity).
MIGRATIONS: list[tuple[int, str]] = [
    (1, SCHEMA_SQL),
    # Version 2 (Phase A): job lease / ownership columns for safe multi-worker
    # claiming, time-limited leases and scheduled runs. All are NULL-able so
    # existing rows keep their old (lease-less) semantics.
    (
        2,
        """
        ALTER TABLE processing_jobs ADD COLUMN owner_id TEXT;
        ALTER TABLE processing_jobs ADD COLUMN lease_expires_at TEXT;
        ALTER TABLE processing_jobs ADD COLUMN heartbeat_at TEXT;
        ALTER TABLE processing_jobs ADD COLUMN next_run_at TEXT;
        """,
    ),
    # Version 3 (Phase B): the unified memory foundation. ``memory_facts`` and
    # the legacy ``memory_candidates`` table are left intact; existing facts are
    # copied into ``memory_items`` once, idempotently (this migration runs at
    # most once because it is version-gated).
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS memory_items (
            memory_id            TEXT PRIMARY KEY,
            kind                 TEXT NOT NULL,
            content              TEXT NOT NULL,
            status               TEXT NOT NULL,
            visibility           TEXT NOT NULL,
            created_by           TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            updated_at           TEXT NOT NULL,
            version              INTEGER NOT NULL DEFAULT 1,
            valid_from           TEXT,
            valid_until          TEXT,
            human_confirmed      INTEGER NOT NULL DEFAULT 0,
            source_supported     INTEGER NOT NULL DEFAULT 0,
            self_reported        INTEGER NOT NULL DEFAULT 0,
            third_party_reported INTEGER NOT NULL DEFAULT 0,
            model_inferred       INTEGER NOT NULL DEFAULT 0,
            extra                TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memory_subjects (
            memory_id    TEXT NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            subject_type TEXT NOT NULL,
            subject_id   TEXT NOT NULL,
            PRIMARY KEY (memory_id, subject_type, subject_id)
        );

        CREATE TABLE IF NOT EXISTS memory_sources (
            source_id    TEXT PRIMARY KEY,
            source_type  TEXT NOT NULL,
            source_ref   TEXT,
            title        TEXT,
            url          TEXT,
            retrieved_at TEXT,
            published_at TEXT,
            extra        TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memory_evidence (
            memory_id          TEXT NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            source_id          TEXT NOT NULL REFERENCES memory_sources(source_id) ON DELETE CASCADE,
            evidence_span      TEXT,
            verification_level TEXT NOT NULL DEFAULT 'UNVERIFIED',
            retrieved_at       TEXT,
            PRIMARY KEY (memory_id, source_id)
        );

        CREATE TABLE IF NOT EXISTS memory_links (
            memory_id        TEXT NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            linked_memory_id TEXT NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            link_type        TEXT NOT NULL,
            PRIMARY KEY (memory_id, linked_memory_id, link_type)
        );

        CREATE TABLE IF NOT EXISTS memory_revisions (
            revision_id TEXT PRIMARY KEY,
            memory_id   TEXT NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            version     INTEGER NOT NULL,
            kind        TEXT NOT NULL,
            content     TEXT NOT NULL,
            status      TEXT NOT NULL,
            changed_by  TEXT NOT NULL,
            changed_at  TEXT NOT NULL,
            reason      TEXT,
            snapshot    TEXT NOT NULL DEFAULT '{}'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
            content,
            memory_id UNINDEXED,
            tokenize='unicode61'
        );

        CREATE INDEX IF NOT EXISTS idx_memory_items_kind ON memory_items(kind);
        CREATE INDEX IF NOT EXISTS idx_memory_items_status ON memory_items(status);
        CREATE INDEX IF NOT EXISTS idx_memory_items_visibility ON memory_items(visibility);
        CREATE INDEX IF NOT EXISTS idx_memory_subjects_lookup
            ON memory_subjects(subject_type, subject_id);

        -- Migrate existing hand-authored facts into the unified store (idempotent
        -- because this migration is applied exactly once). Old ids are preserved.
        INSERT OR IGNORE INTO memory_items(
            memory_id, kind, content, status, visibility, created_by, created_at,
            updated_at, version, human_confirmed, source_supported, self_reported,
            third_party_reported, model_inferred, extra
        )
        SELECT
            fact_id,
            'PERSONAL_FACT',
            content,
            CASE WHEN status = 'manual' THEN 'ACTIVE' ELSE 'CANDIDATE' END,
            'PRIVATE',
            COALESCE(subject_person_id, 'system'),
            created_at,
            updated_at,
            1,
            1, 0, 1, 0, 0,
            '{}'
        FROM memory_facts;

        INSERT OR IGNORE INTO memory_subjects(memory_id, subject_type, subject_id)
        SELECT fact_id, 'PERSON', subject_person_id
        FROM memory_facts
        WHERE subject_person_id IS NOT NULL;

        INSERT INTO memory_items_fts(content, memory_id)
        SELECT content, fact_id FROM memory_facts;
        """,
    ),
    # Version 4 (Phase C/D): management UI support and idle research.
    # - app_settings holds runtime configuration toggles (idle/research/web
    #   search), edited from the Settings page.
    # - audit_log is the append-only, cross-entity activity trail shown on the
    #   Activity/Audit Log page. Rows are written by the memory service and the
    #   person registry, never deleted.
    # - research_tasks is the durable store for the idle research planner
    #   (Phase D); the Research Manager page reads and schedules it.
    (
        4,
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id    TEXT PRIMARY KEY,
            actor       TEXT NOT NULL,
            action      TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id   TEXT,
            detail      TEXT,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS research_tasks (
            task_id      TEXT PRIMARY KEY,
            question     TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'PENDING',
            priority     INTEGER NOT NULL DEFAULT 0,
            source_kind  TEXT NOT NULL DEFAULT 'curiosity',
            source_ref   TEXT,
            result       TEXT,
            error        TEXT,
            created_at   TEXT NOT NULL,
            started_at   TEXT,
            completed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);
        CREATE INDEX IF NOT EXISTS idx_audit_log_entity ON audit_log(entity_type, entity_id);
        CREATE INDEX IF NOT EXISTS idx_research_tasks_status ON research_tasks(status);
        """,
    ),
]
