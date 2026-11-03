"""SQLite schema (V0.0).

The schema is idempotent: every ``CREATE ... IF NOT EXISTS`` statement is safe
to run against an existing database and never drops or truncates data.

``memory_facts`` is intentionally minimal — V0.0 stores only hand-authored
facts. Future multi-subject associations (e.g. a fact relating several people)
will be added as a join table rather than by widening ``subject_person_id``;
see ``docs/DATA_MODEL.md``.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

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
MIGRATIONS: list[tuple[int, str]] = [
    (1, SCHEMA_SQL),
]
