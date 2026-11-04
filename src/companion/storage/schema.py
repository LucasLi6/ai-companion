"""SQLite schema (V0.0).

The schema is idempotent: every ``CREATE ... IF NOT EXISTS`` statement is safe
to run against an existing database and never drops or truncates data.

``memory_facts`` is intentionally minimal — V0.0 stores only hand-authored
facts. Future multi-subject associations (e.g. a fact relating several people)
will be added as a join table rather than by widening ``subject_person_id``;
see ``docs/DATA_MODEL.md``.
"""

from __future__ import annotations

SCHEMA_VERSION = 16

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
    # Version 5 (stabilisation): privacy and research-reliability columns.
    # - consents.allow_external_ai is an *independent* grant for sending a
    #   person's private context to an external AI API. It defaults to 0 and is
    #   deliberately distinct from allow_memory_learning: learning stores data
    #   locally, external AI uploads it to a remote model.
    # - research_tasks gains the same lease/ownership columns as processing_jobs
    #   so the research engine can atomically claim, recover and idempotently
    #   complete a task, plus a ``memory_id`` idempotency key for its
    #   consolidated memory.
    (
        5,
        """
        ALTER TABLE consents ADD COLUMN allow_external_ai INTEGER NOT NULL DEFAULT 0;

        ALTER TABLE research_tasks ADD COLUMN owner_id TEXT;
        ALTER TABLE research_tasks ADD COLUMN lease_expires_at TEXT;
        ALTER TABLE research_tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE research_tasks ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 3;
        ALTER TABLE research_tasks ADD COLUMN memory_id TEXT;
        """,
    ),
    # Version 6 (stabilisation): demote mock research knowledge that a previous
    # build wrote as ACTIVE WORLD_KNOWLEDGE with source_supported=1. Mock data
    # is identified precisely — kind=WORLD_KNOWLEDGE, not human-confirmed,
    # model-inferred, and carrying a mock web source (example.com URL or an
    # ``extra`` field tagged with a mock provider). Human-confirmed or manual
    # memories never match, and nothing is deleted: ACTIVE -> CANDIDATE and
    # source_supported -> 0. Idempotent by construction (only ACTIVE rows are
    # touched, and the version gate means it runs at most once).
    (
        6,
        """
        INSERT INTO audit_log(audit_id, actor, action, entity_type, entity_id, detail, created_at)
        SELECT 'mig_mock_' || memory_id, 'migration', 'demote_mock_knowledge',
               'memory', memory_id,
               'mock research demoted ACTIVE -> CANDIDATE (source_supported cleared)',
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        FROM memory_items
        WHERE kind = 'WORLD_KNOWLEDGE' AND status = 'ACTIVE'
          AND human_confirmed = 0 AND model_inferred = 1
          AND memory_id IN (
              SELECT e.memory_id FROM memory_evidence e
              JOIN memory_sources s ON s.source_id = e.source_id
              WHERE s.url LIKE 'https://example.com/%' OR s.extra LIKE '%mock%'
          );

        UPDATE memory_items
        SET status = 'CANDIDATE', source_supported = 0
        WHERE kind = 'WORLD_KNOWLEDGE' AND status = 'ACTIVE'
          AND human_confirmed = 0 AND model_inferred = 1
          AND memory_id IN (
              SELECT e.memory_id FROM memory_evidence e
              JOIN memory_sources s ON s.source_id = e.source_id
              WHERE s.url LIKE 'https://example.com/%' OR s.extra LIKE '%mock%'
          );
        """,
    ),
    # Version 7 (DeepSeek integration): make model-proposed candidates
    # idempotent per source event. Existing duplicate rows are collapsed onto
    # the first insertion before the unique index is created; from then on
    # ``INSERT OR IGNORE`` (see AnalysisRepository.save_candidate) makes a
    # repeated proposal a no-op instead of a duplicate row. Candidates remain
    # inert proposals — this migration never promotes anything to ACTIVE.
    (
        7,
        """
        DELETE FROM memory_candidates
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM memory_candidates
            GROUP BY source_event_id, content
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidates_event_content
            ON memory_candidates(source_event_id, content);
        """,
    ),
    # Version 8 (V0.2A): durable memory write-through, truthful outcomes,
    # memory assignments and structured cognitive decisions.
    #
    # - memory_candidates gains the *suggested* semantic classification and
    #   evidence span. These are model hints; the trusted consolidation layer
    #   still decides whether a durable write is allowed.
    # - cognitive_analyses / agent_actions gain the structured decision fields
    #   so a model suggestion and the local policy's final action are stored
    #   separately and remain inspectable.
    # - memory_candidate_writes is the durable candidate -> memory mapping and
    #   the idempotency anchor (one row per candidate). Existing legacy
    #   candidates are stamped NO_MEMORY rather than blindly promoted.
    # - memory_assignments associates a memory with a person for retrieval.
    #   It never changes the memory's meaning or its privacy/visibility state.
    # - memory_provenance records where a model-extracted memory came from
    #   (source event, candidate, speaker, subject, assertion, topic).
    (
        8,
        """
        ALTER TABLE memory_candidates ADD COLUMN assertion_type TEXT NOT NULL DEFAULT 'UNKNOWN';
        ALTER TABLE memory_candidates ADD COLUMN certainty_category TEXT NOT NULL DEFAULT 'UNKNOWN';
        ALTER TABLE memory_candidates ADD COLUMN explicit_remember_request INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE memory_candidates ADD COLUMN topic TEXT;
        ALTER TABLE memory_candidates ADD COLUMN kind TEXT;
        ALTER TABLE memory_candidates ADD COLUMN source_evidence_span TEXT;

        ALTER TABLE cognitive_analyses ADD COLUMN intent TEXT NOT NULL DEFAULT 'unknown';
        ALTER TABLE cognitive_analyses ADD COLUMN addressed_to TEXT;
        ALTER TABLE cognitive_analyses ADD COLUMN silence_requested INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE cognitive_analyses ADD COLUMN reply_necessity TEXT NOT NULL DEFAULT 'OPTIONAL';
        ALTER TABLE cognitive_analyses ADD COLUMN conversation_state TEXT NOT NULL DEFAULT 'unknown';

        ALTER TABLE agent_actions ADD COLUMN model_suggested_action TEXT;
        ALTER TABLE agent_actions ADD COLUMN policy_final_action TEXT;
        ALTER TABLE agent_actions ADD COLUMN override_reason TEXT;

        CREATE TABLE IF NOT EXISTS memory_candidate_writes (
            candidate_id    TEXT PRIMARY KEY
                            REFERENCES memory_candidates(candidate_id) ON DELETE CASCADE,
            memory_id       TEXT REFERENCES memory_items(memory_id) ON DELETE SET NULL,
            source_event_id TEXT NOT NULL,
            outcome         TEXT NOT NULL,
            detail          TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_candidate_writes_memory
            ON memory_candidate_writes(memory_id);
        CREATE INDEX IF NOT EXISTS idx_candidate_writes_outcome
            ON memory_candidate_writes(outcome);

        CREATE TABLE IF NOT EXISTS memory_assignments (
            memory_id  TEXT NOT NULL REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            person_id  TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL,
            PRIMARY KEY (memory_id, person_id)
        );

        CREATE INDEX IF NOT EXISTS idx_memory_assignments_person
            ON memory_assignments(person_id);

        CREATE TABLE IF NOT EXISTS memory_provenance (
            memory_id              TEXT PRIMARY KEY
                                   REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            source_event_id        TEXT,
            candidate_id           TEXT,
            speaker_person_id      TEXT,
            subject_person_id      TEXT,
            source_type            TEXT NOT NULL DEFAULT 'utterance',
            assertion_type         TEXT NOT NULL DEFAULT 'UNKNOWN',
            certainty_category     TEXT NOT NULL DEFAULT 'UNKNOWN',
            topic                  TEXT,
            explicit_remember_request INTEGER NOT NULL DEFAULT 0,
            self_reported          INTEGER NOT NULL DEFAULT 0,
            third_party_reported   INTEGER NOT NULL DEFAULT 0,
            model_inferred         INTEGER NOT NULL DEFAULT 0,
            created_at             TEXT NOT NULL,
            updated_at             TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_provenance_candidate
            ON memory_provenance(candidate_id) WHERE candidate_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_provenance_subject_topic
            ON memory_provenance(subject_person_id, topic);

        -- Legacy candidates predate durable write-through. Record them
        -- explicitly as not back-filled: no memory is invented, and the
        -- consolidation handler treats a candidate with a mapping row as
        -- already decided.
        INSERT OR IGNORE INTO memory_candidate_writes(
            candidate_id, memory_id, source_event_id, outcome, detail,
            created_at, updated_at
        )
        SELECT candidate_id, NULL, source_event_id, 'NO_MEMORY',
               'legacy_candidate_not_backfilled',
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        FROM memory_candidates;
        """,
    ),
    # Version 9 (CMF 1.0): portable memory-file foundation, agent core,
    # relationships, append-only provenance, research/learning durability,
    # goals, media reservations and embedding reservations.
    #
    # Every table lives in the same SQLite file. The manifest is a singleton
    # (``singleton = 1``) holding the stable agent_id and the CMF format
    # version, which is deliberately independent of the schema version.
    # Seeded rows are inserted only when absent, so a legacy database upgraded
    # from version 8 receives one default identity without losing any data.
    (
        9,
        """
        CREATE TABLE IF NOT EXISTS cmf_manifest (
            singleton                INTEGER PRIMARY KEY CHECK (singleton = 1),
            format_version           TEXT NOT NULL,
            agent_id                 TEXT NOT NULL UNIQUE,
            display_name             TEXT,
            created_at               TEXT NOT NULL,
            updated_at               TEXT NOT NULL,
            lineage_id               TEXT,
            lineage_parent_agent_id  TEXT,
            clone_of                 TEXT,
            notes                    TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_identity (
            agent_id     TEXT PRIMARY KEY
                         REFERENCES cmf_manifest(agent_id) ON DELETE CASCADE,
            display_name TEXT NOT NULL,
            pronouns     TEXT,
            description  TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            revision     INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS agent_persona (
            agent_id             TEXT PRIMARY KEY
                                 REFERENCES cmf_manifest(agent_id) ON DELETE CASCADE,
            humor                INTEGER NOT NULL DEFAULT 50
                                 CHECK (humor BETWEEN 0 AND 100),
            warmth               INTEGER NOT NULL DEFAULT 60
                                 CHECK (warmth BETWEEN 0 AND 100),
            curiosity            INTEGER NOT NULL DEFAULT 60
                                 CHECK (curiosity BETWEEN 0 AND 100),
            formality            INTEGER NOT NULL DEFAULT 45
                                 CHECK (formality BETWEEN 0 AND 100),
            verbosity            INTEGER NOT NULL DEFAULT 45
                                 CHECK (verbosity BETWEEN 0 AND 100),
            assertiveness        INTEGER NOT NULL DEFAULT 45
                                 CHECK (assertiveness BETWEEN 0 AND 100),
            empathy              INTEGER NOT NULL DEFAULT 65
                                 CHECK (empathy BETWEEN 0 AND 100),
            playfulness          INTEGER NOT NULL DEFAULT 50
                                 CHECK (playfulness BETWEEN 0 AND 100),
            patience             INTEGER NOT NULL DEFAULT 65
                                 CHECK (patience BETWEEN 0 AND 100),
            updated_at           TEXT NOT NULL,
            revision             INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS persona_revisions (
            revision_id TEXT PRIMARY KEY,
            agent_id    TEXT NOT NULL,
            revision    INTEGER NOT NULL,
            changes     TEXT NOT NULL DEFAULT '{}',
            changed_by  TEXT NOT NULL,
            changed_at  TEXT NOT NULL,
            reason      TEXT,
            UNIQUE (agent_id, revision)
        );

        CREATE TABLE IF NOT EXISTS person_profiles (
            person_id   TEXT PRIMARY KEY
                        REFERENCES persons(person_id) ON DELETE CASCADE,
            familiarity TEXT NOT NULL DEFAULT 'STRANGER'
                        CHECK (familiarity IN ('STRANGER', 'ACQUAINTANCE', 'FAMILIAR', 'CLOSE')),
            notes       TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relationship_presets (
            tier        TEXT PRIMARY KEY,
            label       TEXT NOT NULL,
            description TEXT,
            style_deltas TEXT NOT NULL DEFAULT '{}',
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS person_interaction_profiles (
            person_id        TEXT PRIMARY KEY
                             REFERENCES persons(person_id) ON DELETE CASCADE,
            relationship_tier TEXT NOT NULL DEFAULT 'STRANGER'
                             REFERENCES relationship_presets(tier),
            style_overrides  TEXT NOT NULL DEFAULT '{}',
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relationship_history (
            history_id TEXT PRIMARY KEY,
            person_id  TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            old_tier   TEXT,
            new_tier   TEXT NOT NULL,
            actor      TEXT NOT NULL,
            reason     TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_relationship_history_person
            ON relationship_history(person_id, created_at);

        CREATE TABLE IF NOT EXISTS memory_provenance_events (
            event_id          TEXT PRIMARY KEY,
            memory_id         TEXT NOT NULL
                              REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            source_event_id   TEXT,
            candidate_id      TEXT,
            speaker_person_id TEXT,
            subject_person_id TEXT,
            old_version       INTEGER,
            new_version       INTEGER,
            event_type        TEXT NOT NULL,
            reason            TEXT,
            payload           TEXT NOT NULL DEFAULT '{}',
            created_at        TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_provenance_events_memory
            ON memory_provenance_events(memory_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_provenance_events_candidate
            ON memory_provenance_events(candidate_id);

        ALTER TABLE memory_provenance ADD COLUMN entity_key TEXT;
        ALTER TABLE memory_provenance ADD COLUMN attribute_key TEXT;

        CREATE INDEX IF NOT EXISTS idx_provenance_claim
            ON memory_provenance(subject_person_id, entity_key, attribute_key);

        CREATE TABLE IF NOT EXISTS research_projects (
            project_id TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            question   TEXT NOT NULL,
            status     TEXT NOT NULL DEFAULT 'OPEN',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            extra      TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS research_runs (
            run_id       TEXT PRIMARY KEY,
            project_id   TEXT NOT NULL
                         REFERENCES research_projects(project_id) ON DELETE CASCADE,
            task_id      TEXT,
            status       TEXT NOT NULL,
            provider     TEXT,
            started_at   TEXT,
            completed_at TEXT,
            error        TEXT,
            budget_used  TEXT NOT NULL DEFAULT '{}',
            extra        TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_research_runs_project
            ON research_runs(project_id, started_at);

        CREATE TABLE IF NOT EXISTS search_queries (
            query_id     TEXT PRIMARY KEY,
            run_id       TEXT NOT NULL
                         REFERENCES research_runs(run_id) ON DELETE CASCADE,
            query_text   TEXT NOT NULL,
            provider     TEXT,
            executed_at  TEXT NOT NULL,
            result_count INTEGER NOT NULL DEFAULT 0,
            extra        TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS web_sources (
            source_id    TEXT PRIMARY KEY,
            url          TEXT,
            title        TEXT,
            publisher    TEXT,
            published_at TEXT,
            retrieved_at TEXT NOT NULL,
            source_kind  TEXT NOT NULL DEFAULT 'web',
            trust_level  TEXT NOT NULL DEFAULT 'UNVERIFIED',
            extra        TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_web_sources_url ON web_sources(url);

        CREATE TABLE IF NOT EXISTS source_excerpts (
            excerpt_id  TEXT PRIMARY KEY,
            source_id   TEXT NOT NULL
                        REFERENCES web_sources(source_id) ON DELETE CASCADE,
            run_id      TEXT,
            excerpt     TEXT NOT NULL,
            char_count  INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS knowledge_claims (
            claim_id           TEXT PRIMARY KEY,
            project_id         TEXT,
            claim_text         TEXT NOT NULL,
            status             TEXT NOT NULL DEFAULT 'PROPOSED',
            confidence         REAL NOT NULL DEFAULT 0,
            verification_level TEXT NOT NULL DEFAULT 'UNVERIFIED',
            created_at         TEXT NOT NULL,
            updated_at         TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS claim_evidence (
            claim_id   TEXT NOT NULL
                       REFERENCES knowledge_claims(claim_id) ON DELETE CASCADE,
            source_id  TEXT NOT NULL
                       REFERENCES web_sources(source_id) ON DELETE CASCADE,
            excerpt_id TEXT NOT NULL DEFAULT '',
            relation   TEXT NOT NULL DEFAULT 'supports',
            created_at TEXT NOT NULL,
            PRIMARY KEY (claim_id, source_id, excerpt_id)
        );

        CREATE TABLE IF NOT EXISTS research_findings (
            finding_id  TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL
                        REFERENCES research_projects(project_id) ON DELETE CASCADE,
            run_id      TEXT,
            summary     TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'DRAFT',
            created_at  TEXT NOT NULL,
            extra       TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS source_refresh_rules (
            rule_id          TEXT PRIMARY KEY,
            source_id        TEXT NOT NULL
                             REFERENCES web_sources(source_id) ON DELETE CASCADE,
            next_check_at    TEXT,
            interval_seconds INTEGER NOT NULL DEFAULT 86400,
            last_checked_at  TEXT,
            active           INTEGER NOT NULL DEFAULT 1,
            extra            TEXT NOT NULL DEFAULT '{}'
        );

        ALTER TABLE research_tasks ADD COLUMN project_id TEXT;
        ALTER TABLE research_tasks ADD COLUMN run_id TEXT;

        CREATE TABLE IF NOT EXISTS curiosity_queue (
            item_id      TEXT PRIMARY KEY,
            question     TEXT NOT NULL,
            source_kind  TEXT NOT NULL DEFAULT 'curiosity',
            source_ref   TEXT,
            priority     INTEGER NOT NULL DEFAULT 0,
            status       TEXT NOT NULL DEFAULT 'PENDING',
            created_at   TEXT NOT NULL,
            scheduled_at TEXT,
            extra        TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS reflection_records (
            reflection_id     TEXT PRIMARY KEY,
            subject_memory_id TEXT,
            reflection_text   TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'PROPOSED',
            source_access     TEXT,
            created_at        TEXT NOT NULL,
            extra             TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS learning_runs (
            run_id       TEXT PRIMARY KEY,
            job_id       TEXT,
            kind         TEXT NOT NULL,
            status       TEXT NOT NULL,
            started_at   TEXT,
            completed_at TEXT,
            budget_json  TEXT NOT NULL DEFAULT '{}',
            provider     TEXT,
            error        TEXT,
            extra        TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS learning_checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            run_id        TEXT NOT NULL
                          REFERENCES learning_runs(run_id) ON DELETE CASCADE,
            stage         TEXT NOT NULL,
            payload       TEXT NOT NULL DEFAULT '{}',
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS knowledge_refresh_jobs (
            job_id       TEXT PRIMARY KEY,
            source_id    TEXT,
            claim_id     TEXT,
            status       TEXT NOT NULL DEFAULT 'PENDING',
            scheduled_at TEXT,
            attempts     INTEGER NOT NULL DEFAULT 0,
            last_error   TEXT,
            extra        TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS goals (
            goal_id     TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            description TEXT,
            status      TEXT NOT NULL DEFAULT 'OPEN',
            priority    INTEGER NOT NULL DEFAULT 0,
            next_action TEXT,
            due_at      TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            extra       TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS goal_checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            goal_id       TEXT NOT NULL
                          REFERENCES goals(goal_id) ON DELETE CASCADE,
            status        TEXT NOT NULL,
            summary       TEXT,
            created_at    TEXT NOT NULL,
            extra         TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS goal_dependencies (
            goal_id         TEXT NOT NULL
                            REFERENCES goals(goal_id) ON DELETE CASCADE,
            depends_on_goal_id TEXT NOT NULL
                            REFERENCES goals(goal_id) ON DELETE CASCADE,
            created_at      TEXT NOT NULL,
            PRIMARY KEY (goal_id, depends_on_goal_id)
        );

        CREATE TABLE IF NOT EXISTS media_assets (
            asset_id   TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            path       TEXT NOT NULL,
            mime_type  TEXT,
            sha256     TEXT,
            created_at TEXT NOT NULL,
            extra      TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS media_annotations (
            annotation_id   TEXT PRIMARY KEY,
            asset_id        TEXT NOT NULL
                            REFERENCES media_assets(asset_id) ON DELETE CASCADE,
            annotation_type TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            extra           TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS media_links (
            asset_id  TEXT NOT NULL
                      REFERENCES media_assets(asset_id) ON DELETE CASCADE,
            memory_id TEXT NOT NULL
                      REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            link_type TEXT NOT NULL,
            PRIMARY KEY (asset_id, memory_id, link_type)
        );

        CREATE TABLE IF NOT EXISTS embedding_models (
            model_id    TEXT PRIMARY KEY,
            provider    TEXT NOT NULL,
            model_name  TEXT NOT NULL,
            dimensions  INTEGER NOT NULL,
            created_at  TEXT NOT NULL,
            extra       TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id  TEXT NOT NULL
                       REFERENCES memory_items(memory_id) ON DELETE CASCADE,
            model_id   TEXT NOT NULL
                       REFERENCES embedding_models(model_id) ON DELETE CASCADE,
            vector     BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (memory_id, model_id)
        );

        CREATE TABLE IF NOT EXISTS embedding_index_metadata (
            singleton     INTEGER PRIMARY KEY CHECK (singleton = 1),
            model_id      TEXT,
            status        TEXT NOT NULL DEFAULT 'EMPTY',
            last_built_at TEXT,
            extra         TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS retrieval_feedback (
            feedback_id TEXT PRIMARY KEY,
            memory_id   TEXT,
            person_id   TEXT,
            query       TEXT,
            signal      TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            extra       TEXT NOT NULL DEFAULT '{}'
        );

        INSERT OR IGNORE INTO relationship_presets(tier, label, description, style_deltas, created_at)
        VALUES
            ('STRANGER', 'Stranger', 'Polite, reserved, low familiarity',
             '{"humor": -5, "warmth": -5, "formality": 10, "verbosity": -10}',
             strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            ('ACQUAINTANCE', 'Acquaintance', 'Friendly but still polite',
             '{"warmth": 5, "formality": 5}',
             strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            ('FAMILIAR', 'Familiar', 'Relaxed, conversational, some shared humour',
             '{"humor": 10, "warmth": 15, "formality": -10, "verbosity": 5}',
             strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            ('CLOSE', 'Close', 'Warm, playful, informal',
             '{"humor": 15, "warmth": 25, "formality": -20, "verbosity": 10, "playfulness": 10}',
             strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

        INSERT OR IGNORE INTO cmf_manifest(
            singleton, format_version, agent_id, display_name, created_at,
            updated_at, lineage_id
        )
        SELECT 1, '1.0', 'agent_' || lower(hex(randomblob(8))), 'Companion',
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
               'lineage_' || lower(hex(randomblob(8)))
        WHERE NOT EXISTS (SELECT 1 FROM cmf_manifest);

        INSERT OR IGNORE INTO agent_identity(
            agent_id, display_name, pronouns, description, created_at, updated_at, revision
        )
        SELECT agent_id, COALESCE(display_name, 'Companion'), NULL,
               'A persistent AI companion identity stored in this memory file.',
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 1
        FROM cmf_manifest;

        INSERT OR IGNORE INTO agent_persona(agent_id, updated_at)
        SELECT agent_id, strftime('%Y-%m-%dT%H:%M:%fZ', 'now') FROM cmf_manifest;

        INSERT OR IGNORE INTO persona_revisions(
            revision_id, agent_id, revision, changes, changed_by, changed_at, reason
        )
        SELECT 'persona_seed_' || agent_id, agent_id, 1, '{}', 'system',
               strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'initial_defaults'
        FROM cmf_manifest;

        INSERT OR IGNORE INTO embedding_index_metadata(singleton, status, extra)
        VALUES (1, 'EMPTY', '{}');
        """,
    ),
    # Version 10 (V0.3): hybrid vector retrieval. Extends the Schema-9
    # embedding reservations (never rebuilds or drops them). All additions are
    # NULL-able or DEFAULT-ed so any existing Schema-9 rows keep working; an
    # existing vector row without provenance is conservatively marked
    # UNVERIFIED so the backfill rebuilds it rather than trusting it.
    (
        10,
        """
        ALTER TABLE embedding_models ADD COLUMN model_revision TEXT;
        ALTER TABLE embedding_models ADD COLUMN normalization TEXT NOT NULL DEFAULT 'l2';
        ALTER TABLE embedding_models ADD COLUMN query_prefix TEXT;
        ALTER TABLE embedding_models ADD COLUMN document_prefix TEXT;
        ALTER TABLE embedding_models ADD COLUMN distance_metric TEXT NOT NULL DEFAULT 'cosine';
        ALTER TABLE embedding_models ADD COLUMN max_input_tokens INTEGER;
        ALTER TABLE embedding_models ADD COLUMN configuration_hash TEXT;

        ALTER TABLE memory_embeddings ADD COLUMN memory_version INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE memory_embeddings ADD COLUMN content_hash TEXT;
        ALTER TABLE memory_embeddings ADD COLUMN embedding_status TEXT NOT NULL DEFAULT 'UNVERIFIED';
        ALTER TABLE memory_embeddings ADD COLUMN updated_at TEXT;
        ALTER TABLE memory_embeddings ADD COLUMN last_error TEXT;

        CREATE INDEX IF NOT EXISTS idx_memory_embeddings_status
            ON memory_embeddings(embedding_status);
        CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model_status
            ON memory_embeddings(model_id, embedding_status);
        CREATE INDEX IF NOT EXISTS idx_embedding_models_hash
            ON embedding_models(configuration_hash);
        """,
    ),
    # Version 11 (V0.3): persistent emotion framework. Stable personality,
    # persistent appraisal rules, durable emotion events, temporary state
    # snapshots, feedback/audit and provider-neutral voice profiles. Emotion
    # state decays lazily; event memory and rules are never dropped by decay.
    (
        11,
        """
        CREATE TABLE IF NOT EXISTS emotion_profiles (
            agent_id TEXT PRIMARY KEY,
            baseline_valence REAL NOT NULL DEFAULT 0
                CHECK (baseline_valence BETWEEN -1 AND 1),
            baseline_arousal REAL NOT NULL DEFAULT 0
                CHECK (baseline_arousal BETWEEN 0 AND 1),
            default_half_life_seconds INTEGER NOT NULL DEFAULT 3600
                CHECK (default_half_life_seconds > 0),
            responsiveness REAL NOT NULL DEFAULT 0.5
                CHECK (responsiveness BETWEEN 0 AND 1),
            expression_style TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS emotion_appraisal_rules (
            rule_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            pattern TEXT NOT NULL DEFAULT '{}',
            emotion TEXT NOT NULL,
            valence_delta REAL NOT NULL DEFAULT 0
                CHECK (valence_delta BETWEEN -2 AND 2),
            arousal_delta REAL NOT NULL DEFAULT 0
                CHECK (arousal_delta BETWEEN -1 AND 1),
            intensity REAL NOT NULL DEFAULT 0.5
                CHECK (intensity BETWEEN 0 AND 1),
            priority INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            source TEXT NOT NULL DEFAULT 'system',
            evidence_ref TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS emotion_events (
            emotion_event_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            conversation_id TEXT,
            person_id TEXT,
            source_event_id TEXT,
            appraisal_rule_id TEXT,
            memory_id TEXT,
            emotion TEXT NOT NULL,
            valence REAL NOT NULL,
            arousal REAL NOT NULL,
            intensity REAL NOT NULL,
            appraisal_reason TEXT,
            created_at TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_emotion_events_source
            ON emotion_events(source_event_id)
            WHERE source_event_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS emotion_state_snapshots (
            agent_id TEXT NOT NULL,
            person_id TEXT NOT NULL,
            valence REAL NOT NULL,
            arousal REAL NOT NULL,
            intensity REAL NOT NULL,
            emotion TEXT NOT NULL,
            half_life_seconds REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (agent_id, person_id)
        );

        CREATE TABLE IF NOT EXISTS emotion_feedback (
            feedback_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            person_id TEXT,
            conversation_id TEXT,
            source_event_id TEXT,
            feedback_text TEXT NOT NULL,
            suggestion_type TEXT,
            suggestion_payload TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'OPEN',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS voice_profiles (
            voice_profile_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT 'none',
            voice_id TEXT,
            language TEXT NOT NULL DEFAULT 'zh-CN',
            speaking_rate REAL NOT NULL DEFAULT 1.0
                CHECK (speaking_rate > 0),
            pitch REAL,
            extra TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    ),
    # Version 12 (V0.3.1): semantic affect hint + atomic emotion turn receipts.
    #
    # - ``cognitive_analyses.affective_signal_json`` lets a crash-recovered turn
    #   reuse the exact validated signal the model produced, instead of losing
    #   it or calling the model again.
    # - ``emotion_turn_receipts`` records that one utterance was evaluated by
    #   the emotion engine, independent of whether a rule fired. It is the
    #   idempotency anchor for retries.
    # - The unique (agent_id, name) index makes default-rule seeding a
    #   concurrent-safe INSERT OR IGNORE. Pre-existing duplicate names (only
    #   possible before this constraint) are reduced to their earliest row.
    (
        12,
        """
        ALTER TABLE cognitive_analyses ADD COLUMN affective_signal_json TEXT;

        CREATE TABLE IF NOT EXISTS emotion_turn_receipts (
            source_event_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            person_id TEXT NOT NULL,
            outcome TEXT NOT NULL
                CHECK (outcome IN ('APPLIED', 'NO_CHANGE')),
            processed_at TEXT NOT NULL
        );

        DELETE FROM emotion_appraisal_rules
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM emotion_appraisal_rules GROUP BY agent_id, name
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_emotion_rules_agent_name
            ON emotion_appraisal_rules(agent_id, name);
        """,
    ),
    # Version 13 (V0.3.2): explainable nonresponse.
    #
    # - ``cognitive_analyses.nonresponse_motive_json`` stores the model's
    #   locally validated silence hint (NULL when absent/invalid/RESPOND).
    # - ``agent_actions.nonresponse_trace_json`` stores the *final* reason
    #   derived after local policy, including when policy overrode the model.
    # Both are nullable so schema-12 rows remain readable without back-filling
    # a motive or evidence that was never observed.
    (
        13,
        """
        ALTER TABLE cognitive_analyses ADD COLUMN nonresponse_motive_json TEXT;

        ALTER TABLE agent_actions ADD COLUMN nonresponse_trace_json TEXT;
        """,
    ),
    # Version 14 (V0.4): continuous scene awareness and personality-driven
    # dialogue.
    #
    # - ``scene_snapshots`` holds one optimistic-concurrency scene per
    #   (conversation, person) scope. ``revision`` is the CAS token.
    # - ``scene_transition_receipts`` is the idempotency anchor: a source event
    #   can advance the scene at most once, even across retries/crashes.
    # - ``cognitive_analyses.scene_proposal_json`` preserves the locally
    #   validated proposal so an action recovered without a receipt can be
    #   finalised idempotently instead of re-running model semantics.
    # - ``dialogue_style_examples`` stores trusted, admin-authored style
    #   references. ``person_id IS NULL`` means global; a person-scoped example
    #   is never shown to another person. The model can never insert here.
    (
        14,
        """
        ALTER TABLE cognitive_analyses ADD COLUMN scene_proposal_json TEXT;

        CREATE TABLE IF NOT EXISTS scene_snapshots (
            conversation_id        TEXT NOT NULL,
            person_id              TEXT NOT NULL,
            revision               INTEGER NOT NULL DEFAULT 0,
            state_json             TEXT NOT NULL,
            source_event_ids_json  TEXT NOT NULL DEFAULT '[]',
            last_committed_event_id TEXT,
            updated_at             TEXT NOT NULL,
            PRIMARY KEY (conversation_id, person_id),
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );

        CREATE TABLE IF NOT EXISTS scene_transition_receipts (
            source_event_id   TEXT PRIMARY KEY
                              REFERENCES utterances(event_id) ON DELETE CASCADE,
            conversation_id   TEXT NOT NULL,
            person_id         TEXT NOT NULL,
            result_revision   INTEGER NOT NULL,
            transition_json   TEXT NOT NULL,
            applied_at        TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_scene_receipts_scope
            ON scene_transition_receipts(conversation_id, person_id);

        CREATE TABLE IF NOT EXISTS dialogue_style_examples (
            example_id    TEXT PRIMARY KEY,
            agent_id      TEXT NOT NULL,
            person_id     TEXT REFERENCES persons(person_id) ON DELETE CASCADE,
            scenario      TEXT NOT NULL,
            user_text     TEXT NOT NULL,
            ideal_response TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'ACTIVE'
                          CHECK (status IN ('ACTIVE', 'DISABLED')),
            created_by    TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_style_examples_scope
            ON dialogue_style_examples(agent_id, person_id, status);
        """,
    ),
    # Version 15 (V0.4.1): character identity, affective dynamics repair and
    # explainable appraisal.
    #
    # - ``agent_persona_core`` is the admin-maintained stable identity layer
    #   (values, temperament, preferences, quirks, contradictions, boundaries,
    #   voice identity). It is never written by a model.
    # - Emotion snapshots gain the three independently decaying layers plus the
    #   per-category activations needed to derive a consistent label. New
    #   columns have neutral defaults, so old snapshots read as legacy acute
    #   affect without inventing a mood or tension value.
    # - Emotion events gain the validated event/target/intent and before/after
    #   diagnostics. Old rows keep NULL, which means "not recorded", never
    #   "hostile".
    # - Receipts gain ``reason``/``detail`` so NO_CHANGE is explainable.
    # - ``cognitive_analyses.dialogue_impulse_json`` preserves the model's
    #   optional turn interpretation for the single response call.
    (
        15,
        """
        CREATE TABLE IF NOT EXISTS agent_persona_core (
            agent_id             TEXT PRIMARY KEY
                                 REFERENCES cmf_manifest(agent_id) ON DELETE CASCADE,
            core_values_json     TEXT NOT NULL DEFAULT '[]',
            temperament_json     TEXT NOT NULL DEFAULT '{}',
            preferences_json     TEXT NOT NULL DEFAULT '{}',
            quirks_json          TEXT NOT NULL DEFAULT '[]',
            contradictions_json  TEXT NOT NULL DEFAULT '[]',
            conversational_boundaries_json TEXT NOT NULL DEFAULT '[]',
            voice_identity_json  TEXT NOT NULL DEFAULT '{}',
            revision             INTEGER NOT NULL DEFAULT 1,
            updated_at           TEXT NOT NULL,
            updated_by           TEXT NOT NULL DEFAULT 'admin'
        );

        ALTER TABLE emotion_state_snapshots
            ADD COLUMN activations_json TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN mood_valence REAL NOT NULL DEFAULT 0;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN mood_arousal REAL NOT NULL DEFAULT 0;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN mood_intensity REAL NOT NULL DEFAULT 0;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN mood_emotion TEXT NOT NULL DEFAULT 'NEUTRAL';
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN mood_half_life_seconds REAL NOT NULL DEFAULT 1800;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN mood_updated_at TEXT;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN relationship_tension REAL NOT NULL DEFAULT 0;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN tension_half_life_seconds REAL NOT NULL DEFAULT 2700;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN tension_updated_at TEXT;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN last_event_type TEXT;
        ALTER TABLE emotion_state_snapshots
            ADD COLUMN last_event_at TEXT;

        ALTER TABLE emotion_events ADD COLUMN event_type TEXT;
        ALTER TABLE emotion_events ADD COLUMN target TEXT;
        ALTER TABLE emotion_events ADD COLUMN social_intent TEXT;
        ALTER TABLE emotion_events ADD COLUMN validation TEXT;
        ALTER TABLE emotion_events ADD COLUMN acute_before REAL;
        ALTER TABLE emotion_events ADD COLUMN acute_after REAL;
        ALTER TABLE emotion_events ADD COLUMN mood_after REAL;
        ALTER TABLE emotion_events ADD COLUMN tension_after REAL;

        ALTER TABLE emotion_turn_receipts ADD COLUMN reason TEXT;
        ALTER TABLE emotion_turn_receipts ADD COLUMN detail TEXT;

        ALTER TABLE cognitive_analyses ADD COLUMN dialogue_impulse_json TEXT;
        """,
    ),
    # Version 16 (V0.4.2): conversation-driven affect, character identity and
    # stance continuity.
    #
    # - ``agent_persona_core.character_description`` stores a trusted,
    #   admin-authored long-form character description. It is prompt context,
    #   never a model-authored biography and never a source of real-world facts.
    # - ``agent_stances`` records stances the companion actually expressed in an
    #   emitted response (or an admin explicitly configured). It deliberately
    #   has no model-private-thought source: continuity may only cite what was
    #   said or configured.
    # - ``cognitive_analyses.pragmatic_json`` preserves the locally validated
    #   pragmatic reading for retries, so an analysis is not recomputed and the
    #   response sees the same frame it was analysed with.
    # - Emotion events gain the transition cause/frame fields for audit. Old
    #   rows keep NULL, which means "not recorded".
    (
        16,
        """
        ALTER TABLE agent_persona_core
            ADD COLUMN character_description TEXT NOT NULL DEFAULT '';

        CREATE TABLE IF NOT EXISTS agent_stances (
            stance_id        TEXT PRIMARY KEY,
            agent_id         TEXT NOT NULL
                             REFERENCES cmf_manifest(agent_id) ON DELETE CASCADE,
            person_id        TEXT REFERENCES persons(person_id) ON DELETE CASCADE,
            conversation_id  TEXT,
            topic            TEXT NOT NULL DEFAULT 'UNKNOWN',
            stance           TEXT NOT NULL,
            source           TEXT NOT NULL
                             CHECK (source IN ('ADMIN', 'RESPONSE')),
            source_event_id  TEXT,
            status           TEXT NOT NULL DEFAULT 'ACTIVE'
                             CHECK (status IN ('ACTIVE', 'RETIRED')),
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_stances_source
            ON agent_stances(source_event_id, stance)
            WHERE source_event_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_agent_stances_scope
            ON agent_stances(agent_id, person_id, status, created_at);

        ALTER TABLE cognitive_analyses ADD COLUMN pragmatic_json TEXT;

        ALTER TABLE emotion_events ADD COLUMN transition_cause TEXT;
        ALTER TABLE emotion_events ADD COLUMN frame_from TEXT;
        ALTER TABLE emotion_events ADD COLUMN frame_to TEXT;
        ALTER TABLE emotion_events ADD COLUMN scene_relevance REAL;
        """,
    ),
]
