# CMF 1.0 — Migration Plan

## Policy

Migrations are **append-only and numbered**. A published migration is never
edited; the only way to change the schema is to add a new migration with a
higher number. Each migration commits atomically: all of its SQL statements
and its `schema_version` stamp commit together, so a mid-migration failure
rolls back the partial DDL *and* leaves the version unchanged.

Migration SQL is deliberately simple: split on `;`, no `PRAGMA`, no string
literals containing semicolons, no trigger bodies. This keeps the splitter
correct without a full SQL parser.

## Current state

- `SCHEMA_VERSION = 9`
- Migrations `1..9`
- CMF format version `"1.0"` (major `1`), independent of the schema version.
- `application_id = 0x434D4601`.

## History

| Version | Purpose | Data-loss risk |
| --- | --- | --- |
| 1 | Base tables (persons, consents, conversations, utterances, analyses, `memory_facts`) | none |
| 2 | Job lease/ownership columns | additive |
| 3 | Unified memory foundation | additive (old `memory_facts` retired, not dropped) |
| 4 | Management UI + idle research | additive |
| 5 | Privacy + research-reliability columns | additive (`allow_external_ai` defaults off) |
| 6 | Demote mock research knowledge | truthfulness correction, no user data dropped |
| 7 | Model-proposed memory candidates | additive |
| 8 | V0.2A durable write-through + provenance | additive |
| 9 | **CMF 1.0** (see below) | additive + seeded defaults |

## Migration 9 in detail

Migration 9 is the CMF migration. It is designed to upgrade a version-8
database in place with **no data loss**:

- **New tables** for manifest, agent core, relationships, provenance events,
  research state, learning state, goals, media and embedding reservations.
- **Column additions** (all backward compatible `ALTER TABLE ... ADD COLUMN`):
  - `memory_provenance.entity_key`, `memory_provenance.attribute_key`
  - `research_tasks.project_id`, `research_tasks.run_id`
- **Index** `idx_provenance_claim` on
  `memory_provenance(subject_person_id, entity_key, attribute_key)`.
- **Seeds**, all guarded by `INSERT OR IGNORE` / `WHERE NOT EXISTS`:
  - four `relationship_presets` rows;
  - one `cmf_manifest` row with a generated `agent_id` and `lineage_id`;
  - one `agent_identity` row derived from the manifest;
  - one `agent_persona` row with the default trait values;
  - one `persona_revision` row recording the initial defaults;
  - one `embedding_index_metadata` row (`status = 'EMPTY'`).

A legacy database that already has rows in the new tables is left untouched,
so re-running migration 9 is idempotent.

## Upgrade path (legacy `.db`)

1. Open the file with `CmfFileService.migrate(path)` (or `open(path)`).
2. `_refuse_foreign_database` confirms it is a Companion database (has a
   `schema_version` table, `application_id` is `0` or `0x434D4601`).
3. Migrations `> current` apply in order, each atomically.
4. `_ensure_application_id` stamps `0x434D4601`.
5. `_refuse_newer_format` confirms the manifest format major is not newer than
   `1`.

The result is a valid CMF 1.0 file with the original agent data preserved and
a new default identity added.

## Future migrations

A future change must:

1. append a new `(version, sql)` entry to `MIGRATIONS`;
2. bump `SCHEMA_VERSION`;
3. keep every statement backward compatible for an upgrade path (prefer
   `CREATE TABLE IF NOT EXISTS`, `ALTER TABLE ... ADD COLUMN`, guarded seeds);
4. add regression coverage for upgrade-from-previous-version;
5. if it changes the *file contract* in a non-backward-compatible way, bump
   `CMF_FORMAT_MAJOR` and document the refusal behaviour for older readers.
