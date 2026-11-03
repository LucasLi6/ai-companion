# CMF 1.0 — Schema Reference

`SCHEMA_VERSION = 9`. Migrations `1..9` are append-only and each commit
atomically with its `schema_version` stamp. Migration 9 is the CMF migration;
older migrations are untouched.

Every table lives in the same SQLite file. Timestamps are timezone-aware UTC
ISO-8601 strings. Foreign keys are enforced (`PRAGMA foreign_keys = ON`).

## Identity / manifest

| Table | Purpose | Key notes |
| --- | --- | --- |
| `cmf_manifest` | Singleton file identity | `singleton = 1`, `agent_id UNIQUE`, `format_version`, `lineage_id`, `lineage_parent_agent_id`, `clone_of` |
| `agent_identity` | Display name/pronouns/description | `agent_id` FK → manifest, `revision` counter |
| `agent_persona` | Nine `0..100` stylistic traits | `agent_id` FK → manifest, `revision` |
| `persona_revisions` | Traceable persona changes | `changes` JSON, `changed_by`, `reason` |

Persona traits: `humor`, `warmth`, `curiosity`, `formality`, `verbosity`,
`assertiveness`, `empathy`, `playfulness`, `patience` — each `INTEGER CHECK
(0 <= value <= 100)`.

## People and relationships

| Table | Purpose | Key notes |
| --- | --- | --- |
| `persons` | Registered people | `registration_status` |
| `consents` | Per-person consent flags | `allow_processing`, `allow_persistence`, `allow_memory_learning`, `allow_external_ai` |
| `person_profiles` | Tier + style overrides per person | `familiarity`, `relationship_tier`, `style_overrides` JSON |
| `relationship_presets` | Per-tier style deltas | seeded for `STRANGER`/`ACQUAINTANCE`/`FAMILIAR`/`CLOSE` |
| `person_interaction_profiles` | Supplementary interaction state | |
| `relationship_history` | Tier change log | `old_tier`, `new_tier`, `actor`, `reason` |

## Memory

| Table | Purpose |
| --- | --- |
| `memory_items` | Unified memory records (kind, status, visibility, version) |
| `memory_subjects` | Subject associations (real, loaded — never fabricated) |
| `memory_sources` | Provenance sources |
| `memory_evidence` | Per-source verification level |
| `memory_links` | Between-memory links |
| `memory_revisions` | Append-only content history |
| `memory_items_fts` | FTS5 full-text index |
| `memory_provenance` | Current origin + `entity_key` / `attribute_key` claim key |
| `memory_provenance_events` | Append-only correction/promotion history (old/new versions) |
| `memory_candidate_writes` | Idempotent candidate → memory mapping + truthful outcome |
| `memory_assignments` | Memory/person association (access still enforced) |

`memory_provenance.entity_key` / `attribute_key` and
`memory_provenance_events` are added in migration 9. The claim-key index is
`idx_provenance_claim`.

## Cognition / jobs

| Table | Purpose |
| --- | --- |
| `conversations` | Conversations (incl. `quiet_mode`) |
| `utterances` | Utterance events |
| `cognitive_analyses` | One analysis per event (`UNIQUE(event_id)`) |
| `agent_actions` | Derived actions |
| `memory_candidates` | Model-proposed candidate writes |
| `processing_jobs` | Lease-gated durable jobs |
| `research_tasks` | Research queue (owner/lease/attempts/memory_id) |
| `app_settings` | Key/value settings (policy + idle gates) |
| `audit_log` | Audit trail |

`research_tasks` gains `project_id` and `run_id` columns in migration 9 via
`ALTER TABLE` (backward compatible).

## Research state

| Table | Purpose |
| --- | --- |
| `research_projects` | One project per research question |
| `research_runs` | Runs with `provider`, `budget_used`, `error` |
| `search_queries` | Per-run queries and result counts |
| `web_sources` | Source candidates (`retrieved_at` ≠ `published_at`, `trust_level`) |
| `source_excerpts` | Bounded quoted material |
| `knowledge_claims` | Proposed claims (`status`, `verification_level`, `confidence`) |
| `claim_evidence` | claim ↔ source ↔ excerpt links |
| `research_findings` | Synthesized drafts |
| `source_refresh_rules` | Staleness/refresh intervals |

## Learning state

| Table | Purpose |
| --- | --- |
| `curiosity_queue` | Queued research questions |
| `reflection_records` | Proposed reflections |
| `learning_runs` | Idle-cycle runs (`kind`, `provider`, `budget_json`, `status`) |
| `learning_checkpoints` | Per-stage run checkpoints |
| `knowledge_refresh_jobs` | Scheduled source/claim refresh |

## Goals

| Table | Purpose |
| --- | --- |
| `goals` | Durable goals (`status`, `priority`, `next_action`, `due_at`) |
| `goal_checkpoints` | Per-goal checkpoints |
| `goal_dependencies` | Goal dependency edges (cycle-checked at write time) |

## Reserved (not built)

| Table | Purpose |
| --- | --- |
| `media_assets` / `media_annotations` / `media_links` | Media model |
| `embedding_models` / `memory_embeddings` / `embedding_index_metadata` / `retrieval_feedback` | Semantic search |

## Migration summary

| Version | Introduces |
| --- | --- |
| 1 | Base tables (persons, consents, conversations, utterances, analyses, `memory_facts`) |
| 2 | Job lease/ownership columns |
| 3 | Unified memory foundation (memory_items, subjects, sources, evidence, links, revisions, FTS) |
| 4 | Management UI + idle research |
| 5 | Privacy columns (`allow_external_ai`) + research lease columns |
| 6 | Demote mock research knowledge (truthfulness) |
| 7 | Model-proposed `memory_candidates` |
| 8 | V0.2A durable write-through, truthful outcomes, provenance |
| 9 | **CMF 1.0**: manifest, agent core, relationships, provenance events, research/learning/goals durability, media + embedding reservations |

## Tables reused / extended / added by CMF (migration 9)

- **Reused unchanged**: `persons`, `consents`, `conversations`, `utterances`,
  `cognitive_analyses`, `memory_items`, `memory_subjects`, `memory_sources`,
  `memory_evidence`, `memory_links`, `memory_revisions`, `app_settings`,
  `audit_log`, `processing_jobs`, `research_tasks` (columns added).
- **Extended**: `memory_provenance` (+`entity_key`, `attribute_key`);
  `research_tasks` (+`project_id`, `run_id`).
- **Added**: `cmf_manifest`, `agent_identity`, `agent_persona`,
  `persona_revisions`, `person_profiles`, `relationship_presets`,
  `person_interaction_profiles`, `relationship_history`,
  `memory_provenance_events`, and all research/learning/goal/media/embedding
  tables listed above.
