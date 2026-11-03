# CMF 1.0 — Companion Memory Format Specification

The Companion Memory Format (CMF) is a portable, single-file representation of
the complete persistent state of **one** AI companion: its identity,
personality, relationships, memories, knowledge, research state and learning
state.

A CMF file is a SQLite database. The preferred extension is `.memory`; the
legacy `.db`, `.sqlite` and `.sqlite3` extensions remain accepted on open so an
existing database is never forcibly renamed. All data lives in that one file —
the system deliberately creates no sidecar database and no second persistent
store.

## 1. File identity

Every valid CMF 1.0 file carries three independent identifiers:

| Identifier | Where | Meaning |
| --- | --- | --- |
| `application_id` | SQLite header field | `0x434D4601` ("CMF1"). Distinguishes a Companion memory file from an arbitrary SQLite database. |
| `format_version` | `cmf_manifest.format_version` | `"1.0"`. The portable file contract, independent of the schema version. |
| `schema_version` | `schema_version` table | The append-only migration level inside the format. Currently `9`. |

A reader that encounters a newer **major** format version must refuse to open
the file and must not modify it. A reader that encounters a newer schema
version must likewise refuse. A lower schema version is upgraded in place on
open.

The **agent identity** is stored in `cmf_manifest.agent_id`. It is stable
across close/reopen, backup and restore. Only a *clone* changes it (see
section 5).

## 2. Manifest and Agent Core

- `cmf_manifest` is a singleton table (enforced by `CHECK (singleton = 1)`).
  It holds the stable `agent_id`, the `format_version`, an optional
  `display_name`, and lineage fields (`lineage_id`,
  `lineage_parent_agent_id`, `clone_of`).
- `agent_identity` holds the display name, pronouns, description and a
  revision counter. It is foreign-keyed to the manifest, so the identity can
  never disagree with the manifest's agent id.
- `agent_persona` holds nine stylistic integer traits in `0..100`: `humor`,
  `warmth`, `curiosity`, `formality`, `verbosity`, `assertiveness`, `empathy`,
  `playfulness`, `patience`. Personality is **stylistic, never probabilistic**
  and never overrides policy.
- `persona_revisions` records every persona change (who, when, reason), so a
  persona proposal is traceable and never authoritative.

The three pre-existing policy values — `policy.response_conservatism`,
`policy.interruption_threshold`, `policy.curiosity_level` — remain in
`app_settings` and are passed through into the effective context. They are
**not** duplicated as persona integer fields: the persona's `curiosity` trait
is conversational style, while `policy.curiosity_level` governs whether the
runtime may initiate background work.

## 3. Relationships

Relationships are descriptive, and **never** a grant:

- A familiarity tier is one of `STRANGER`, `ACQUAINTANCE`, `FAMILIAR`,
  `CLOSE`.
- `relationship_presets` stores a `style_deltas` JSON object per tier (the
  seeded defaults are reproduced in `companion.agent.constants`).
- `person_profiles` stores a person's current tier and per-person style
  overrides; `person_interaction_profiles` stores supplementary interaction
  state; `relationship_history` records tier changes.

Role is not consent, familiarity is not authorization, and an assignment is
not sharing. Familiarity is never inferred from message count — only an
explicit operator action changes it.

### Effective context resolution order

```
Agent Core (base persona)
  → relationship preset delta for the person's tier
  → absolute per-person overrides
  → temporary per-conversation context
  → local Policy (authoritative; response only if allowed)
```

This is computed per turn into an immutable `EffectiveAgentContext`; there are
no mutable globals.

## 4. Memory, provenance and privacy

The unified memory store (`memory_items`, `memory_subjects`, `memory_sources`,
`memory_evidence`, `memory_links`, `memory_revisions`, FTS5,
`memory_provenance`, `memory_candidate_writes`, `memory_assignments`) is
preserved and extended, not replaced.

- **Claim keys** (`memory_provenance.entity_key` / `attribute_key`) make
  conflict detection narrow: only the same attribute of the same entity may
  replace a memory. Unrelated preferences or projects coexist.
- **Grounding** prevents a verbatim span from promoting an unsupported claim.
  "I wrote code today" never authoritatively becomes "I especially love
  programming".
- **`memory_provenance_events`** is append-only and preserves every
  `created`/`promoted`/`corrected`/`updated` event with the memory id, source
  event id, candidate id, speaker, subject, old/new versions, reason and
  timestamp. Originals are kept in `memory_revisions`; nothing is discarded.
- **Assignments** affect retrieval only after the same visibility check as any
  other access. An assignment can never reveal a private memory about someone
  else.
- Consent flags (`allow_processing`, `allow_persistence`,
  `allow_memory_learning`, `allow_external_ai`) are independent and unchanged.
  `allow_external_ai` is required before any person subject is sent to an
  external model; multi-subject memories require **all** subjects to consent.

## 5. Clone lineage

A **backup/export/import** preserves `agent_id` exactly. A **clone** copies
the full memory file but assigns a new `agent_id`, sets
`lineage_parent_agent_id` to the source agent id, preserves the shared
`lineage_id`, and records `clone_of`. The two files are then independent
identities with a traceable common ancestor.

## 6. Research, sources, claims

Research state is durable and resumable across restart:

- `research_projects`, `research_runs`, `search_queries` record the job and
  its runs and queries with provider and budget.
- `web_sources` stores a source candidate with `retrieved_at` **separate**
  from `published_at`; `source_excerpts` stores bounded (truncated) quoted
  material; `source_refresh_rules` expresses staleness, never falsity.
- `knowledge_claims` + `claim_evidence` + `research_findings` keep the
  synthesis as a *proposal* (`PROPOSED` / `UNVERIFIED`), never a trusted fact.

A search hit is not verified evidence; an excerpt is not an instruction; a
summary is not a trusted fact. A mock provider's name is recorded so it can
never be presented as real.

## 7. Learning

`curiosity_queue`, `reflection_records`, `learning_runs`,
`learning_checkpoints` and `knowledge_refresh_jobs` persist the idle-learning
cycle. A run records its provider and budget; a checkpoint marks how far the
cycle got. A checkpoint is bookkeeping, not permission: resuming still
requires the same gates as a fresh run. Idle learning is disabled by default
(`idle.enabled=false`); the real DeepSeek idle provider is opt-in and gated
independently.

## 8. Goals and tasks

`goals`, `goal_checkpoints` and `goal_dependencies` persist durable intent.
The file stores *plans*, never execution: `next_action` is a plan, and a
restored tool action must be re-authorised by the runtime before it runs.
Dependency readiness is planning data only.

## 9. Reserved: media and embeddings

`media_assets`, `media_annotations`, `media_links` document the media model
without building the full system. `embedding_models`, `memory_embeddings`,
`embedding_index_metadata` and `retrieval_feedback` reserve semantic search in
the same database. Embeddings are rebuildable and are not required now; any
privacy filtering happens **before** disclosure.

## 10. Safety invariants

- One SQLite file per identity; no separate persistent databases.
- Identity is separate from runtime. The memory file never overrides host
  safety, authorization or tool restrictions.
- No credentials or API keys are stored in the file.
- A corrupt or foreign file is refused and never silently becomes a new empty
  companion.
- Opening a file never executes code.
- All migrations are append-only; published migrations are never edited.
