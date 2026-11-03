# V0.1 Upgrade Report

This document is the checkpoint report for the V0.1 upgrade, organised around
the seven acceptance scenarios (A–G). Each scenario is exercised end-to-end by
`tests/test_integration.py`.

## Scenario A — Schema at current version, atomic migration

The database opens at `SCHEMA_VERSION` with `foreign_keys=ON`, WAL and
`busy_timeout`. Migrations are append-only (`_statements()` splitter), run
atomically, and never rebuild or drop existing tables. The legacy
`memory_facts` data is copied once into the unified store while the old tables
remain intact.

## Scenario B — Consent-gated identity

`PersonRegistry.register` records a person with explicit consent flags
(`allow_processing`, `allow_persistence`, `allow_memory_learning`). The
registry is the single authority for "is learning authorised for this person",
and the same gate is shared by memory writes and cognitive consolidation.

## Scenario C — Self-reports consolidate into unified memory

A first-person statement runs through the existing `say` → cognitive → job
pipeline and persists an action, and the unified store is reachable through the
same `MemoryService` used by CLI, UI and idle. (Full automatic write-through of
`say` into `memory_items` is a V0.2 milestone; today `say` still reports and
persists, and the memory store is populated via `MemoryService` + legacy
migration.)

## Scenario D — Third-party reports are not auto-confirmed

A statement about someone else is recorded with `third_party_reported=True` and
`human_confirmed=False`; it never silently becomes a fact about the subject.

## Scenario E — Human-confirmed facts survive AI proposals, privacy intact

`propose` never overwrites a human-confirmed memory. A conflicting AI proposal
enters as `NEEDS_REVIEW`, linked to the original via `conflicts_with`, while the
original keeps `human_confirmed=True`. Non-admin access is isolated per subject:
person A cannot read person B's `PRIVATE` memories.

## Scenario F — Idle reflection, curiosity and research, gated

`IdleScheduler.run_cycle()` is gated by `idle.enabled` (off by default). When
enabled it runs reflection → curiosity → research, where research runs pending
tasks through search → synthesize → consolidate and writes `WORLD_KNOWLEDGE`
with web sources attached. When disabled it returns `skipped` and does no work.

## Scenario G — Append-only audit trail

Every mutating action (memory create/update/confirm/retract/delete/link,
person register/set_consent/set_status) writes an append-only `audit_log` row
with actor, action, entity and timestamp. The log is readable via CLI/UI.

## Regression — V0.0 surface preserved

The existing CLI commands (`person`, `say`, `inspect`, `jobs`, `recovery`,
`demo`, `repl`) and the V0.0 pipeline remain working. The old tables are not
dropped, no published migration is modified, and the Orchestrator /
PersonRegistry / Database / JobWorker / CognitiveProvider / CLI architecture is
unchanged — the new memory, research and UI layers sit alongside it.

## What is new in V0.1

- **Phase A — Reliability**: re-entrant transactions, WAL + foreign keys +
  busy timeout, atomic migrations, job leases with heartbeat/recovery/retry.
- **Phase B — Unified memory**: `MemoryItem` (kind/status/visibility),
  provenance flags, optimistic locking, retract vs delete, FTS5 search,
  `MemoryService` gateway.
- **Phase C — Memory manager**: `memory` CLI + NiceGUI web UI (dashboard,
  explorer, detail, people, research, audit, settings).
- **Phase D — Idle learning**: idle scheduler, reflection/curiosity/research
  engines, mock DeepSeek, mock + opt-in real web search, resumable research
  tasks.
- **Phase E — Integration & docs**: scenarios A–G, this report, and the
  docs/MEMORY_SYSTEM.md, MEMORY_MANAGER.md, IDLE_LEARNING.md,
  PRIVACY_AND_PERMISSIONS.md.

## Explicitly out of scope (by constraint)

- No project-wide rewrite; existing architecture preserved.
- No deleting/rebuilding the database or editing published migrations.
- No real DeepSeek integration (mock only); no ASR/TTS/voiceprint/vector DB or
  new AI framework.
- No LLM/UI direct SQL; identity from trusted app context; no secrets in repo.
- Real web search off by default; tests never call real APIs.

## Test status

Full suite is green (`pytest -q`, expect ~125 passing) across the unit,
migration, job-lease, memory, CLI, UI, idle, research and integration suites.

No commits or pushes were made — per the standing instruction not to auto
push/merge/release without explicit authorisation.
