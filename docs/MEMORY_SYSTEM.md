# Memory System (Unified Memory Foundation)

The unified memory store replaces the ad-hoc `memory_facts` table with a single,
policy-enforced model for *every* kind of memory: personal facts, episodes,
relationships, project state, externally-sourced world knowledge, AI
reflections and hypotheses.

It lives in `src/companion/memory/` and is reached through one gateway,
`MemoryService`, so every read and write obeys the same rules.

## Why "unified"

Before V0.1, "facts" and "candidates" were separate tables with separate
lifecycles. Now one `MemoryItem` carries both *what* a memory says and *how much
we should trust it*, via independent provenance flags rather than a single
opaque "confidence" number.

## Kinds, status and visibility

| Dimension | Values | Meaning |
|-----------|--------|---------|
| `kind` | `PERSONAL_FACT`, `EPISODE`, `RELATIONSHIP`, `PROJECT_STATE`, `WORLD_KNOWLEDGE`, `REFLECTION`, `HYPOTHESIS` | What the memory is about. `REFLECTION`/`HYPOTHESIS` are never treated as facts. |
| `status` | `CANDIDATE`, `ACTIVE`, `NEEDS_REVIEW`, `CONFLICTED`, `RETRACTED`, `OUTDATED` | Lifecycle/trust state. `ACTIVE` means "currently used", **not** "absolutely correct". |
| `visibility` | `PRIVATE`, `SHARED`, `PUBLIC` | Who may read it. `PUBLIC` is world-readable; `PRIVATE`/`SHARED` are subject-scoped. |

## Provenance and trust

Every item records independent booleans:

- `human_confirmed` — a person has explicitly approved this as correct.
- `source_supported` — it cites at least one external source.
- `self_reported` — the subject stated it about themselves.
- `third_party_reported` — someone else stated it about the subject.
- `model_inferred` — an AI produced it (reflection, research synthesis, …).

These are deliberately **not** collapsed into one score. A human-confirmed fact
and a model-inferred hypothesis look different and are treated differently.

## Structure

- `memory_items` — the live version of each record (with `version` for
  optimistic locking).
- `memory_subjects` — who/what the memory is about (many-to-many).
- `memory_sources` — citable origins (utterance, web page, manual, research).
- `memory_evidence` — which source supports which memory, at what verification
  level (`UNVERIFIED`, `SINGLE_SOURCE`, `SUPPORTED`, `CONFLICTED`, `OUTDATED`).
- `memory_links` — relationships between memories (e.g. `conflicts_with`,
  `supports`, `related_to`).
- `memory_revisions` — an immutable snapshot of every version (audit/history).
- `memory_items_fts` — an FTS5 index for text search.

## Rules enforced by `MemoryService`

1. **Requester identity** — every call takes an `AccessContext` supplied by the
   trusted application, never by an LLM. Non-admin requesters only see `PUBLIC`
   items or items where they are a `PERSON` subject.
2. **Subject privacy** — writing a personal/private item about a person requires
   that person to be currently authorised for memory learning.
3. **Optimistic locking** — `update` must present the current `version`; a stale
   version raises `MemoryVersionConflictError`.
4. **Human-confirmed priority** — `propose` never overwrites a human-confirmed
   fact; a conflicting AI proposal becomes `NEEDS_REVIEW` linked to the
   original, and the original stays untouched until a human approves.
5. **Retract vs delete** — `retract` marks `RETRACTED` (kept for audit, excluded
   from default search); `delete` permanently removes the record, its private
   content and its index entries. Retract is reversible, delete is not.

## Search

`MemoryService.search` combines FTS5 phrase matching (AND-of-phrases) with
exact filters on kind, status, visibility, subject, source and time ranges.
FTS5 does not segment CJK well, so when a query has no CJK-unfriendly matches
the repository falls back to a substring `LIKE` search.

## Migration of legacy data

Opening a V0.0 database migrates it in place (append-only migrations, never a
rebuild). Existing `memory_facts` rows are copied once into `memory_items` as
`PERSONAL_FACT`/`ACTIVE`, human-confirmed, self-reported, private, with their
original ids preserved and their `subject_person_id` mapped into
`memory_subjects`. The old tables are left intact — nothing is dropped or
deleted.
