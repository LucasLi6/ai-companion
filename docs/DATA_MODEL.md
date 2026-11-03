# Data model — AI Companion V0.0

## Identifiers

Every durable entity uses a stable, unique id with a human-readable prefix; no
id is derived from a person's name (duplicate names are allowed):

| Prefix  | Entity            |
|---------|-------------------|
| `person_` | registered person |
| `evt_`    | utterance event   |
| `ana_`    | cognitive analysis|
| `act_`    | agent action      |
| `memc_`   | memory candidate  |
| `job_`    | processing job    |
| `fact_`   | memory fact       |

Conversation ids are caller-supplied strings (the CLI defaults to `default`).

## Tables

| Table                | Purpose                                                            |
|----------------------|--------------------------------------------------------------------|
| `schema_version`     | migration guard (V0.0 = 1)                                          |
| `persons`            | identity + role + registration status                               |
| `consents`           | per-person `allow_processing` / `allow_persistence` / `allow_memory_learning` |
| `conversations`      | quiet mode + last action (persistent session state)                 |
| `utterances`         | authorised, persisted input (one row per confirmed event)           |
| `cognitive_analyses` | one provider analysis per event (`UNIQUE(event_id)`)                |
| `agent_actions`      | one final decision per event (`UNIQUE(event_id)`)                   |
| `memory_candidates`  | proposed memories (stored only, never promoted in V0.0)             |
| `memory_facts`       | reserved for long-term facts; V0.0 never auto-populates it          |
| `processing_jobs`    | durable, deduplicated, recoverable background jobs                  |

## Event lifecycle

```
PENDING → PROCESSING → COMPLETED   (successful pipeline)
                 └──→ FAILED       (provider error; retryable)
```

`processing_status` lives on `utterances`. A `FAILED` (or crash-left
`PROCESSING`) event can be re-submitted with the same id/content, and the
pipeline **resumes** from the durable records already present instead of
clearing them:

- an existing `cognitive_analyses` row is reused (the provider is not called
  again),
- existing `memory_candidates` and their `processing_jobs` row are left intact
  (so a completed job never strands new candidates),
- an existing `agent_actions` row is reused and its response is **not**
  re-emitted.

Re-submitting a `COMPLETED` event returns `DUPLICATE`; re-submitting the same id
with a different identity fingerprint returns `CONFLICT`. `recovery list` /
`recovery retry` expose this resume behaviour for events left by a crash.

### Event identity fingerprint

`UtteranceEvent.identity_fingerprint()` returns the immutable content fields
that define an event's identity:

```
(conversation_id, speaker_person_id, speaker_label, text, source, addressee_hint)
```

`event_id` is the primary key; `occurred_at` and `processing_status` are *not*
part of identity (the former is caller timing, the latter is lifecycle state).
Two submissions with the same id and a different fingerprint are a `CONFLICT`.

## Schema versioning

`schema_version` records the applied version (V0.0 = 1; V0.2A = 8).
`schema.py` holds an ordered, append-only `MIGRATIONS` list;
`Database.initialize` applies missing versions in order, each inside one
transaction, is a no-op when already current, and refuses to open a database
stamped with a newer, unknown version. A database is never deleted to "fix" a
version mismatch. Migration 8 adds the V0.2A columns/tables and stamps legacy
candidates as `NO_MEMORY` in `memory_candidate_writes` rather than promoting
them.

## Consent semantics

The three consent flags are **independent**, and are gated by the person's
`registration_status` (`active`/`suspended`/`revoked`):

- `allow_processing` — may this person enter the cognitive flow at all? If
  false, the input is rejected with `PERMISSION_DENIED` and *nothing* is
  persisted.
- `allow_persistence` — may their utterances be durably stored? If false, the
  cognitive flow still runs (and a response may be produced) but nothing is
  written to disk — not the utterance, analysis, action, candidate, job, or any
  conversation metadata. *Processing does not imply persistence.*
- `allow_memory_learning` — may memory candidates be created and consolidated?
  If false, no candidates are saved and no memory job is enqueued.

A person must be `active` *and* hold `allow_processing=True` to be processed.
`SUSPENDED` and `REVOKED` deny new input regardless of prior consent; `owner`
role does not bypass consent or status. The consolidation job re-checks
authorisation at run time, so revoking learning consent (or suspending/revoking
the person) after a job is queued causes the candidate to be `SKIPPED` rather
than consolidated. Revoking authorisation never deletes existing history.

Unregistered speakers may carry a transient `speaker_label` within a single
conversation, but they are never promoted to a permanent profile and never
enter persistent processing.

## Cognitive results vs. final decisions

`cognitive_analyses` stores the provider's **suggestion** (`action`,
`reason_code`, `provider`, `meaning`, `referenced_memory_ids`,
`response_style`) plus the V0.2A structured fields (`intent`, `addressed_to`,
`silence_requested`, `reply_necessity`, `conversation_state`).
`agent_actions` stores the **final decision** after policy (`action_type`,
`reason_code`, `response_text`) and, separately, `model_suggested_action`,
`policy_final_action` and `override_reason`. They differ, for example, when
quiet mode downgrades a `RESPOND` suggestion to `WAIT`, when an explicit
current-turn silence forces `IGNORE`, or when a human addressee forces
`IGNORE`. The model's suggestion is never overwritten.

## Memory candidates, write-through and assignments

A `memory_candidate` is a proposal with an `operation` (`ADD`/`UPDATE`/`LINK`/
`SKIP`), a `status` (`PENDING`/`CONSOLIDATED`/`SKIPPED`/`NEEDS_REVIEW`/
`FAILED`) and V0.2A semantic hints (`assertion_type`, `certainty_category`,
`explicit_remember_request`, `topic`, `kind`, `source_evidence_span`). The
`MEMORY_CONSOLIDATION` job routes every candidate through the trusted
`MemoryWriter`, which writes a durable memory (or decides truthfully that none
should be written) and records exactly one outcome in
`memory_candidate_writes`:

- `SAVED` / `UPDATED` / `ALREADY_EXISTS` / `PENDING_REVIEW` / `DENIED` /
  `FAILED` / `NO_MEMORY`;
- a direct self-report may become an `ACTIVE`, `self_reported` item; uncertain
  and third-party proposals stay `CANDIDATE`; hypothetical/quoted/empty
  proposals write nothing;
- consent is re-checked at run time (`DENIED` blocks new writes).

`memory_provenance` records the source event, candidate, speaker, subject,
assertion, certainty and topic for each written memory. The memory, provenance
and mapping commit together; a retry reads the mapping (or recovers from the
uniquely indexed provenance candidate id) instead of writing twice.

`memory_assignments(memory_id, person_id, created_at, created_by)` associates a
memory with people for retrieval. It does not change meaning or visibility and
is never an access grant. Self-reported facts are auto-assigned to their
speaker; assignment to other people is an explicit (admin) action.

## Multi-subject memories

The unified store already supports N-way associations without breaking the
legacy schema: `memory_subjects(memory_id, subject_type, subject_id)` records
who/what a memory *is about* (with `subject_type` allowing `PERSON` or other
types), while `memory_assignments(memory_id, person_id)` records who the memory
is *associated with for retrieval*. They are deliberately separate: a memory
may describe one person but be assigned to several, and an assignment never
grants access. The legacy `memory_facts` table is left intact for older data;
new writes go through `memory_items`.
