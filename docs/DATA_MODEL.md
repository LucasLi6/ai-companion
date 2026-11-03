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

## Consent semantics

The three consent flags are **independent**:

- `allow_processing` — may this person enter the cognitive flow at all? If
  false, the input is rejected with `PERMISSION_DENIED` and *nothing* is
  persisted.
- `allow_persistence` — may their utterances be durably stored? If false, the
  cognitive flow still runs (and a response may be produced) but nothing is
  written to disk. *Processing does not imply persistence.*
- `allow_memory_learning` — may memory candidates be created and consolidated?
  If false, no candidates are saved and no memory job is enqueued.

V0.0 **never defaults these to `True`**. The CLI sets them explicitly via
`person add` flags.

Unregistered speakers may carry a transient `speaker_label` within a single
conversation, but they are never promoted to a permanent profile and never
enter persistent processing.

## Event lifecycle

```
PENDING → PROCESSING → COMPLETED   (successful pipeline)
                 └──→ FAILED       (provider error; retryable)
```

`processing_status` lives on `utterances`. A `FAILED` event can be re-submitted
with the same id/content; stale cognitive outputs are cleared and the pipeline
re-runs (idempotent by construction). Re-submitting a `COMPLETED` event returns
`DUPLICATE`; re-submitting the same id with different content returns
`CONFLICT`.

## Cognitive results vs. final decisions

`cognitive_analyses` stores the provider's **suggestion** (`action`,
`reason_code`, `provider`, `meaning`, `referenced_memory_ids`,
`response_style`). `agent_actions` stores the **final decision** after policy
(`action_type`, `reason_code`, `response_text`). They differ, for example, when
quiet mode downgrades a `RESPOND` suggestion to a `WAIT` decision with
`reason_code=quiet_mode`.

## Memory candidates and facts

A `memory_candidate` is a proposal with an `operation` (`ADD`/`UPDATE`/`LINK`/
`SKIP`) and a `status` (`PENDING`/`CONSOLIDATED`/…). V0.0 stores candidates and
the `MEMORY_CONSOLIDATION` job validates their associations (source event and
subject person must exist) and marks them `CONSOLIDATED`. **No candidate is
ever promoted to `memory_facts` in V0.0.**

## Future multi-subject facts

`memory_facts.subject_person_id` currently names a single subject. Facts that
relate *several* people (e.g. "A and B work together") must not force every
fact to have exactly one subject. The planned change is to add a join table:

```sql
CREATE TABLE fact_subjects (
    fact_id  TEXT NOT NULL REFERENCES memory_facts(fact_id),
    person_id TEXT NOT NULL REFERENCES persons(person_id),
    PRIMARY KEY (fact_id, person_id)
);
```

This keeps `memory_facts` single-subject-optional while allowing N-way
associations without breaking the existing schema.
