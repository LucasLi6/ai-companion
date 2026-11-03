# Architecture — AI Companion V0.0

## Goals of this release

V0.0 is the *foundation*. It must be stable enough that the following can be
added later without rework:

- a real DeepSeek cognitive engine (V0.1),
- real long-term fact learning (V0.2),
- ASR/TTS and microphone/speaker I/O,
- speaker/voiceprint identification and diarization.

The design principles are: **identity over names**, **explicit consent**,
**one durable event per confirmed input**, **pluggable cognition**, **a policy
boundary between suggestion and execution**, and **recoverable persistence**.

## Module map

```
src/companion/
├── cli.py               argparse surface; thin async command handlers
├── config.py            Config dataclass, default DB path, logging setup
├── runtime.py           composition root (wires every component)
├── recovery.py          RecoveryManager (resume unfinished events after a crash)
├── domain/
│   ├── models.py        pydantic models + id/time factories
│   ├── enums.py         string enums shared across layers
│   └── protocols.py     CognitiveProvider / ResponseProvider / MemoryStore
├── people/registry.py   registration + consent (identity ≠ access rules)
├── storage/
│   ├── schema.py        idempotent DDL + SCHEMA_VERSION
│   ├── database.py      aiosqlite wrapper (WAL, FK on, explicit txn)
│   └── repositories.py  thin parameterised-SQL data access
├── core/
│   ├── orchestrator.py  the ordered pipeline
│   ├── conversation.py  quiet mode / last_action / recent-ids state
│   └── policy.py        final arbiter of whether a suggestion may run
├── memory/store.py      MemoryStore impl (substring match, not semantic)
├── providers/mock.py    deterministic mock cognition + response
└── jobs/
    ├── worker.py        durable single worker + retry/recovery
    └── handlers.py      MEMORY_CONSOLIDATION handler
```

## The processing pipeline

`Orchestrator.process_utterance(event)` acquires a per-conversation
`asyncio.Lock` and then runs these steps in order:

1. **Validate person, status & consent.** If `speaker_person_id` is `None`, the
   person is unknown, their `registration_status` is not `ACTIVE`
   (`suspended`/`revoked`), or `allow_processing` is false, the event is
   **rejected** with `PERMISSION_DENIED` and *nothing* is written. Role is a
   descriptive label and never bypasses consent or status.
2. **Load the conversation.** With `allow_persistence` the conversation is
   created/loaded (quiet mode, last action, recent ids). Without it the
   conversation is read-only: an existing quiet mode is still honoured but no
   conversation row is created or updated.
3. **Deduplicate / conflict / resume.** Same `event_id` + same identity
   fingerprint that already `COMPLETED` returns `DUPLICATE`. Same id +
   *different* fingerprint returns `CONFLICT`. A partial (`PROCESSING`/
   `FAILED`) event is *resumed*, never cleared.
4. **Persist input** (only when `allow_persistence`).
5. **Retrieve permitted memories** from `MemoryStore`.
6. **Cognitive analysis — reuse or compute.** If a saved analysis exists it is
   reused without calling the provider again; otherwise the provider is called
   and the result validated (`event_id` must match).
7. **Apply policy.** `Policy.decide` finalises the action (V0.0: quiet mode
   downgrades `RESPOND` to `WAIT`).
8. **Persist analysis + candidates + memory job in one short transaction.**
   Candidates are written only when `allow_memory_learning`; the job only when
   a worker is wired. The transaction never spans a provider call.
9. **Generate a response** only when the final decision is `RESPOND` *and* no
   action has been saved yet.
10. **Persist the `AgentAction`** (or reuse an already-saved one), record the
    conversation's last action, and mark the event `COMPLETED`.

The pipeline is **resumable**: every durable stage leaves a database record
(utterance → analysis → candidates/job → action), so a retry or recovery
re-uses whatever already succeeded. A `ResponseProvider` failure therefore never
discards a valid analysis or strands a memory job, and an already-persisted
action is never re-emitted. `IGNORE` never skips persistence; `WAIT` is a
normal decision; a provider exception leaves the event `FAILED` (retryable)
rather than `COMPLETED`.

### Ordering

Within a conversation, events are processed strictly in **submission order** —
the order in which `process_utterance` acquires that conversation's lock.
`occurred_at` is caller-supplied timing, not a scheduling guarantee: a late
event (an older `occurred_at` arriving after newer ones) is processed when it
arrives, not re-ordered by timestamp. Different conversations run concurrently
under the shared SQLite connection lock.

### Why policy is separate from cognition

The mock (or, later, DeepSeek) *suggests* `RESPOND`/`WAIT`/`IGNORE`. The
`Policy` layer decides whether that suggestion is allowed to execute — today it
only enforces quiet mode, but this is where future gating (per-person
permissions, skill allow-lists, interruption rules) lives. Model output is
never executed unchecked.

## Data flow for a RESPOND event

```
say --person P --to agent --text "你好"
  → UtteranceEvent(event_id=evt_…, speaker_person_id=P, addressee_hint=agent)
  → person P registered & consent.allow_processing ✓
  → conversation "default" created/loaded
  → utterance persisted (status PROCESSING)
  → memories = [] (no hand-authored facts)
  → MockCognitiveProvider → action=RESPOND, reason=direct_agent_address,
      memory_candidates=[mock candidate]
  → Policy → RESPOND (quiet_mode false)
  → cognitive_analyses row + memory_candidates row + processing_jobs row (PENDING)
  → MockResponseProvider → "模拟回复：我收到你的话了。"
  → agent_actions row (RESPOND) + utterance status COMPLETED
  → (worker) MEMORY_CONSOLIDATION validates candidate → CONSOLIDATED → COMPLETED
```

## Storage

- SQLite via `aiosqlite`, with `PRAGMA foreign_keys=ON`,
  `PRAGMA journal_mode=WAL`, and `PRAGMA busy_timeout=5000`.
- All SQL is parameterised; the schema is append-only migrations (`CREATE …
  IF NOT EXISTS`) that never drop or truncate data.
- **Concurrency.** All access to a `Database` is serialised through a
  re-entrant async lock: `execute` holds it for one statement, `transaction`
  holds it for the whole `BEGIN … COMMIT/ROLLBACK` span. A statement can
  therefore never be injected into another task's open transaction. The lock is
  per-connection; correctness across *separate* connections (or processes)
  relies on database-level `UNIQUE` constraints and atomic `UPDATE … WHERE`
  statements (e.g. job claiming).
- Transactions are short and local; they are **never** held across a provider
  call (which may be slow or remote).
- **Migrations.** `schema.py` holds an ordered, append-only `MIGRATIONS` list.
  `Database.initialize` applies any missing versions in order, is a no-op when
  already current, and **refuses** to open a database stamped with a newer,
  unknown version.
- All persisted timestamps are timezone-aware UTC ISO-8601; durations use
  `time.monotonic()` and are only logged, never persisted as wall-clock deltas.

## Jobs

Jobs live in `processing_jobs`, not an `asyncio.Queue`. Each has a unique
`job_key` (deduplication), an attempt counter, a max attempt count, and a
status (`PENDING`/`RUNNING`/`COMPLETED`/`FAILED`). Claiming is an atomic,
conditional `UPDATE … WHERE status = 'PENDING' AND attempts < max_attempts`;
a worker runs a handler only if its claim won `rowcount == 1`, so concurrent
workers (even on separate connections) never both run the same job. On startup
the worker resets leftover `RUNNING` jobs to `PENDING`; idempotent handlers
guarantee this never duplicates side effects. A failing job is requeued
automatically until `max_attempts`, then marked `FAILED` with `last_error`;
`jobs retry` requeues it with attempts reset.

The `MEMORY_CONSOLIDATION` handler re-checks the subject's *current*
authorisation at run time: if learning consent was revoked or the person was
suspended/revoked after the job was queued, the candidate is marked `SKIPPED`
rather than `CONSOLIDATED` — no new long-term fact is learned.

## Event recovery

A crash can leave an utterance in `PROCESSING` after some durable stages
completed. `RecoveryManager` (`recovery list` / `recovery retry EVENT_ID`)
finds those events and re-submits them through the orchestrator, which resumes
from the records already present. It never blindly resets every `PROCESSING`
event to `PENDING` and re-runs the whole pipeline.

## Observability

Structured logs carry `event_id`, `conversation_id`, `speaker_person_id`,
`provider`, `processing_stage`, `duration_ms`, `status`, `reason_code` so a
single event can be traced end to end. Raw utterance text is intentionally
never copied into logs; it lives only in the authorised database record.

## Extension points (kept, but not implemented)

- **V0.1 DeepSeek**: replace `MockCognitiveProvider` with a
  `DeepSeekCognitiveProvider` behind the same `CognitiveProvider` protocol.
- **V0.2 memory learning**: replace the `MEMORY_CONSOLIDATION` handler body
  with real extraction/validation/conflict resolution.
- **ASR/TTS**: produce a `UtteranceEvent` from ASR instead of CLI text; emit
  real audio after `RESPOND`.
- **Speaker ID**: replace the manually-supplied `speaker_person_id` with a
  verified recognition result.
- **Skills**: extend the `Policy`/action layer to run registered skills, never
  arbitrary model-generated code.
