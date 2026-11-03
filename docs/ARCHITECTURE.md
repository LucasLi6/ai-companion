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

`Orchestrator.process_utterance(event)` runs these steps in order:

1. **Validate person & consent.** If `speaker_person_id` is `None`, the person
   is unknown, or `allow_processing` is false, the event is **rejected** with
   `PERMISSION_DENIED` and *nothing* is written (not the text, not a log line
   carrying the text).
2. **Ensure the conversation exists** (the utterance row has an FK to it) and
   load its persistent state (quiet mode, last action) plus recent utterance ids.
3. **Deduplicate.** Same `event_id` + same content that already `COMPLETED`
   returns `DUPLICATE` without re-running. Same `event_id` + *different* content
   returns `CONFLICT` and never overwrites. A previously `FAILED`/partial event
   is cleared of stale cognitive outputs and retried.
4. **Persist input** (only when `allow_persistence`).
5. **Retrieve permitted memories** from `MemoryStore` (hand-authored facts).
6. **Call the cognitive provider** — a *suggestion*, not a decision.
7. **Validate the analysis** (`event_id` must match).
8. **Apply policy.** The `Policy` layer owns the final decision. V0.0 rule:
   a `RESPOND` suggestion is downgraded to `WAIT` with reason `quiet_mode`
   while the conversation is quiet.
9. **Persist analysis + memory candidates** (candidates only when
   `allow_memory_learning`).
10. **Enqueue a memory job** (`MEMORY_CONSOLIDATION`, keyed by event id, so
    re-submission never duplicates it).
11. **Update conversation state** (last action).
12. **Generate a response** only when the final decision is `RESPOND`.
13. **Persist the `AgentAction`** and mark the event `COMPLETED`.

`IGNORE` never skips steps 4–10; `WAIT` is a normal decision, not an error; a
provider exception leaves the event `FAILED` (retryable) rather than
`COMPLETED`.

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

- SQLite via `aiosqlite`, with `PRAGMA foreign_keys=ON` and
  `PRAGMA journal_mode=WAL`.
- All SQL is parameterised; the schema is idempotent (`CREATE … IF NOT
  EXISTS`) and never drops or truncates data.
- Transactions are short and local; they are **never** held across a provider
  call (which may be slow or remote).
- All persisted timestamps are timezone-aware UTC ISO-8601; durations use
  `time.monotonic()` and are only logged, never persisted as wall-clock deltas.

## Jobs

Jobs live in `processing_jobs`, not an `asyncio.Queue`. Each has a unique
`job_key` (deduplication), an attempt counter, a max attempt count, and a
status (`PENDING`/`RUNNING`/`COMPLETED`/`FAILED`). On startup the worker resets
leftover `RUNNING` jobs to `PENDING`; idempotent handlers guarantee this never
duplicates side effects. A failing job is requeued automatically until
`max_attempts`, then marked `FAILED` with `last_error`; `jobs retry` requeues it
with attempts reset.

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
