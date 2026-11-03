# AI Companion — V0.0 Foundation

AI Companion is an event-driven, multi-person companion core. V0.0 builds the
foundation: person identity and explicit consent, durable event persistence,
a replaceable cognitive/response provider boundary, behaviour policy, and
recoverable background memory jobs — so that real DeepSeek cognition, ASR/TTS,
speaker recognition and long-term memory can be added later without reworking
the architecture.

> V0.0 uses a **mock** cognitive provider. It does **not** understand language,
> recognise voices, or learn real long-term facts. See
> [Known limitations](#known-limitations).

---

## What V0.0 does

- Multi-person registration with unique, stable ids (duplicate names allowed).
- Explicit per-person consent: `allow_processing`, `allow_persistence`,
  `allow_memory_learning` — **never defaulted on** — plus a registration
  status (`active`/`suspended`/`revoked`) that gates processing independently
  of role.
- A single validated `UtteranceEvent` per confirmed input, with deduplication
  and conflict detection on a documented identity fingerprint.
- A pluggable `CognitiveProvider` / `ResponseProvider` / `MemoryStore`
  interface; the orchestrator depends only on these, never on the mock.
- A `Policy` layer that decides whether a cognitive *suggestion* may execute
  (V0.0 rule: quiet mode suppresses `RESPOND`).
- A **resumable** processing pipeline: each durable stage leaves a database
  record, so a retry re-uses an already-completed analysis, memory job and
  action instead of clearing and re-running them.
- Per-conversation serialization (`asyncio.Lock`) with cross-conversation
  concurrency; serialized SQLite access via a re-entrant connection lock; and
  atomic, conditional job claiming.
- Strict no-persistence: an `allow_persistence=False` interaction writes *no*
  utterance, analysis, action, candidate, job or conversation metadata.
- Persistent conversation state (quiet mode, last action) that survives restarts.
- A durable, deduplicated, recoverable background job queue backed by SQLite
  (job type `MEMORY_CONSOLIDATION`), with authorisation re-checked at run time.
- Event recovery (`recovery list` / `recovery retry`) for utterances left
  `PROCESSING` by a crash.
- Ordered, append-only schema migrations that refuse an unknown newer version.
- Structured `key=value` logging for end-to-end traceability.
- A full CLI (`init`, `person`, `say`, `inspect`, `history`, `quiet`, `jobs`,
  `recovery`, `demo`, `repl`) and an automated test suite.

## What V0.0 deliberately does **not** do

- Real DeepSeek API, ASR, TTS, microphone/speaker I/O.
- Speaker diarization or voiceprints (CAM++/other).
- Semantic vector search or automatic long-term fact extraction.
- GUI, web service, or cloud database.
- Continuous environment listening.

---

## Requirements

- Python **3.11+** (tested on 3.12)
- `uv` (recommended) or `pip`

## Install

```bash
cd ai-companion

# with uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,ui,web]"

# or with pip
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,ui,web]"
```

## Quick start

All commands run via `python -m companion` (or the `companion` console script
after install). The default database is `data/companion.db`; override with the
`COMPANION_DB` environment variable or the `--db` flag (accepted before or
after a subcommand).

```bash
# 1. Initialise the database
python -m companion init

# 2. Register a person (returns a new person_id)
python -m companion person add \
  --name "Owner" \
  --role owner \
  --allow-processing \
  --allow-persistence \
  --allow-memory-learning
# -> person_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 3. Say something to the agent (mock RESPOND)
python -m companion say \
  --person person_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --to agent \
  --text "你好，今天怎么样？"

# 4. Say something to another human (recorded, but IGNORE — no reply)
python -m companion say \
  --person person_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --to human \
  --text "我昨天完成了一个项目。"

# 5. Inspect the full processing trace of an event
python -m companion inspect evt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 6. Quiet mode (persists across restarts)
python -m companion quiet on
python -m companion quiet off

# 7. Other commands
python -m companion person list
python -m companion history --limit 50
python -m companion jobs list
python -m companion jobs retry job_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python -m companion recovery list                 # events left unfinished by a crash
python -m companion recovery retry evt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python -m companion demo      # scripted demo on an isolated temp database
python -m companion repl --person person_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`python -m companion --help` (or any subcommand `--help`) lists every option.

### Example `say` output

```
[EVENT] Saved: evt_0c7756e65f764f8eab4e183564d09788
[SPEAKER] person_2cb3661ab2184cf08889e6644449c337
[CONTEXT] Conversation loaded
[MEMORY] Retrieved 0 items
[COGNITION] Analysis completed by mock
[DECISION] RESPOND
[RESPONSE] 模拟回复：我收到你的话了。
[MEMORY_JOB] Queued
[STATUS] Completed
```

For a `--to human` utterance the same trace shows `[DECISION] IGNORE` and no
`[RESPONSE]` line — the event, analysis, and memory job are still recorded.

---

## Real DeepSeek mode (V0.1)

Mock remains the default: no command makes a network call or spends tokens
unless you explicitly pass `--provider deepseek`.

### 1. Install the SDK extra

```bash
uv pip install --python .venv/bin/python -e ".[deepseek]"
```

### 2. Configure the key (environment only)

```bash
export DEEPSEEK_API_KEY="sk-..."           # read from the environment only
# optional overrides
export DEEPSEEK_MODEL="deepseek-flash"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_TIMEOUT="60"
export DEEPSEEK_MAX_RETRIES="2"
```

The key is never accepted as a CLI flag, never written to SQLite, never logged,
and never included in error messages.

### 3. Grant external-AI consent

External AI is a **separate** grant from memory learning:

```bash
python -m companion person add --name "Owner" --role owner \
  --allow-processing --allow-persistence --allow-memory-learning \
  --allow-external-ai
```

A person without `--allow-external-ai` is still processed locally, but their
input is never sent to DeepSeek: the analysis becomes `WAIT` /
`permission_denied` and no request leaves the machine.

### 4. Verify without spending tokens

```bash
python -m companion deepseek check          # offline config check
python -m companion deepseek check --live   # one minimal real call
```

### 5. Talk to it

```bash
python -m companion say --provider deepseek \
  --person person_xxx --to agent --text "你好"

python -m companion repl --provider deepseek --person person_xxx
# /status shows backend, person, conversation, addressee and quiet mode
```

For every authorised input the real pipeline builds a privacy-filtered context
(recent turns with speaker labels + permitted memories), asks DeepSeek for a
strict-JSON cognition result (`RESPOND`/`WAIT`/`IGNORE` plus meaning, reason
code and memory candidates), applies the local policy, and only then asks
DeepSeek for a reply when the final action is `RESPOND`. The analysis, memory
candidates and decision are persisted even when no reply is sent. Candidates
remain inert proposals (`CONSOLIDATED`/`SKIPPED`) and are never auto-promoted
to `ACTIVE` memory.

Safety invariants:

- the model never controls `event_id`, `speaker_person_id`,
  `subject_person_id` or memory ids — those are filled by the program;
- fabricated `referenced_memory_ids` are dropped;
- an explicit `--to human` is never turned into a reply, even if the model says
  `RESPOND`;
- quiet mode and privacy grants are enforced locally, never delegated to the
  model;
- a missing key fails loudly — it never silently falls back to mock;
- `--provider deepseek` cannot pair a real cognitive provider with a mock
  response provider (or vice versa).

---

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

The suite (250 tests) covers: multi-person independence and duplicate names;
a decision for every allowed utterance; `IGNORE` still recording everything;
idempotent re-submission and conflict rejection; database-restart persistence;
background-job recovery (including a crashed `RUNNING` job); provider-exception
retryability; consent denial; person `SUSPENDED`/`REVOKED` denial; revoking
learning consent after a job is queued; strict no-persistence (including no
conversation metadata); response-failure retry that preserves the analysis and
memory job; crash recovery at each pipeline stage; real async concurrency
(gather, same/different conversations, competing workers and connections,
transaction isolation); schema-version migration and refusal of unknown
versions; model validation (UTC datetimes, non-empty ids, illegal statuses);
REPL identity reuse; and a 100-event integration run.

The V0.1 DeepSeek layers are covered with fake clients only (no key, no
network): request mapping, fail-fast auth/invalid-request handling, retry with
backoff for 429/5xx/timeout/empty responses, the foreground privacy gateway
(per-speaker and per-memory-subject authorisation), the context builder
(multi-turn interleaving, truncation, filtering), keyword/CJK memory retrieval
and referenced-id validation, strict-JSON parsing with one repair retry, the
explicit-human-addressee guard, candidate deduplication/never-auto-`ACTIVE`,
runtime provider-mixing rejection, missing-key failure, and response-failure
recovery that reuses a persisted analysis.

---

## Architecture at a glance

```
CLI / test
   │  UtteranceEvent (pydantic-validated)
   ▼
Orchestrator.process_utterance  (per-conversation asyncio.Lock)
   1 validate person + status + consent   2 load conversation (read-only if no persistence)
   3 deduplicate / conflict / resume      4 persist input
   5 retrieve permitted memories          6 CognitiveProvider.analyze (or reuse saved analysis)
   7 Policy.decide (quiet mode)           8 persist analysis + candidates + job (one txn)
   9 ResponseProvider.generate (only if RESPOND)   10 persist AgentAction (or reuse it)
   11 record last action + mark COMPLETED
```

Each durable stage leaves a database record, so a failed/partial event is
**resumed**, not cleared and re-run.

- `domain/` — pydantic models, enums, provider protocols.
- `storage/` — SQLite schema (WAL, FK on, parameterised SQL), connection, repositories.
- `people/` — person registry + consent.
- `core/` — orchestrator, conversation manager, policy.
- `cognition/` — foreground privacy gateway + external-AI context builder.
- `providers/` — mock providers, DeepSeek client, cognitive and response providers.
- `prompts/` — versioned cognitive/response prompt builders.
- `memory/` — `MemoryStore` (hand-authored facts, keyword/CJK retrieval) and candidate dedupe.
- `jobs/` — durable `JobWorker` + `MEMORY_CONSOLIDATION` handler.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full flow and
[docs/DATA_MODEL.md](docs/DATA_MODEL.md) for the data model.

## Known limitations

- **Mock has no real understanding.** The default mock provider decides purely
  from `addressee_hint` and records `provider=mock` with explicit reason codes.
  Real understanding requires `--provider deepseek`; it is never simulated.
- **No automatic memory learning.** Memory *candidates* are stored, deduplicated
  and validated by the consolidation job, but nothing is promoted to long-term
  facts (`memory_items` is only hand-populated or human-confirmed).
- **No speech.** CLI text input stands in for a future ASR-generated
  `UtteranceEvent`; `RESPOND` output is text, not TTS.
- **No speaker recognition.** The caller supplies `speaker_person_id`
  explicitly; there is no voiceprint or diarization.
- **Single-process.** One ordered consumer per conversation, with a
  re-entrant SQLite connection lock and atomic job claiming; no distributed
  queue, no multi-process coordination, and no concurrency pool (intentional
  for V0.0). Multiple independent processes sharing one SQLite file are not
  coordinated beyond SQLite's own locking.

## Next steps

V0.1 real-text DeepSeek cognition is implemented (see “Real DeepSeek mode”
above). Still deliberately out of scope: ASR/TTS and voiceprints, real DeepSeek
idle reflection, autonomous web research, a vector database, automatic
long-term fact merging, and a React UI.
