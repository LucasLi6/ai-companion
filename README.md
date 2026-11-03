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
  `allow_memory_learning` — **never defaulted on**.
- A single validated `UtteranceEvent` per confirmed input, with deduplication
  and conflict detection.
- A pluggable `CognitiveProvider` / `ResponseProvider` / `MemoryStore`
  interface; the orchestrator depends only on these, never on the mock.
- A `Policy` layer that decides whether a cognitive *suggestion* may execute
  (V0.0 rule: quiet mode suppresses `RESPOND`).
- Persistent conversation state (quiet mode, last action) that survives restarts.
- A durable, deduplicated, recoverable background job queue backed by SQLite
  (job type `MEMORY_CONSOLIDATION`).
- Structured `key=value` logging for end-to-end traceability.
- A full CLI (`init`, `person`, `say`, `inspect`, `history`, `quiet`, `jobs`,
  `demo`, `repl`) and an automated test suite.

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
uv pip install --python .venv/bin/python -e ".[dev]"

# or with pip
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
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
python -m companion demo      # scripted demo on an isolated temp database
python -m companion repl      # interactive text session
```

`python -m companion --help` (or any subcommand `--help`) lists every option.

### Example `say` output

```
[EVENT] Saved: evt_0c7756e65f764f8eab4e183564d09788
[SPEAKER] person_2cb3661ab2184cf08889e6644449c337
[CONTEXT] Conversation loaded
[MEMORY] Retrieved 0 items
[COGNITION] Mock analysis completed
[DECISION] RESPOND
[RESPONSE] 模拟回复：我收到你的话了。
[MEMORY_JOB] Queued
[STATUS] Completed
```

For a `--to human` utterance the same trace shows `[DECISION] IGNORE` and no
`[RESPONSE]` line — the event, analysis, and memory job are still recorded.

---

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

The suite (32 tests) covers: multi-person independence and duplicate names;
a decision for every allowed utterance; `IGNORE` still recording everything;
idempotent re-submission and conflict rejection; database-restart persistence;
background-job recovery (including a crashed `RUNNING` job); provider-exception
retryability; consent denial; quiet-mode behaviour and persistence; and a
100-event integration run.

---

## Architecture at a glance

```
CLI / test
   │  UtteranceEvent (pydantic-validated)
   ▼
Orchestrator.process_utterance
   1 validate person + consent   2 ensure conversation   3 deduplicate
   4 persist input   5 retrieve permitted memories
   6 CognitiveProvider.analyze (mock)   7 validate analysis
   8 Policy.decide (quiet mode)   9 persist analysis + memory candidates
   10 enqueue memory job   11 update conversation state
   12 ResponseProvider.generate (only if RESPOND)   13 persist AgentAction
```

- `domain/` — pydantic models, enums, provider protocols.
- `storage/` — SQLite schema (WAL, FK on, parameterised SQL), connection, repositories.
- `people/` — person registry + consent.
- `core/` — orchestrator, conversation manager, policy.
- `providers/` — mock cognitive + response providers.
- `memory/` — `MemoryStore` (hand-authored facts via substring match, *not* semantic).
- `jobs/` — durable `JobWorker` + `MEMORY_CONSOLIDATION` handler.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full flow and
[docs/DATA_MODEL.md](docs/DATA_MODEL.md) for the data model.

## Known limitations

- **No real understanding.** The mock provider decides purely from
  `addressee_hint` and records `provider=mock` with explicit reason codes. It
  never fabricates confidence scores.
- **No automatic memory learning.** Memory *candidates* are stored and
  validated by the consolidation job, but nothing is promoted to long-term
  facts (`memory_facts` is only hand-populated in V0.0).
- **No speech.** CLI text input stands in for a future ASR-generated
  `UtteranceEvent`; `RESPOND` output is text, not TTS.
- **No speaker recognition.** The caller supplies `speaker_person_id`
  explicitly; there is no voiceprint or diarization.
- **Single-worker, single-process.** One ordered consumer per conversation;
  no distributed queue or concurrency pool (intentional for V0.0).

## Next steps

See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) for the concrete V0.1 plan to swap
in a real DeepSeek cognitive provider.
