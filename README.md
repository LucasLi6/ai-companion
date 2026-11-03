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

# 8. Long-term memory inspection and assignment
python -m companion memory search "川菜"
python -m companion memory show mem_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python -m companion memory confirm mem_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python -m companion memory assign mem_xxx --person person_xxx
python -m companion memory unassign mem_xxx --person person_xxx
python -m companion memory assignments mem_xxx

# 9. Behaviour policy (typed settings, persisted in SQLite)
python -m companion policy show
python -m companion policy set policy.response_conservatism conservative
python -m companion policy set policy.interruption_threshold high
python -m companion policy set policy.curiosity_level low

# 10. Memory files, identity and relationships (CMF 1.0)
python -m companion memory-file create ava.memory --name Ava
python -m companion memory-file info ava.memory
python -m companion memory-file validate ava.memory
python -m companion memory-file export ava.memory ava-backup.memory
python -m companion memory-file clone ava.memory ava-clone.memory --name "Ava 2"
python -m companion memory-file migrate legacy.db

python -m companion agent show
python -m companion agent set-name Ava
python -m companion agent set humor 75
python -m companion agent set warmth 80
python -m companion agent revisions
python -m companion agent effective --person person_xxx

python -m companion relationship set person_xxx familiar
python -m companion relationship override person_xxx humor 30
python -m companion relationship history person_xxx

python -m companion goal add "设计底盘" --next "画图"
python -m companion goal list
python -m companion goal ready
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

## V0.2A — durable memory writes and reply policy

V0.2A closes the gap between "the model proposed a memory" and "the companion
remembers it". All of it still lives in the one SQLite file.

### What becomes a durable memory

A model candidate is a *proposal*, never a fact. The trusted consolidation
layer (`src/companion/memory/consolidation.py`) decides from the stored event,
the speaker's consent and the candidate's semantic hints:

| Case | Result |
|------|--------|
| Direct, clear self-report ("我叫李悦。") | `ACTIVE`, `self_reported=1`, never `human_confirmed` |
| Uncertain or inferred | durable `CANDIDATE` (not presented as fact) |
| Third-party report ("Bob 喜欢川菜") | durable `CANDIDATE` with `third_party_reported=1` and speaker provenance |
| Hypothetical / quoted / empty | no memory (`NO_MEMORY`) |
| Learning consent revoked, or subject not authorised | `DENIED`; nothing is written |

Every durable memory records `memory_provenance` (source event, candidate,
speaker, subject, assertion type, certainty, topic, explicit-remember flag).
`memory_candidate_writes` maps each candidate to its one durable outcome, so a
retry, restart or duplicate event cannot create a second memory.

### Truthful write outcomes

The write outcome is one of `SAVED`, `UPDATED`, `ALREADY_EXISTS`,
`PENDING_REVIEW`, `DENIED`, `FAILED`, `NO_MEMORY`. An explicit "请记住…" performs
a bounded synchronous write *before* the acknowledgement is generated, and the
response prompt receives only the committed outcomes — a failed or
pending-review write is never described as saved. `inspect` shows each
candidate's outcome and memory id.

A correction ("我刚才说错了，我的名字是李悦。") updates the current-state memory
in place; the old content remains in `memory_revisions`. A conflict with a
human-confirmed fact becomes `NEEDS_REVIEW` and never overwrites it.

### Silence, addressees and policy

The final action is decided locally, in this order: permission → explicit
current-turn silence → quiet mode → explicit human addressee → model
suggestion → conservative ambiguity. An explicit "这句话不要回复。" produces
`IGNORE` with no output, while analysis and memory learning still happen;
quoted or discussed silence ("我的朋友刚才说：不要回答我") does not suppress a
reply. The model's suggestion and the policy's final action are stored
separately (`model_suggested_action`, `policy_final_action`,
`override_reason`).

Policy settings are typed and persisted in `app_settings` (no random
non-response probabilities):

```bash
python -m companion policy show
python -m companion policy set policy.response_conservatism conservative   # conservative | balanced | responsive
python -m companion policy set policy.interruption_threshold high          # low | medium | high
python -m companion policy set policy.curiosity_level low                  # off | low | medium | high
```

In the real-provider REPL the default addressee is `unknown` (conservative):
use `/to agent` to address the companion explicitly.

### Assignments

`memory_assignments` associates a memory with a person for retrieval. It is
**not** an access grant: visibility and subject privacy are still enforced by
`MemoryService`, and a non-admin can only assign a visible memory to
themselves. Memories created from a self-report are auto-associated with their
speaker; third-party reports are not silently attached to the subject.

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

The suite (291 tests) covers: multi-person independence and duplicate names;
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

The V0.2A write-through/policy layers are also fake-client only: clear
self-reports becoming `ACTIVE` with provenance; uncertain, third-party and
hypothetical candidates; corrections updating in place with revisions;
duplicate events/content; revocation blocking writes; restart-and-new-
conversation identity retrieval; explicit-remember durability before the
reply; truthful response-prompt outcomes; raw/quoted silence; quiet mode;
human addressees; assignment privacy and restart persistence; lost-mapping
recovery; and database failures never recorded as success.

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
   7 persist analysis + candidates + job (one txn)
   8 explicit "remember" -> bounded durable write; otherwise read existing outcomes
   9 Policy.decide (permission, silence, quiet mode, addressee, necessity)
  10 ResponseProvider.generate (only if RESPOND)   11 persist AgentAction (or reuse it)
  12 record last action + mark COMPLETED
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
- `memory/` — `MemoryStore` (keyword/CJK + deterministic identity retrieval),
  `MemoryService` (the policy-enforcing gateway), candidate preparation and
  durable write-through/consolidation.
- `jobs/` — durable `JobWorker` + `MEMORY_CONSOLIDATION` handler.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full flow and
[docs/DATA_MODEL.md](docs/DATA_MODEL.md) for the data model.

## Known limitations

- **CMF 1.0 is the portable file format** (see
  [docs/CMF_SPEC.md](docs/CMF_SPEC.md), [docs/CMF_SCHEMA.md](docs/CMF_SCHEMA.md),
  [docs/CMF_SECURITY.md](docs/CMF_SECURITY.md),
  [docs/CMF_PORTABILITY.md](docs/CMF_PORTABILITY.md) and
  [docs/CMF_MIGRATION_PLAN.md](docs/CMF_MIGRATION_PLAN.md)). The media model
  (`media_assets`/`media_annotations`/`media_links`) and semantic embeddings
  (`embedding_models`/`memory_embeddings`/`embedding_index_metadata`/
  `retrieval_feedback`) are reserved in the schema but not built.
- **Real DeepSeek idle learning is not tested against the live API.** The
  opt-in `DeepSeekIdleProvider` is gated behind `idle.real_deepseek.enabled`
  and `--real-provider`; there is no silent fallback, but no real key/request
  was used in CI. Idle learning remains disabled by default.
- **Mock has no real understanding.** The default mock provider decides purely
  from `addressee_hint` and records `provider=mock` with explicit reason codes.
  Real understanding requires `--provider deepseek`; it is never simulated.
- **No autonomous semantic memory.** Memory *candidates* are consolidated by the
  trusted local layer (self-reports may become `ACTIVE`; uncertain and
  third-party candidates stay `CANDIDATE`), but there is no vector search, no
  semantic merging and no autonomous promotion of raw model output.
- **Conservative correction handling.** Only unambiguous same-attribute
  corrections update a current-state memory; anything else creates a review
  item rather than guessing.
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

V0.2A durable memory write-through, the structured reply policy and CMF 1.0
(the portable single-file memory format with the Agent Core, relationships,
durable research/learning state and goals) are implemented. Still
deliberately out of scope: ASR/TTS and voiceprints, a live real-DeepSeek idle
cycle, autonomous web research, the semantic vector index, automatic semantic
merging, the full media system, the Memory Board drag-and-drop UI, and a React
UI. A likely V0.2B focus is real-model acceptance testing, a manual
review/approval workflow for `NEEDS_REVIEW` items, and richer assignment
management in the existing manager UI.
