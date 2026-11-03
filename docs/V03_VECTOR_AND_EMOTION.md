# V0.3 — Hybrid Vector Memory & Persistent Emotion

V0.3 adds two strictly additive layers on top of schema 9:

- **Hybrid vector retrieval** (schema 10) — semantically related memories are
  retrieved even when the wording differs, fused with the existing keyword
  ranking.
- **Persistent emotion framework** (schema 11) — the companion's own simulated
  affect, its decay, and emotion-aware reply tone.

Both live in the *same* single-file SQLite `.memory` database. There are no
sidecar stores (no ChromaDB/Pinecone/Qdrant/FAISS), and CMF 1.0 portability is
unchanged. Migrations 1–9 are untouched; 10 and 11 are append-only.

---

## Part A — Hybrid vector retrieval

### Storage

`memory_embeddings` stores one validated float32 BLOB per `(memory_id,
model_id)`; `embedding_models` records the full configuration
(`provider, model_name, model_revision, dimensions, normalization,
query_prefix, document_prefix, distance_metric, max_input_tokens,
configuration_hash`). The model id is a deterministic hash of that
configuration (`emb_<hash[..24]>`), so a fresh process lands on the same model
row.

Vectors are validated before storage: wrong dimensions or any `NaN`/`Inf`
raise `VectorValidationError` and are recorded as `FAILED`, never written.

### Providers

`EmbeddingProvider` is a dependency-injected protocol:

```python
class EmbeddingProvider(Protocol):
    model_id: str
    dimensions: int
    spec: EmbeddingModelSpec
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
```

- **`FakeEmbeddingProvider`** — deterministic bag-of-token-hashing (CJK
  bigrams included). It is a plumbing fixture for tests and offline demos and
  is **not** a semantic model.
- **`LocalSentenceTransformerProvider`** — the real local model,
  `intfloat/multilingual-e5-small` (384 dims, `query:`/`passage:` prefixes,
  L2-normalised; Chinese + English + Portuguese). It is imported lazily and
  requires the optional extra:

  ```bash
  pip install -e '.[embedding]'
  ```

  Weights are downloaded once on first use and are **runtime dependencies** —
  they are never stored in the `.memory` file. No model is downloaded in CI.

### Lifecycle

`EmbeddingService.embed_memory` is idempotent and safe:

- **Idempotency key** `embed:{memory_id}:{model_id}:{memory_version}`, plus a
  content hash — re-running the same write is a no-op (`UNCHANGED`).
- **Stale-write protection** — a job carrying an older `memory_version` than
  the stored vector returns `STALE_SKIPPED` and never overwrites newer data.
- **Failure isolation** — a provider failure records `FAILED` on the vector row
  and **never** touches the memory row; the memory stays keyword-retrievable.
- **Crash safety** — writing a vector never mutates the memory itself.

Background jobs use `JobType.EMBEDDING` (handler in
`jobs/embedding_handler.py`), which re-reads the *current* memory row before
embedding, so a retried job always works from the latest content.

Writes are indexed automatically: `MemoryService` notifies an `on_change`
observer after a committed write, and the runtime enqueues a job keyed
`embed:{memory_id}:{model_id}:{memory_version}` for ACTIVE memories. Candidates
and retractions enqueue nothing; a broken observer is logged and can never
fail the memory write. `memory index backfill` remains the safe repair path.

### Hybrid retrieval (RRF)

`HybridRetriever` fuses the keyword ranking with vector similarity using
Reciprocal Rank Fusion:

```
score(m) = Σ 1 / (k + rank_i(m)),  k = 60
```

Contract:

- **Authorised scope first.** The keyword candidate list comes from
  `MemoryService` (already visible to the requester). Vector search returns
  *ids only*; every id is re-checked through `load_memory(id, access)` before
  it may enter the ranking. A vector can never reveal a memory the keyword
  layer would not show.
- **Identity leads.** Deterministic subject+topic identity matches sort ahead
  of any fused result.
- **Diagnostics without leaking content.** Fused rows carry `keyword_rank`,
  `vector_rank`, `fusion_score`, `matched_memory_id`, `index_model`,
  `index_status` — only on rows the requester may already see.

`SqliteMemoryStore.retrieve(query, person_id)` remains the public entrypoint;
when no embedding provider is wired, behaviour is exactly the legacy
keyword/identity path.

### Optional `sqlite-vec`

If `sqlite-vec` is installed, `VectorIndex.using_sqlite_vec` is true; otherwise
a pure-Python bounded fallback ranks READY vectors. The extension is probed at
runtime and never required: startup succeeds without it, FTS5 keeps working,
and CMF validation never reports corruption because the optional extension is
absent.

### CLI

```bash
python -m companion memory index status
python -m companion memory index backfill --fake-provider
python -m companion memory index rebuild --fake-provider
python -m companion memory search --query "机器学习" --mode hybrid --fake-provider
```

Without `--fake-provider`, the commands use the real local model and print a
clear error if the `[embedding]` extra is missing. They never silently fall
back to the fake provider.

---

## Part B — Persistent emotion framework

Emotion is **computational affect and expression** — a deterministic simulation
of the companion's own presentation state. It never asserts anything about a
human's real feelings.

### Four layers

| Layer | Table | Meaning |
|-------|-------|---------|
| Stable personality | `emotion_profiles` | Per-agent baselines, half-life, responsiveness. Configured, never inferred from one turn. |
| Persistent appraisal rules | `emotion_appraisal_rules` | Versioned rules mapping a situation to an emotion. Learned rules are proposals until approved. |
| Temporary state | `emotion_state_snapshots`, `emotion_events` | The current decaying affect (per person) and its durable causes. |
| Emotion-aware reply | `voice_profiles` + `SpeechStylePlan` | Delivery hints that shape *how* a reply sounds. |

Two representations are kept:

- **Continuous** — valence ∈ [-1, 1], arousal ∈ [0, 1], intensity ∈ [0, 1].
- **Categorical** — `NEUTRAL, JOY, EXCITEMENT, RELIEF, CURIOSITY, CONCERN,
  SADNESS, FRUSTRATION, SERIOUS`.

### Decay

```
I(t) = I0 * 2 ** (-elapsed_seconds / half_life_seconds)
```

Valence and arousal drift toward the profile baseline with the same factor.
Decay is computed **lazily on read** — there is no background timer writing to
the database. The state is checkpointed in bounded writes (one snapshot row per
agent+person). Decay **never** deletes an event or a rule: the emotion fades,
the cause and reason persist.

### Appraisal

`AppraisalEngine.appraise` is local and deterministic. It considers the current
utterance, the already-authorised retrieved memories, the agent persona/
relationship, the ACTIVE rules and the decayed state, and returns a validated
`AppraisalResult` (emotion, continuous values, and the rule/reason that fired).
The model may only *suggest* a new rule through reflection; it is stored
`DISABLED` with its evidence and never applied to the active profile without
human approval. No third DeepSeek call is required.

Default rules cover ordinary situations (milestone, setback, gratitude, urgency,
a question) and are seeded per agent on first use.

### Emotion-aware reply

The `SpeechPlanner` turns the decayed state into a `SpeechStylePlan`
(`text`, `emotion`, `intensity`, `speaking_rate`, `delivery_instruction`,
`voice_profile_id`) — provider-neutral, with a future Qwen-TTS adapter contract.
It only affects **how** a reply is delivered (rhythm, warmth, humour,
formality, word choice, question style); it **never** changes whether a reply is
sent, and never touches permission, privacy or silence.

In the orchestrator, emotion is computed after memory retrieval and gated on the
consent `allow_persistence` flag for event/snapshot writes. Any emotion error is
logged and swallowed: it can never fail a turn.

### Privacy

The framework reuses the existing consent gates. It keeps three things distinct:
the user's inferred state (never asserted as fact), the AI's own simulated
state, and the reply's tone. Emotion snapshots are scoped per `agent_id` and
per `person_id`, so two people's interactions never mix private emotional
context, and a clone gets its own `agent_id`.

### CLI

```bash
python -m companion emotion show
python -m companion emotion inspect --person person_xxx
python -m companion emotion rules list
python -m companion emotion config show
python -m companion emotion config set default_half_life_seconds 1800
python -m companion emotion simulate --person person_xxx --elapsed-seconds 3600
```

`emotion config set` and rule status changes are audited. `emotion simulate`
uses an injected clock derived from the stored snapshot time plus the requested
elapsed seconds, so decay can be inspected deterministically without waiting.
