# V0.3 Implementation Report

**Branch:** `v0.1-deepseek-integration`
**Base HEAD:** `51d669c5ec14aeb5227e811a3be003a183fc130e` (CMF 1.0)
**Schema:** 11 (schema 9 → 10 → 11, both migrations append-only)
**Test result:** `372 passed` (334 pre-existing + 38 new: 21 vector, 17 emotion)

This report covers the two V0.3 parts — hybrid vector memory retrieval and the
persistent emotion framework — and states exactly what is and is not verified.

---

## Part A — Hybrid vector memory retrieval

### Schema (migration 10)

Additive `ALTER TABLE ... ADD COLUMN` only; migrations 1–9 are untouched.

- `embedding_models` gains `model_revision`, `normalization`,
  `query_prefix`, `document_prefix`, `distance_metric`, `max_input_tokens`,
  `configuration_hash`.
- `memory_embeddings` gains `memory_version`, `content_hash`,
  `embedding_status`, `updated_at`, `last_error`.
- Indexes: `idx_memory_embeddings_status`, `idx_memory_embeddings_model_status`,
  `idx_embedding_models_hash`.

Existing rows receive conservative defaults (`normalization='l2'`,
`distance_metric='cosine'`, `embedding_status='UNVERIFIED'`,
`memory_version=0`) so nothing is silently treated as a valid vector.

### Code

| Module | Responsibility |
|--------|----------------|
| `memory/embeddings.py` | `EmbeddingProvider` protocol, validated float32 pack/unpack, pure-Python cosine/dot/L2, `EmbeddingModelSpec`, deterministic `FakeEmbeddingProvider`, lazy `LocalSentenceTransformerProvider`. |
| `memory/embedding_repository.py` | Parameterised SQL for the model registry and vector rows. |
| `memory/vector_index.py` | Top-k search; optional `sqlite-vec` probe with pure-Python bounded fallback. |
| `memory/embedding_service.py` | Idempotent `embed_memory`, stale-write guard, failure isolation, `backfill`/`rebuild`/`status`, `embed:{id}:{model}:{version}` job key. |
| `memory/hybrid_retriever.py` | RRF fusion with authorised-scope rechecks and diagnostics. |
| `jobs/embedding_handler.py` | `EMBEDDING` job handler that re-reads the current memory row. |

`SqliteMemoryStore.retrieve(query, person_id)` remains the public entrypoint;
`MemoryService` remains the authorisation gateway. With no provider wired,
retrieval is byte-for-byte the legacy keyword/identity path.

### Automatic lifecycle

`MemoryService` accepts an optional `on_change` observer and calls it after a
committed create/update/confirm/retract/delete. `build_runtime` wires that
observer to enqueue an `EMBEDDING` job with the key
`embed:{memory_id}:{model_id}:{memory_version}` whenever an ACTIVE memory is
written; candidate/retracted writes enqueue nothing (retrieved candidates must
be ACTIVE, and a retract is filtered at query time). A failing observer is
logged and swallowed — indexing is a derived concern and can never break or
roll back a memory write. Missed work is still recoverable with
`memory index backfill`.

### Guarantees verified by tests

1. Float32 pack/unpack round-trip; non-multiple-of-4 BLOB rejected.
2. Dimension mismatch and `NaN`/`Inf` rejected.
3. L2 normalisation (including the zero-vector case).
4. Cosine/dot arithmetic without numpy.
5. Content-hash determinism and sensitivity to kind and content.
6. Fake provider determinism and correct dimensions.
7. Query vs document prefixes differ (E5 convention exercised).
8. Model spec hash / id stability.
9. `embed_memory` idempotency (`UNCHANGED`).
10. Stale-version write protection (`STALE_SKIPPED`).
11. Embedding failure does not destroy the memory; keyword retrieval still finds it.
12. Backfill embeds only `ACTIVE` memories.
13. RRF ordering: identity match leads; dual-list match outranks single-list.
14. Hybrid diagnostics present on fused rows and absent from identity rows.
15. Vector candidate rechecked through access control (a denied id never appears).
16. An ACTIVE write enqueues exactly one job with the versioned key, and the
    worker turns it into a READY vector with no manual backfill.
17. An update enqueues the next version; retract and non-ACTIVE (candidate)
    writes enqueue nothing.
18. (plus primitive and spec tests as above).

### CLI

```bash
python -m companion memory index status
python -m companion memory index backfill [--fake-provider]
python -m companion memory index rebuild [--fake-provider]
python -m companion memory search --query "..." --mode hybrid [--fake-provider]
```

`memory index status` also reports `sqlite_vec_available` and the active
backend. Without `--fake-provider` the real local model is used; a missing
`[embedding]` extra prints a clear error (no silent fake fallback).

---

## Part B — Persistent emotion framework

### Schema (migration 11)

Six new tables: `emotion_profiles`, `emotion_appraisal_rules`, `emotion_events`,
`emotion_state_snapshots`, `emotion_feedback`, `voice_profiles`. All `agent_id`
columns are plain `TEXT` (no FK to `cmf_manifest`), matching the existing
`persona_revisions` convention — this keeps `memory-file clone` correct, because
re-identification updates the manifest `agent_id` in place rather than deleting
the parent row.

### Code

`src/companion/emotion/`:

- `models.py` — `EmotionProfile`, `AppraisalRule`, `EmotionState`,
  `AppraisalResult`, `SpeechStylePlan`, `VoiceProfile`, 9 categorical labels.
- `dynamics.py` — exact exponential decay and baseline drift, categorical
  labelling, clamping.
- `appraisal.py` — deterministic rule matching (utterance/memory triggers,
  priority ordering) and the 5 seeded default rules.
- `speech.py` — `SpeechPlanner`, provider-neutral delivery instructions and
  bounded speaking-rate modulation.
- `repository.py` — parameterised SQL for all six tables.
- `service.py` — `ensure_defaults`, profile updates (audited, versioned), rule
  status/proposals, lazy-decayed `get_state`, idempotent `apply_result`,
  `plan_speech`, feedback.

Integration: `Orchestrator._apply_emotion` computes affect after memory
retrieval, attaches `emotion_state` + `speech_plan` to `CognitiveContext`, and
the response prompt renders an 【表达方式】 section. Emotion is gated on consent
persistence, and any emotion error is logged and swallowed so it can never fail
a turn. It never influences policy, permission, silence or addressee.

### Transcript privacy bug found and fixed

An early version loaded vector candidates through a hard-coded admin access
context, which would have bypassed visibility checks. `HybridRetriever.retrieve`
now takes the requester's `AccessContext` and passes it to `load_memory`, and a
test asserts a denied id can never enter the results.

### Guarantees verified by tests

- Defaults seeded once (profile + 5 rules) and idempotently.
- Milestone → `JOY`; priority ordering (urgent beats milestone); no-match holds
  the existing state.
- Exact half-life decay (`3600 s → ×0.5`, `7200 s → ×0.25`); neutral floor;
  baseline drift; label thresholds.
- `apply_result` persists event + snapshot, is idempotent by `source_event_id`,
  and writes nothing when `persist=False`.
- Profile update clamps to valid ranges and bumps the revision (audited).
- Proposed rules are created `DISABLED` with `source=reflection`.
- Rule status changes take effect and are audited.
- Speech plan rate is bounded and carries the voice profile id.
- Only an actual appraisal (a rule fired) writes an emotion event; a
  no-rule turn checkpoints the decayed state without event bloat.

### CLI

```bash
python -m companion emotion show
python -m companion emotion inspect [--person ...]
python -m companion emotion rules list
python -m companion emotion config show|set
python -m companion emotion simulate --person ... --elapsed-seconds N
```

---

## Manual acceptance scenarios (observed)

All run against a throwaway `.memory` file; the production memory file was not
touched.

- **A — semantic recall despite different wording.** With the fake provider,
  backfilled memory "我是一名机械工程专业的学生，正在做机器人项目" was
  retrieved for query "机器人 专业 学习" in hybrid mode (vector rank 1),
  while the keyword list was empty.
- **B — no optional extension needed.** `memory index status` reports
  `sqlite_vec_available: False` and `backend: pure-python fallback`; hybrid
  search still works, FTS5 is unaffected, and CMF validation reports
  `valid: True`.
- **C — embedding failure isolation.** A wrong-dimension provider records
  `FAILED` and leaves the memory row intact and keyword-retrievable (tested).
- **D — injected-clock decay.** `emotion simulate --elapsed-seconds 3600`
  shows intensity `0.50 → 0.25` and valence `0.40 → 0.20` for a JOY turn.
- **E — tone only.** `say` with "我拿到比赛第一名了！" logs
  `[EMOTION] JOY (intensity=0.50)` and still emits `[DECISION] RESPOND`; the
  CMF file validates afterwards.

---

## Honest disclosure

- **No real TTS.** The `SpeechStylePlan` and `voice_profiles` table are a
  provider-neutral contract only. No Qwen-TTS (or any other) call is made.
- **The fake embedding provider is not semantic.** It is a deterministic
  token-hashing fixture for tests and offline demos. The real semantic model
  (`intfloat/multilingual-e5-small`) was **not** exercised here: the optional
  `[embedding]` extra is not installed in this environment, so no real-model
  recall quality is claimed. The code path is wired but unverified against a
  live model.
- **`sqlite-vec` was not installed**, so only the pure-Python fallback was
  exercised; the acceleration path is implemented behind a runtime probe but
  was not run.
- **Emotion is simulated affect.** No claim is made about detecting or knowing
  a human's real emotions; the framework only computes the companion's own
  presentation state from deterministic rules.
- **No push, merge, publish, or modification of the production memory file was
  performed.** Changes are in the working tree on the target branch only.
- **Existing behaviour is unchanged by default.** No embedding provider and no
  emotion wiring change are forced on existing callers; the default runtime and
  the whole pre-existing 334-test suite pass unchanged.

## Files added / changed

Added: `memory/{embeddings,embedding_repository,embedding_service,vector_index,hybrid_retriever}.py`,
`jobs/embedding_handler.py`, `emotion/{__init__,models,repository,service,appraisal,dynamics,speech}.py`,
`tests/test_vector_retrieval.py`, `tests/test_emotion.py`,
`docs/V03_VECTOR_AND_EMOTION.md`, `docs/V03_REPORT.md`.

Changed: `storage/schema.py` (migration 10/11, `SCHEMA_VERSION=11`),
`cmf/constants.py` (required tables), `domain/enums.py` (`EMBEDDING` job type),
`memory/store.py` (hybrid hook), `runtime.py` (wiring), `core/orchestrator.py`
(emotion stage), `cognition/models.py` (emotion fields), `prompts/response.py`
(delivery section), `cli.py` (index/emotion commands, hybrid search), `README.md`.
