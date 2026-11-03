# V0.4 Implementation Report — Continuous Scene Awareness & Natural Dialogue

**Branch:** `v0.1-deepseek-integration` (not committed, not pushed)
**Base HEAD:** `37b9c1e` (V0.3.2)
**Schema:** 13 → 14 (append-only migration)
**Test result:** `646 passed in 17.48s`
**Live model:** none available; all provider-boundary tests use the real
DeepSeek provider classes over a scripted transport and synthetic data.

This report covers what was actually implemented locally. Where something is
unverified, it says so instead of claiming a live result.

---

## 1. Files

### New production modules

| File | Responsibility |
|------|----------------|
| `src/companion/cognition/scene_models.py` | `SceneState`, `SceneEvidence`, `OpenLoop`, bounds (`MAX_OPEN_LOOPS=5`, `MAX_TOPIC_STACK=5`, `MAX_SOURCE_EVENTS=30`, `MAX_EVIDENCE_ITEMS=12`, `OPEN_LOOP_TTL_TURNS=8`, `ENVIRONMENT_TTL_TURNS=6`), prompt rendering, `default_environment()`. |
| `src/companion/cognition/scene_manager.py` | `load()` consent/source revalidation, `preview()`, `finalize()`, topic stack, loop creation/resolution, environment expiry. |
| `src/companion/cognition/scene_repository.py` | CAS snapshot writes (`SceneRevisionConflict`), idempotent per-event receipts, receipt-only recovery, scoped deletion. |
| `src/companion/agent/dialogue_style.py` | `DialogueStyleProfile`, `StyleThresholds`, `compile_dialogue_style()`. |
| `src/companion/agent/personality_compiler.py` | `PersonalityCompiler.compile()` / `pragmatic_guidance()`; stable principles and bounded context. |
| `src/companion/agent/style_examples.py` | Trusted, admin-authored style references; person/global scoping, disable, limit 3, validation. |

### Modified production modules

| File | Change |
|------|--------|
| `src/companion/domain/models.py` | `SceneProposal` / `SceneEnvironmentProposal` / `SceneOpenLoopProposal`; `validate_scene_proposal()`; `scene_proposal` on `CognitiveResult`. |
| `src/companion/cognition/models.py` | `scene_state`, `dialogue_style`, `personality`, `style_examples`, `pragmatic_guidance` on `CognitiveContext`. |
| `src/companion/core/orchestrator.py` | `_prepare_dialogue`, `_effective_context`, `_stored_trace`, `_commit_turn`, scene/personality flag helpers, non-persistence invalidation. |
| `src/companion/prompts/cognitive.py` | `cognitive-v5` → `cognitive-v6`; scene schema/rules/rendering. |
| `src/companion/prompts/response.py` | `response-v4` → `response-v5`; scene/personality/style sections and rules 12–17. |
| `src/companion/providers/deepseek_cognitive.py` | Parses and locally validates `scene_proposal`; evidence must come from the same registered speaker. |
| `src/companion/providers/deepseek_client.py` | Default `reasoning_effort="none"` for both analysis and response; env override preserved. |
| `src/companion/runtime.py` | Wires `SceneManager`, `PersonalityCompiler`, `StyleExampleRepository`; DeepSeek defaults on, mock defaults off. |
| `src/companion/storage/schema.py` | `SCHEMA_VERSION=14`; new columns/tables/indexes. |
| `src/companion/storage/repositories.py` | `UtteranceRepository.get_many()`; analysis scene proposal serialization/deserialization. |
| `src/companion/cli.py` | `agent style-example add/list/disable`. |

### New tests

| File | Tests |
|------|-------|
| `tests/test_scene_models.py` | 22 |
| `tests/test_scene_repository.py` | 8 |
| `tests/test_scene_manager.py` | 16 |
| `tests/test_personality_compiler.py` | 26 |
| `tests/test_natural_response.py` | 12 |
| `tests/test_dialogue_continuity.py` | 16 |
| `tests/test_v04_integration.py` | 10 |
| `tests/test_v04_scenarios.py` | 41 (40 synthetic three-turn scenarios + catalog check) |

Modified tests: `tests/test_cli.py` (+1 CLI style-example test),
`tests/test_deepseek_client.py` (default reasoning effort expectation).

**New-test total: 151 tests in V0.4 files, plus 1 CLI test = 152.**

---

## 2. Schema and migration

Migration 14 is additive; migrations 1–13 are untouched.

- `cognitive_analyses.scene_proposal_json TEXT` — persisted, locally validated
  proposal, so recovery can rebuild a scene without a second model call.
- `scene_snapshots(conversation_id, person_id, revision, state_json,
  source_event_ids_json, last_committed_event_id, updated_at)` — one row per
  `(conversation, person)`; `revision` is the CAS token.
- `scene_transition_receipts(source_event_id PK, conversation_id, person_id,
  result_revision, transition_json, applied_at)` — one receipt per source
  event; `ON DELETE CASCADE` from `utterances`.
- `dialogue_style_examples(example_id, agent_id, person_id NULL=global, ...,
  status)` — trusted admin data only; index on `(agent_id, person_id, status)`.

`test_migration_13_to_14_preserves_existing_data` builds a real version-13
database, upgrades it, and checks legacy utterances/analyses survive, new
tables are empty, `PRAGMA foreign_key_check` is clean, and reopening is
idempotent. Scene state is always in the single CMF `.memory` database; there
is no second store.

---

## 3. Actual request / processing flow

A `say` command enters `Orchestrator.process_utterance()`; one conversation is
serialized by an in-process lock.

1. Validate person, registration status and consent.
2. Load or create the conversation according to `allow_persistence`;
   non-persistence turns write no event/analysis/scene.
3. Load privacy-filtered history, labels and retrieved memories.
4. Build the provider `CognitiveContext`. For the real DeepSeek path it carries
   turns, action history, agent context, and then:
   - `SceneManager.load(conversation_id, person_id)` re-checks every source
     utterance against current consent/external-AI grants. A revoked source
     deletes the derived scene and returns `None`.
   - `PersonalityCompiler` compiles a style profile; current explicit request >
     memory-expressed preference > trait band.
   - At most 3 active style examples visible to this person are attached. They
     are marked as references, never history.
5. **DeepSeek call #1 — analysis** (`cognitive-v6`, JSON mode,
   `reasoning_effort="none"`). The model returns the legacy analysis fields
   plus an optional `scene_proposal`.
6. The provider locally validates the proposal: topic/environment/open loop
   require a verbatim span from the current text or this same speaker's
   authorized turns; malformed enums degrade; model IDs are ignored.
   `Policy` then makes the final action decision; the model never controls
   permission, silence or identity.
7. Emotion and the nonresponse trace are applied after the analysis exists.
8. If the final action is `RESPOND`, **DeepSeek call #2 — response**
   (`response-v5`) uses the same prepared context plus scene, compiled style,
   personality principles and style references. `WAIT`/`IGNORE` makes no
   response call. **There is no third model request.**
9. `_commit_turn()` writes action + scene snapshot/receipt + conversation
   `last_action` + utterance `COMPLETED` in one transaction. The scene write is
   CAS-protected:
   - a stale revision rolls the transaction back, reloads the newer scene and
     retries the *same already-generated* action, up to 3 attempts;
   - a source event is applied at most once via its receipt;
   - crash recovery reuses the persisted action/analysis: if the snapshot
     already advanced, it writes a receipt only; if the snapshot is missing it
     rebuilds the transition from `scene_proposal_json`; it never re-runs model
     semantics or generates a second reply;
   - after exhausted conflicts the turn is `FAILED`, no newer scene is
     overwritten, and the event stays retryable.

Feature flags: `dialogue.scene.enabled` and
`dialogue.personality_generation.enabled` (default true for the DeepSeek
runtime; mock runtime stays off unless explicitly enabled). Both off restores
V0.3.2 prompt sections and writes no scene rows.

---

## 4. Context-continuity evidence

`tests/test_dialogue_continuity.py` runs the real
`DeepSeekCognitiveProvider` and `DeepSeekResponseProvider` over a scripted
transport with synthetic authorized people. Sixteen tests cover:

- casual multi-turn continuity; topic shift and stack; return to an earlier
  topic; technical task mode; interruptions;
- pending-loop creation only when the local trace says `pending_followup`;
  no loop for a bare `WAIT`;
- a new topic after silence; corrections;
- two people in one conversation never sharing scene evidence;
- 30-turn revision/personality consistency;
- external-AI grant revocation deleting derived scene state;
- non-persistence turns writing no scene;
- a provider-dropped, non-verbatim scene claim not changing the topic;
- a style example reaching the response prompt but not becoming history;
- both flags off removing all V0.4 prompt sections.

`tests/test_v04_scenarios.py` adds 40 parameterized three-turn scenarios
(10 categories × 4): casual, playful, technical, interruption, switch, return,
correction, environment, open loops, affect. These assert the trusted-side
invariants (no invented environment, no stale topic resurrection, no loop
without pending follow-up, consistent persona), not model wording.

The integration file separately verifies atomic commit, stale-read retry with
no second model call, receipt backfill, snapshot rebuild, conflict rollback,
feature flags, non-persistence, withdrawal deletion, and untrusted proposals.

---

## 5. Old vs new dialogue quality

Honest scope: there was no live DeepSeek key, so this is a **structural and
synthetic comparison**, not a measured A/B of model output.

| Dimension | V0.3.2 (`cognitive-v5` / `response-v4`) | V0.4 (`cognitive-v6` / `response-v5`) | How verified |
|-----------|------------------------------------------|---------------------------------------|--------------|
| Continuity | Only recent turns; no persistent short-horizon topic | Persisted scene topic, bounded topic stack, open loops, explicit environment with TTL | Synthetic continuity/scenario tests |
| Topic return | Model must reconstruct from raw history | `RETURN` matches the stored stack; an unknown return degrades to a shift | Unit + scenario tests |
| Personality | Identity/familiarity parameters only | Compiled style profile (formality, directness, length, humor, warmth, questioning) with current-request precedence | Compiler unit tests |
| Style references | None | Up to 3 trusted, person-scoped examples, explicitly not fixed replies/history | Repo + prompt tests |
| Question spam | No explicit rule | Rules 14/13: do not end every turn with a question; do not use category templates | Prompt assertions |
| Environment hallucination | No explicit scene, so any claim was unconstrained | `UNKNOWN` unless an explicit, evidence-backed statement exists; expires after 6 turns | Validation/manager tests |
| Repeats self-intro | No rule | Rule 16: stable persona without per-turn self-introduction | Prompt assertions |
| Hard-coded persona | None | Still none: `Ava` appears only in tests/trusted data, never in system prompts | `test_response_prompt_has_no_hard_coded_persona_lines` |

What is **not** claimed: that the live model will always obey these rules, that
V0.4 is measurably more "natural" in human evaluation, or that any specific
reply quality improved. Those require a live model run, which was unavailable.

---

## 6. Known unstable scenarios

1. **Semantically wrong topic with a valid span.** Topic validation only
   requires that the evidence span exists in authorized text, not that the
   topic is entailed. A model can pair "我去了日本" with topic "工作" and the
   span check passes. This is deliberately permissive to allow abstract topic
   labels; it is the main residual trust gap.
2. **Pronoun-only continuity.** "那后来呢？" without a noun can be handled only
   if the model proposes `CONTINUE`; local code cannot infer the referent. A
   wrong proposal silently keeps or shifts the topic.
3. **Multi-person conversations.** Scene scope is `(conversation, person)`.
   A jointly discussed topic is not shared between people, so cross-person
   continuity is intentionally absent; rapid speaker switching may feel
   repetitive.
4. **Environment.** Only explicit statements are accepted and they expire after
   6 turns. There are no sensors, so "still in the same room" is never known;
   the model may sound less situated than a human.
5. **Open loops.** A loop is created only when the local nonresponse trace has
   `pending_followup`. Unanswered threads that do not produce that trace are not
   tracked.
6. **Style example matching.** Scenario matching is exact SQL string equality
   (with global/person preference), not semantic matching; no vector index was
   added, per the constraint.
7. **Cross-process contention.** CAS retries are bounded at 3. Under sustained
   contention the turn returns `FAILED` and must be retried; newer state is
   never overwritten, but the user may not get an immediate reply.
8. **Prompt-only personality.** Style examples and principles are guidance; a
   live model can still drift, over-copy an example, or become verbose.

---

## 7. Unfinished / not done

- No live DeepSeek acceptance run (no key in this environment); all model
  boundaries are tested with scripted transports and synthetic data.
- No automatic style-example learning or model-authored examples — writes are
  CLI/admin-only by design.
- No semantic scenario matching / vector index (explicitly out of scope).
- No shared multi-person scene, long-term scene promotion, or autonomous
  proactive speech.
- No UI; management is CLI-only.
- These changes are **not committed and not pushed**.
