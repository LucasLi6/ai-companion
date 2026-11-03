# Next steps — V0.1 (real DeepSeek cognition)

V0.1's only goal is to replace the mock cognitive engine with a real DeepSeek
provider **without touching the orchestrator, storage, or policy layers**.

## What to change

### 1. Add a DeepSeek client (new file)

`src/companion/providers/deepseek.py`:

```python
class DeepSeekCognitiveProvider:
    def __init__(self, api_key: str, model: str, ...): ...

    async def analyze(self, event, context, memories) -> CognitiveResult:
        # Build a prompt from event.text + recent context + permitted memories,
        # call the DeepSeek chat API, then parse the response into a
        # CognitiveResult. Emit an explicit reason_code (e.g. direct_agent_address,
        # question_to_agent, statement_to_human, unknown_addressee).
        ...
```

Requirements:

- Return a pydantic-validated `CognitiveResult` (`action` is a **suggestion**;
  the policy layer still decides execution).
- Include `provider="deepseek"` and a real `reason_code`; do **not** invent a
  fake confidence number unless the model actually produces a calibrated one.
- Map model output (e.g. tool-calls or a constrained JSON schema) onto
  `RESPOND`/`WAIT`/`IGNORE` and generate `memory_candidates` as proposals.
- Respect the `Memories`/`context` already passed in; the orchestrator already
  retrieves only permitted facts.

### 2. Wire it in `runtime.py`

```python
from companion.providers.deepseek import DeepSeekCognitiveProvider

cognitive = DeepSeekCognitiveProvider(api_key=..., model=...)
# keep MockResponseProvider or add a DeepSeekResponseProvider
```

Add a config flag (`--provider deepseek`, plus `DEEPSEEK_API_KEY` /
`DEEPSEEK_MODEL` env vars) in `config.py` and `cli.py` so `mock` stays the
default for tests/demo.

### 3. Add configuration

`config.py`: `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`, timeouts.
Keep the API key out of the repo and out of logs.

### 4. Response provider (optional)

Keep `MockResponseProvider`, or add `DeepSeekResponseProvider` behind the same
`ResponseProvider` protocol. The orchestrator already calls `generate` only on
`RESPOND`.

## What must NOT change

- `Orchestrator.process_utterance` (it depends only on the protocols).
- The SQLite schema and repositories (an analysis is an analysis regardless of
  provider; `provider` is already a column).
- `Policy.decide` (permission/quiet-mode gating stays independent of the model).

## Tests to add

- A fake HTTP transport that returns canned DeepSeek responses, so tests stay
  offline and deterministic.
- Assert `provider == "deepseek"` is recorded; assert prompt contains only
  permitted memory (no cross-person leakage).
- Assert a malformed/erroring DeepSeek response leaves the event `FAILED`
  (retryable) — reuse the existing `test_provider_exception_is_retryable`
  pattern with the DeepSeek provider.
- Assert quiet mode still downgrades a DeepSeek `RESPOND` to `WAIT` (policy is
  provider-independent).

## V0.2 and beyond

- **V0.2 memory learning**: replace the `MEMORY_CONSOLIDATION` handler body
  with real extraction, verification, conflict resolution, and writes to
  `memory_facts` (with the multi-subject join table described in
  `DATA_MODEL.md`).
- **Speech**: replace CLI text input with an ASR-produced `UtteranceEvent`; add
  TTS playback after `RESPOND`.
- **Speaker ID**: replace the manually chosen `speaker_person_id` with a
  verified recognition result (diarization + voiceprint).
- **Skills**: extend the policy/action layer to invoke registered skills under
  permission control, never executing arbitrary model-generated code.
