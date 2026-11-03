# Real DeepSeek acceptance test (manual)

This checklist is **manual** and was **not executed** as part of the automated
work: there was no `DEEPSEEK_API_KEY` in the development environment, and the
suite is designed to run without one. All automated tests use a fake client and
make no network calls.

Do not paste a key into a shell history you do not control. Export it in the
current shell only.

## Preconditions

```bash
# 1. Install the SDK extra (the core runtime never imports it).
uv pip install --python .venv/bin/python -e ".[deepseek]"

# 2. Key, environment only. Never pass it as a CLI argument.
export DEEPSEEK_API_KEY="sk-..."

# 3. A disposable database.
export DB=/tmp/companion-acceptance.db
python -m companion --db "$DB" init
```

## 0. Configuration

```bash
python -m companion --db "$DB" deepseek check
# expect: api_key: configured, model: deepseek-flash,
#         "config check passed (offline...)"

# One minimal paid call, explicitly opted in:
python -m companion --db "$DB" deepseek check --live
# expect: "live check passed: ..." or a specific DeepSeekError type.
```

Also verify the failure path:

```bash
env -u DEEPSEEK_API_KEY python -m companion --db "$DB" deepseek check
# expect exit code 1 and "api_key: MISSING"
```

## 1. Register two people with different grants

```bash
python -m companion --db "$DB" person add --name Alice --role owner \
  --allow-processing --allow-persistence --allow-memory-learning --allow-external-ai
# ALICE=person_...

python -m companion --db "$DB" person add --name Bob --role friend \
  --allow-processing --allow-persistence --allow-memory-learning
# BOB=person_... (deliberately NO --allow-external-ai)
```

## 2. Multi-turn REPL (real cognition)

```bash
python -m companion --db "$DB" repl --person "$ALICE" --provider deepseek
```

Run at least 10 turns that exercise understanding, not just greetings:

1. `你好，我叫 Alice，最近在学做饭。`
2. `我特别喜欢川菜。`
3. `你觉得我今晚应该做什么菜？` (expect a reply that uses turn 2)
4. `/to human` then `我明天要去杭州出差。` (expect IGNORE; still recorded)
5. `/to unknown` then `嗯……` (expect WAIT)
6. `/quiet on` then `在吗？` (expect WAIT, no model response)
7. `/quiet off` then `/to agent` `刚才说到哪了？` (expect a reply referencing turn 4)
8. `/status` (provider/person/conversation/addressee/quiet mode)
9. `记住我喜欢少辣。` (a candidate may be proposed; it is not auto-`ACTIVE`)
10. `谢谢！`

Verify afterwards:

```bash
python -m companion --db "$DB" history --limit 20
python -m companion --db "$DB" inspect <EVENT_ID>
# expect: provider=deepseek, a concrete meaning, an explicit reason code,
#         and for IGNORE/WAIT events no [RESPONSE] text.
python -m companion --db "$DB" jobs list
```

## 3. Privacy: Bob never reaches DeepSeek

```bash
python -m companion --db "$DB" say --provider deepseek \
  --person "$BOB" --to agent --text "你好，我是 Bob"
# expect: [DECISION] WAIT, reason permission_denied, no [RESPONSE].
# No request is made for Bob: check the process log for
# "deepseek_analysis_denied ... reason=external_ai_not_granted".
```

Turn on `--debug` for that command if you want to see the audit lines; they
contain ids, model, latency and token counts only — never the text or key.

## 4. Memory influence and multi-subject privacy

```bash
# A fact about Alice (human-authored, therefore trusted).
python -m companion --db "$DB" memory add \
  --kind PERSONAL_FACT --content "Alice 对花生过敏" \
  --subject "PERSON:$ALICE" --status ACTIVE --human-confirmed

# A fact about Bob. Because Bob has no external-AI grant, this memory must be
# dropped from Alice's prompt (the gateway drops it whole).
python -m companion --db "$DB" memory add \
  --kind PERSONAL_FACT --content "Bob 住在上海" \
  --subject "PERSON:$BOB" --status ACTIVE --human-confirmed
```

Ask Alice: `我今晚想点外卖，有什么要注意的吗？`

- Expect the reply to be aware of the peanut allergy.
- The Bob memory must not appear in the prompt. With `--debug`, the context
  builder logs `external_context_dropped reason=unauthorized_subject`.

## 5. Candidate safety

After the REPL session:

```bash
python -m companion --db "$DB" memory search --status ACTIVE
# expect: no ACTIVE memory created from any model-proposed candidate.

python -m companion --db "$DB" jobs list      # consolidation jobs completed
python -m companion --db "$DB" inspect <EVENT_ID>   # candidates listed, inert
```

If the same event is submitted twice with the same `event_id`, the second run
must report `Duplicate (already completed)` and make no API call.

## 6. Error handling (fail fast)

```bash
DEEPSEEK_API_KEY=sk-invalid python -m companion --db "$DB" say \
  --provider deepseek --person "$ALICE" --to agent --text "测试"
# expect a fast failure (auth error), no retry storm, event marked FAILED,
# and no response text. Then restore the real key and run:
python -m companion --db "$DB" recovery list
python -m companion --db "$DB" recovery retry <EVENT_ID> --provider deepseek
```

An insufficient-balance account (HTTP 402) must behave the same way: fail fast,
never retry indefinitely, never fall back to mock.

## 7. Crash recovery

Start a long reply, then interrupt the process between analysis and response
(e.g. Ctrl-C). Re-run:

```bash
python -m companion --db "$DB" recovery list
python -m companion --db "$DB" recovery retry <EVENT_ID> --provider deepseek
```

Expect: the persisted analysis is reused (no second analysis request), and only
the missing response is generated.

## What to record

For each case, note: event id, final action, reason code, whether a response
was produced, and the `deepseek_request` audit line count. Any mismatch between
the expected decision and the model's suggestion should be attributed to the
local policy — the model never decides quiet mode, consent or an explicit
human addressee.
