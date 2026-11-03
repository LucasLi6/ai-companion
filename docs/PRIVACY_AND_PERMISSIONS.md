# Privacy & Permissions

This project treats identity and privacy as hard boundaries, not conventions.
The rules below are enforced in code, in `MemoryService`, `PersonRegistry` and
the repositories — not by the LLM.

## Requester identity comes from the trusted app

Every memory read/write takes an `AccessContext`:

```python
AccessContext(subject_id, is_admin=False, role="...")
```

That context is constructed by the **application** (CLI, UI, idle engine) from
its own knowledge of who is acting. An LLM prompt never supplies its own
identity: there is no "please act as admin" path. The idle engines demonstrate
this — they read as `SYSTEM_ADMIN` but act as non-admin `IDLE_AGENT`.

## Visibility isolation

| Visibility | Who can read |
|------------|--------------|
| `PUBLIC` | everyone |
| `SHARED` | admin, or a `PERSON` subject of the item |
| `PRIVATE` | admin, or a `PERSON` subject of the item |

Non-admin requesters only see `PUBLIC` items and items where they are a
subject. A person cannot read another person's `PRIVATE`/`SHARED` memories.

## Source vs subject

`AccessContext` separates *who is asking* (source) from *who the memory is
about* (subject). Writing a personal/private item about a person requires that
person to be currently authorised for memory learning
(`allow_memory_learning`), regardless of who is asking. This is the consent
gate: a third party cannot silently author a private memory about someone else.

## Consent model

`PersonRegistry` manages per-person consent flags:

- `allow_processing` — may the system process this person's input.
- `allow_persistence` — may it persist that input.
- `allow_memory_learning` — may it store/learn memories about them.

`set_consent` records an audit entry. Consent can be withdrawn at any time, and
withdrawing learning consent stops new private memories about that person from
being written (existing records are governed by their visibility, not deleted
silently).

## Third-party reports

A statement *about* someone else (third-party report) is never auto-confirmed.
It is recorded with `third_party_reported=True` and `human_confirmed=False`
until the subject or a human confirms it.

## The audit log

`audit_log` is append-only. Memory create/update/confirm/retract/delete/link,
person register/set_consent/set_status, and other mutating actions each write a
row with `actor`, `action`, `entity_type`, `entity_id`, and a timestamp. The
"actor" is the trusted application actor, not the LLM.

## Structural safeguards

- **No direct SQL from prompts or UI** — the LLM and the web UI reach data only
  through repositories, which use parameterized SQL and enforce visibility.
- **No secrets in the repo** — admin passwords, API keys and access tokens are
  never written into source, config or migrations.
- **Real web search is opt-in** — `web_search.enabled` defaults off; a real
  provider refuses to run while disabled.
- **Tests never call real APIs** — idle/research tests use the mock DeepSeek and
  mock web search providers, so no test bills a paid API.
