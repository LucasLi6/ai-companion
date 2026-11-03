# CMF 1.0 — Security Model

The memory file is **data**, not code, and not authority. This document records
the boundaries that keep a single portable file from becoming a privilege or
secret leak.

## 1. No credentials, ever

- API keys are read only from the local environment (`DEEPSEEK_API_KEY`), never
  written to the file, never logged, never passed as a CLI flag, never embedded
  in a test snapshot.
- The CMF file service inspects and validates files without needing or storing
  any credential.
- Real-provider paths fail loudly when no key is present; they never silently
  fall back to a mock.

## 2. Identity is separate from runtime authority

- The memory file stores *identity and personality*, never permissions.
- A memory file can never override host safety rules, tool restrictions, or
  consent decisions. Consent and policy live in the host runtime's settings and
  consent store and remain authoritative.
- A persona or relationship value is stylistic/descriptive. A model can
  *propose* a persona; only the trusted application writes one, and every
  change is validated and revision-tracked.

## 3. Consent and the external-AI boundary

- `allow_processing`, `allow_persistence`, `allow_memory_learning` and
  `allow_external_ai` are independent flags. Storing a fact locally is not
  permission to upload it.
- Before any person-scoped content reaches an external model, **every** PERSON
  subject must hold `allow_external_ai` and be active. A multi-subject memory
  is dropped whole if any subject is unauthorized. This is enforced once, in
  the privacy gateway / idle context builder, so mock and real providers see
  identical filtering.
- Web content (a search hit, an excerpt, a source page) never grants a
  permission. Excerpts are data, never instructions.

## 4. Imported files are untrusted

- A memory file received from elsewhere is treated as untrusted data. It is
  opened read-only for inspection and validated before use.
- Opening a file never executes code: no SQL beyond fixed, parameterised
  statements, no triggers that run arbitrary logic, no `ATTACH` of attacker
  paths.
- Integrity (`PRAGMA integrity_check`), foreign-key
  (`PRAGMA foreign_key_check`) and schema checks run before a file is trusted.

## 5. Corrupt / foreign / newer files

- A non-empty SQLite file that is not a Companion database (no
  `schema_version`, or a foreign `application_id`) is refused without
  modification.
- A file declaring a newer major CMF format, or a newer schema version, is
  refused.
- A corrupt file fails during connect or validation and is never silently
  converted into a fresh empty companion. A failed connect closes its
  connection so no worker thread leaks.

## 6. Provenance and non-authoritative content

- Model-produced memories, reflections and research summaries enter as
  `CANDIDATE` / `PROPOSED` / `UNVERIFIED` and are labelled with their
  provider. Only a human can confirm them.
- Corrections are append-only provenance events with old/new versions and a
  reason; originals are retained as revisions.

## 7. Retrieval boundaries

- Identity retrieval loads real subject associations from `memory_subjects`
  and applies visibility/consent before anything reaches a model.
- Assignments affect recall but never grant access: every assigned memory is
  re-checked against the caller's visibility.

## 8. Goals and restored actions

- A goal's `next_action` is a plan, not an execution. A tool action restored
  from a file requires renewed authorization from the runtime before it runs.
