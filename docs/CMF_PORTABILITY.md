# CMF 1.0 — Portability

A CMF file is one SQLite database that can be copied, backed up, imported and
cloned without losing identity, memory, research or learning state.

## 1. Single-file guarantee

- One SQLite file per AI identity. Preferred extension `.memory`; `.db`,
  `.sqlite`, `.sqlite3` remain accepted so a user's database is never forcibly
  renamed.
- No sidecar databases, no second persistent store. WAL and `-shm` sidecars
  exist only while a file is open and are checkpointed away by every export.

## 2. Stable identity across backup and restore

- `cmf_manifest.agent_id` is stable across close/reopen, export/import and
  backup/restore. The `application_id` header (`0x434D4601`) marks the file
  as CMF and is preserved by the backup API.

## 3. Export (consistent snapshot)

`CmfFileService.export(source, destination)` uses the SQLite **backup API** for
a WAL-safe, transactionally consistent copy, then validates the output and
atomically replaces the destination (temp file + `os.replace`). It never
overwrites an existing destination unless `overwrite=True`.

## 4. Clone (new identity with lineage)

`CmfFileService.clone(source, destination)` copies the file, then re-identifies
it: a new `agent_id`, `lineage_parent_agent_id` set to the source agent id, the
shared `lineage_id` preserved, and `clone_of` recorded. The clone is an
independent identity with a traceable ancestor; its manifest and
`agent_identity` agree on the new id.

## 5. Migration of legacy databases

`CmfFileService.migrate(path)` opens a pre-CMF Companion `.db` in place and
applies the append-only migrations. Migration 9 seeds the manifest, identity,
persona and relationship presets **only when absent**, so an upgrade from
schema version 8 loses no existing data and gains one default identity. The
`application_id` is stamped in place.

## 6. Failure safety

- A file declaring a newer major format version, or a newer schema version, is
  refused and left unmodified.
- A corrupt or foreign file is refused and never becomes a fresh empty
  companion.
- A failed connect closes its connection (no leaked worker thread).
- Every file-writing operation validates its output before the atomic
  replacement is committed.

## 7. Inspection and validation

- `inspect(path)` returns the agent id, display name, format/schema versions,
  lineage fields and table counts.
- `validate(path)` runs application-id, table-presence, `integrity_check`,
  `foreign_key_check`, schema-version gate, manifest singleton/format and
  persona-range checks.
