# Memory Manager (CLI + UI)

The unified memory store is human-operable through two surfaces: the `memory`
CLI subcommand and a web UI. Both go through the same `MemoryService` gateway,
so the policy rules in [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) apply identically.

## CLI

```bash
python -m companion --db ./data.db memory add \
    --kind PERSONAL_FACT --content "喜欢机器人" --subject "person-id"

python -m companion --db ./data.db memory search --query "机器人" --kind PERSONAL_FACT
python -m companion --db ./data.db memory show <memory-id>

python -m companion --db ./data.db memory confirm <memory-id>
python -m companion --db ./data.db memory retract <memory-id>
python -m companion --db ./data.db memory delete <memory-id>

python -m companion --db ./data.db memory link <memory-id-a> <memory-id-b> --relation conflicts_with
```

Subcommands:

| Command | Purpose |
|---------|---------|
| `memory add` | Create a record with explicit `--kind`, `--content`, optional `--subject`/`--visibility`/`--human-confirmed`. |
| `memory search` | FTS/`LIKE` search with filters (`--kind`, `--status`, `--subject`, `--query`). |
| `memory show` | Full view of one record: content, provenance flags, subjects, sources, links, revision history. |
| `memory confirm` | Promote a `CANDIDATE`/`NEEDS_REVIEW`/`CONFLICTED` item to human-confirmed `ACTIVE`. |
| `memory retract` | Mark `RETRACTED` (reversible, kept for audit). |
| `memory delete` | Permanently remove the record and its private content (irreversible). |
| `memory link` | Relate two records (`conflicts_with`, `supports`, `related_to`). |

## Web UI

```bash
python -m companion ui --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080`. The UI is served by NiceGUI and defaults to
loopback only; it is not exposed to the network unless you change `--host`.

Pages:

- **Dashboard** (`/`) — overview counts by kind and status, recent activity.
- **Memory Explorer** (`/memory`) — filterable table of records with
  kind/status/visibility badges and provenance flags.
- **Memory Detail** (`/memory/{id}`) — full record with its sources, links and
  revision history, plus confirm/retract/delete actions.
- **Person Manager** (`/people`) — list people, register new ones, and set
  consent / status.
- **Research Manager** (`/research`) — view research tasks, their status, and
  their consolidated output.
- **Activity / Audit Log** (`/audit`) — the append-only audit trail.
- **Settings** (`/settings`) — read/edit `app_settings` key/value pairs, e.g.
  `idle.enabled`, `web_search.enabled`.

Each request opens its own runtime connection against the database, so the UI
has no long-lived shared state and no way for a page or a prompt to reach the
database other than through the repositories.
