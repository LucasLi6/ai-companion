# Idle Cognitive & Research (Phase D)

The idle pipeline lets the companion *generate* new candidate knowledge during
quiet periods: reflect on what it knows, ask useful questions, and research
them. It is **off by default** and every output is mock-driven unless a real
provider is explicitly enabled.

## What runs

One `IdleScheduler.run_cycle()` runs three engines in order:

1. **Reflection** — searches `ACTIVE` memories, asks the provider to spot
   patterns, and `propose`s each as a `CANDIDATE` `REFLECTION` memory.
2. **Curiosity** — searches `ACTIVE` memories, asks the provider for follow-up
   questions, and queues them as research tasks (`source_kind="curiosity"`).
3. **Research** — takes up to `limit` `PENDING` tasks, runs each through
   search → synthesize → consolidate, and marks it `COMPLETED` (or `FAILED`
   with an error).

## Gating

The whole pipeline is behind the `idle.enabled` app setting (default `false`).

- `idle.enabled=false` → `run_cycle()` returns `{"skipped": True, ...}` and does
  no work.
- **Research** has its own gate, `research.enabled` (also default `false`):
  `research run` and `run_all` do nothing until it is enabled.
- The **web search** has its own gate, `web_search.enabled` (also default
  `false`). When disabled, the factory returns the offline mock provider, and a
  real provider refuses to make a network call without authorisation.

```bash
python -m companion --db ./data.db idle run          # no-op unless enabled
python -m companion --db ./data.db research queue --question "…" --source-kind manual
python -m companion --db ./data.db research list
python -m companion --db ./data.db research run --limit 3
```

## Providers

- **DeepSeek** (`src/companion/providers/deepseek.py`) — a mock provider that
  returns `[mock]`-labelled `reflect` / `ask_questions` / `synthesize` output
  with **no network**. It is deliberately not wired to a real API (see
  constraints: DeepSeek stays mock for now).
- **Web search** (`src/companion/providers/websearch.py`) — two
  implementations behind `make_web_search(settings)`:
  - `MockWebSearchProvider` — deterministic, offline, `example.com` results.
  - `HttpWebSearchProvider(enabled)` — a real DuckDuckGo HTML scrape; raises
    `RuntimeError` unless explicitly enabled. Real search is **off by default**.

## Trust: nothing the AI writes is a fact

The idle engines read as `SYSTEM_ADMIN` but **act as `IDLE_AGENT`** (non-admin),
so they can propose and queue but never self-confirm:

- Reflections enter as `CANDIDATE` with `human_confirmed=False`,
  `model_inferred=True`.
- Research consolidations enter as `CANDIDATE` `WORLD_KNOWLEDGE` with
  `source_supported=False`, `model_inferred=True`, `human_confirmed=False`,
  and their web sources attached with `UNVERIFIED` evidence and the search
  provider's name. Mock research never becomes trusted knowledge.
- Both land in the store only for a human to `confirm`, `retract` or `delete`
  through the CLI or UI.

## Resumability

Research tasks are durable rows (`research_tasks`) with explicit
`PENDING`/`RUNNING`/`COMPLETED`/`FAILED` states and timestamps, so an
interrupted run can be resumed with `research run` rather than restarted from
scratch.
