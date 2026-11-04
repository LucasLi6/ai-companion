#!/usr/bin/env python
"""Seed the reviewed Ava style-example batch into one companion database.

Examples are style references, never real conversation history. The import is
idempotent, does not touch memory, and never runs automatically at startup.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from companion.agent.style_examples import StyleExampleRepository
from companion.agent.style_seed import DEFAULT_BATCH, seed_style_examples
from companion.runtime import build_runtime
from companion.storage.database import Database

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "assets" / "ava_style_examples_v1.jsonl"


async def _run(args) -> int:
    db = await Database(Path(args.db)).connect()
    try:
        await db.initialize()
        runtime = build_runtime(db, with_worker=False)
        manifest = await runtime.agent_service.repo.get_manifest() or {}
        agent_id = str(manifest.get("agent_id") or "")
        if not agent_id:
            print("no agent identity in this database; create one first", file=sys.stderr)
            return 1

        if args.disable_batch:
            repo = StyleExampleRepository(db)
            disabled = await repo.disable_batch(
                agent_id=agent_id, created_by=args.batch
            )
            print(f"batch={args.batch} disabled={disabled}")
            return 0

        result = await seed_style_examples(
            db,
            agent_id=agent_id,
            path=args.file,
            dry_run=args.dry_run,
            batch=args.batch,
        )
        print(
            f"batch={result['batch']} total={result['total']} "
            f"created={result['created']} skipped={result['skipped']} "
            f"dry_run={result['dry_run']}"
        )
        return 0
    finally:
        await db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/companion.db")
    parser.add_argument("--file", default=str(DEFAULT_FILE))
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--disable-batch",
        action="store_true",
        help="disable this batch instead of importing it",
    )
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
