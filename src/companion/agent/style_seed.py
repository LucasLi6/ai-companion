"""Idempotent import of the reviewed Ava style-example batch.

The batch is style reference material, not conversation history. Seeding is an
explicit admin action: it never runs at startup and never touches memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from companion.agent.style_examples import StyleExampleRepository
from companion.storage.database import Database

DEFAULT_BATCH = "seed:ava_style_examples_v1"


def load_style_examples(path: str | Path) -> list[dict]:
    """Read and validate one JSONL batch without writing anything."""
    records: list[dict] = []
    source = Path(path)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        for field in ("id", "category", "scenario", "user_text", "ideal_response"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{source}:{line_number}: {field} must be a non-empty string"
                )
        record["category"] = record["category"].strip().lower()
        records.append(record)
    return records


def _key(scenario: Any, user_text: Any, ideal_response: Any) -> str:
    return "\x1f".join(
        " ".join(str(value or "").split())
        for value in (scenario, user_text, ideal_response)
    )


async def seed_style_examples(
    db: Database,
    *,
    agent_id: str,
    path: str | Path,
    dry_run: bool = False,
    batch: str = DEFAULT_BATCH,
) -> dict:
    """Import one batch idempotently; return counts and the batch marker."""
    if not agent_id:
        raise ValueError("agent_id is required to seed style examples")
    records = load_style_examples(path)
    repo = StyleExampleRepository(db)
    existing = await repo.list(
        agent_id=agent_id, include_disabled=True, limit=10000
    )
    seen = {
        _key(row.get("scenario"), row.get("user_text"), row.get("ideal_response"))
        for row in existing
    }

    created = 0
    skipped = 0
    for record in records:
        category = record["category"]
        scenario = f"[{category}] {record['scenario'].strip()}"
        key = _key(scenario, record["user_text"], record["ideal_response"])
        if key in seen:
            skipped += 1
            continue
        if not dry_run:
            await repo.add(
                agent_id=agent_id,
                category=category,
                scenario=record["scenario"],
                user_text=record["user_text"],
                ideal_response=record["ideal_response"],
                created_by=batch,
            )
        seen.add(key)
        created += 1

    return {
        "total": len(records),
        "created": created,
        "skipped": skipped,
        "dry_run": dry_run,
        "batch": batch,
        "agent_id": agent_id,
    }
