#!/usr/bin/env python
"""Apply the reviewed Ava Persona Core to one companion database.

This is the explicit admin path for the stable character layer. It writes only
through the existing trusted service APIs:

- ``AgentService.set_name``          -> the display name;
- ``AgentService.set_persona``       -> the nine numeric expression traits;
- ``AgentService.set_persona_core``  -> the structured Persona Core fields and
  the long-form ``character_description``.

There is no second persona database: everything lands in the CMF file the
runtime already uses. The update is revisioned and audited by the service.
Import is explicit and never runs automatically at startup.

The 4000-character ``PersonaCore.character_description`` limit is checked here
*before* any write. A too-long description fails loudly; it is never silently
truncated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from companion.agent.persona_core import MAX_DESCRIPTION as DESCRIPTION_LIMIT
from companion.runtime import build_runtime
from companion.storage.database import Database

DEFAULT_FILE = (
    Path(__file__).resolve().parents[1] / "assets" / "ava_persona_core_v1.json"
)

CORE_FIELDS = (
    "core_values",
    "temperament",
    "preferences",
    "quirks",
    "contradictions",
    "conversational_boundaries",
    "voice_identity",
    "character_description",
)


def load_persona(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    core = data.get("persona_core") or {}
    description = str(core.get("character_description") or "")
    if not description.strip():
        raise ValueError("persona_core.character_description is required")
    if len(description) > DESCRIPTION_LIMIT:
        raise ValueError(
            f"character_description is {len(description)} chars, over the"
            f" {DESCRIPTION_LIMIT} limit; refusing to truncate silently"
        )
    unknown = set(core) - set(CORE_FIELDS)
    if unknown:
        raise ValueError(f"unknown persona_core fields: {', '.join(sorted(unknown))}")
    traits = data.get("traits") or {}
    if not traits:
        raise ValueError("traits are required")
    return data


async def _run(args) -> int:
    data = load_persona(args.file)
    name = str(data.get("display_name") or "").strip()
    if not name:
        raise ValueError("display_name is required")
    core = {key: data["persona_core"][key] for key in CORE_FIELDS if key in data["persona_core"]}

    if args.dry_run:
        description = core["character_description"]
        print(f"display_name={name}")
        print(f"traits={len(data['traits'])} fields")
        print(f"persona_core={len(core)} fields")
        print(f"character_description_chars={len(description)}")
        print("dry_run=True (nothing written)")
        return 0

    db = await Database(Path(args.db)).connect()
    try:
        await db.initialize()
        runtime = build_runtime(db, with_worker=False)
        manifest = await runtime.agent_service.repo.get_manifest() or {}
        agent_id = str(manifest.get("agent_id") or "")
        if not agent_id:
            print(
                "no agent identity in this database; create one first",
                file=sys.stderr,
            )
            return 1

        await runtime.agent_service.set_name(
            name, actor=args.actor, reason=args.reason
        )
        await runtime.agent_service.set_persona(
            data["traits"], actor=args.actor, reason=args.reason
        )
        updated = await runtime.agent_service.set_persona_core(
            core, actor=args.actor, reason=args.reason
        )
        print(
            f"agent_id={agent_id} display_name={name} "
            f"persona_core_revision={updated.revision} "
            f"character_description_chars={len(updated.character_description)}"
        )
        return 0
    finally:
        await db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/companion.db")
    parser.add_argument("--file", default=str(DEFAULT_FILE))
    parser.add_argument("--actor", default="admin")
    parser.add_argument(
        "--reason", default="apply_ava_persona_v1", help="audit reason"
    )
    parser.add_argument("--dry-run", action="store_true")
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
