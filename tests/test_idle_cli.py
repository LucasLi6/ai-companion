"""CLI integration for the idle research commands."""

from __future__ import annotations

import asyncio

from companion.cli import main
from companion.runtime import build_runtime
from companion.storage.database import Database


def _set_research_enabled(db: str) -> None:
    async def _run() -> None:
        database = await Database(db).connect()
        await database.initialize()
        rt = build_runtime(database)
        await rt.repos.settings.set("research.enabled", "true")
        await database.close()

    asyncio.run(_run())


def test_research_queue_list_run(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "init"]) == 0
    capsys.readouterr()

    assert main([
        "--db", db, "research", "queue",
        "--question", "OTA 为什么不稳定？",
        "--source-kind", "manual",
    ]) == 0
    queued = capsys.readouterr().out
    task_id = queued.splitlines()[0].strip()
    assert task_id.startswith("research_")
    # The command also reports the durable worker job it created.
    assert "job_status=PENDING" in queued

    assert main(["--db", db, "research", "list"]) == 0
    out = capsys.readouterr().out
    assert task_id in out
    assert "PENDING" in out

    # Research is gated by research.enabled (off by default).
    assert main(["--db", db, "research", "run", "--limit", "1"]) == 0
    assert "disabled" in capsys.readouterr().out

    # Once enabled, it runs against the mock web search, completes and
    # consolidates a world-knowledge memory.
    _set_research_enabled(db)
    assert main(["--db", db, "research", "run", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert task_id in out
    assert "COMPLETED" in out

    assert main(["--db", db, "memory", "search", "--kind", "WORLD_KNOWLEDGE"]) == 0
    assert "WORLD_KNOWLEDGE" in capsys.readouterr().out


def test_idle_run_disabled_by_default(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "init"]) == 0
    capsys.readouterr()

    assert main(["--db", db, "idle", "run"]) == 0
    out = capsys.readouterr().out
    assert "disabled" in out
