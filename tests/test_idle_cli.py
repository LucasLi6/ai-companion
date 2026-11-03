"""CLI integration for the idle research commands."""

from __future__ import annotations

from companion.cli import main


def test_research_queue_list_run(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "init"]) == 0
    capsys.readouterr()

    assert main([
        "--db", db, "research", "queue",
        "--question", "OTA 为什么不稳定？",
        "--source-kind", "manual",
    ]) == 0
    task_id = capsys.readouterr().out.strip()
    assert task_id.startswith("research_")

    assert main(["--db", db, "research", "list"]) == 0
    out = capsys.readouterr().out
    assert task_id in out
    assert "PENDING" in out

    # Research runs against the mock web search (off by default), completes and
    # consolidates a world-knowledge memory.
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
