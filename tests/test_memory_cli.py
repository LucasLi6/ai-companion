"""CLI integration for the unified memory store."""

from __future__ import annotations

from companion.cli import main


def test_memory_add_search_show_retract_delete(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "init"]) == 0
    capsys.readouterr()

    assert main([
        "--db", db, "memory", "add",
        "--kind", "PERSONAL_FACT",
        "--content", "正在开发一个机器人",
        "--human-confirmed",
    ]) == 0
    memory_id = capsys.readouterr().out.strip()
    assert memory_id.startswith("mem_")

    # Search finds it.
    assert main(["--db", db, "memory", "search", "--query", "机器人"]) == 0
    out = capsys.readouterr().out
    assert memory_id in out
    assert "PERSONAL_FACT" in out

    # Show prints the full record.
    assert main(["--db", db, "memory", "show", memory_id]) == 0
    out = capsys.readouterr().out
    assert "正在开发一个机器人" in out
    assert "human_confirmed: True" in out

    # Retract hides it from default search but keeps it recoverable.
    assert main(["--db", db, "memory", "retract", memory_id]) == 0
    capsys.readouterr()
    assert main(["--db", db, "memory", "search"]) == 0
    assert memory_id not in capsys.readouterr().out

    assert main(["--db", db, "memory", "search", "--include-retracted"]) == 0
    assert memory_id in capsys.readouterr().out

    # Delete removes it entirely, even from retracted search.
    assert main(["--db", db, "memory", "delete", memory_id]) == 0
    capsys.readouterr()
    assert main(["--db", db, "memory", "search", "--include-retracted"]) == 0
    assert memory_id not in capsys.readouterr().out


def test_memory_add_with_subject_and_confirm_flow(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    main([
        "--db", db, "person", "add", "--name", "Owner", "--role", "owner",
        "--allow-processing", "--allow-persistence", "--allow-memory-learning",
    ])
    person_id = capsys.readouterr().out.strip().splitlines()[-1]

    assert main([
        "--db", db, "memory", "add",
        "--content", "Owner likes robots",
        "--subject", f"PERSON:{person_id}",
        "--status", "CANDIDATE",
    ]) == 0
    memory_id = capsys.readouterr().out.strip()

    # Candidate is visible to search and shows the subject.
    assert main(["--db", db, "memory", "show", memory_id]) == 0
    out = capsys.readouterr().out
    assert f"subject: PERSON:{person_id}" in out
    assert "status: CANDIDATE" in out

    # Human confirmation promotes it.
    assert main(["--db", db, "memory", "confirm", memory_id]) == 0
    assert main(["--db", db, "memory", "show", memory_id]) == 0
    out = capsys.readouterr().out
    assert "status: ACTIVE" in out
    assert "human_confirmed: True" in out


def test_memory_show_missing_returns_error(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    assert main(["--db", db, "memory", "show", "mem_missing"]) == 1
    assert "not found" in capsys.readouterr().out


def test_memory_link_and_delete_missing(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    capsys.readouterr()
    main(["--db", db, "memory", "add", "--content", "A"])
    a = capsys.readouterr().out.strip()
    main(["--db", db, "memory", "add", "--content", "B"])
    b = capsys.readouterr().out.strip()

    assert main(["--db", db, "memory", "link", a, b, "--type", "supports"]) == 0
    assert f"linked {a} -> {b}" in capsys.readouterr().out

    assert main(["--db", db, "memory", "delete", "mem_missing"]) == 1
    assert "not found" in capsys.readouterr().out
