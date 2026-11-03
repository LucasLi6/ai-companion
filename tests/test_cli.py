"""CLI integration tests: exercise the argparse surface end to end."""

from __future__ import annotations

from companion.cli import main


def test_init_and_person_add_and_list(tmp_path, capsys):
    db = str(tmp_path / "cli.db")

    assert main(["--db", db, "init"]) == 0
    assert "initialised" in capsys.readouterr().out.lower()

    assert main([
        "--db", db, "person", "add", "--name", "Owner", "--role", "owner",
        "--allow-processing", "--allow-persistence", "--allow-memory-learning",
    ]) == 0
    person_id = capsys.readouterr().out.strip()
    assert person_id.startswith("person_")

    assert main(["--db", db, "person", "list"]) == 0
    out = capsys.readouterr().out
    assert "Owner" in out
    assert person_id in out


def test_say_respond_and_ignore(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    main([
        "--db", db, "person", "add", "--name", "Owner", "--role", "owner",
        "--allow-processing", "--allow-persistence", "--allow-memory-learning",
    ])
    person_id = capsys.readouterr().out.strip().splitlines()[-1]

    assert main(["--db", db, "say", "--person", person_id, "--to", "agent", "--text", "你好"]) == 0
    out = capsys.readouterr().out
    assert "[DECISION] RESPOND" in out
    assert "模拟回复" in out

    assert main(["--db", db, "say", "--person", person_id, "--to", "human", "--text", "对朋友说"]) == 0
    out = capsys.readouterr().out
    assert "[DECISION] IGNORE" in out
    assert "模拟回复" not in out


def test_say_inspect_quiet_jobs(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    main([
        "--db", db, "person", "add", "--name", "Owner", "--role", "owner",
        "--allow-processing", "--allow-persistence", "--allow-memory-learning",
    ])
    person_id = capsys.readouterr().out.strip().splitlines()[-1]

    main(["--db", db, "say", "--person", person_id, "--to", "agent", "--text", "inspect me"])
    say_out = capsys.readouterr().out
    event_line = next(line for line in say_out.splitlines() if line.startswith("[EVENT]"))
    event_id = event_line.split("Saved: ")[1]

    assert main(["--db", db, "inspect", event_id]) == 0
    out = capsys.readouterr().out
    assert "inspect me" in out
    assert "decision: RESPOND" in out
    assert "CONSOLIDATED" in out

    assert main(["--db", db, "quiet", "on"]) == 0
    main(["--db", db, "say", "--person", person_id, "--to", "agent", "--text", "quiet please"])
    out = capsys.readouterr().out
    assert "[DECISION] WAIT" in out

    assert main(["--db", db, "quiet", "off"]) == 0
    assert main(["--db", db, "jobs", "list"]) == 0
    assert "MEMORY_CONSOLIDATION" in capsys.readouterr().out


def test_demo_runs_on_isolated_db(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "[DECISION] RESPOND" in out
    assert "[DECISION] IGNORE" in out
    assert "[DECISION] WAIT" in out


def test_help(tmp_path, capsys):
    # argparse prints --help to stdout and exits 0.
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()
