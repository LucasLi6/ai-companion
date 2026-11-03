"""P2 — REPL never implicitly registers a new identity; it reuses a chosen
person across restarts.
"""

from __future__ import annotations

from companion.cli import _select_repl_person, main


async def test_repl_reuses_existing_person(register, rt):
    p = await register("Alice")
    person, err = await _select_repl_person(rt, p.person_id)
    assert err is None
    assert person.person_id == p.person_id


async def test_repl_requires_registration_when_empty(rt):
    person, err = await _select_repl_person(rt, None)
    assert person is None
    assert "no registered person" in err


async def test_repl_requires_person_when_multiple(register, rt):
    await register("A")
    await register("B")
    person, err = await _select_repl_person(rt, None)
    assert person is None
    assert "multiple" in err


async def test_repl_unknown_person_id(rt):
    person, err = await _select_repl_person(rt, "person_missing")
    assert person is None
    assert "not found" in err


async def test_repl_selects_single_person_implicitly(register, rt):
    p = await register("Solo")
    person, err = await _select_repl_person(rt, None)
    assert err is None
    assert person.person_id == p.person_id


def test_repl_does_not_auto_register(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])

    # No people: REPL refuses to invent one.
    assert main(["--db", db, "repl"]) == 1
    assert "no registered person" in capsys.readouterr().out

    # Register one person, then run REPL as that person across two sessions.
    main([
        "--db", db, "person", "add", "--name", "Owner", "--role", "owner",
        "--allow-processing", "--allow-persistence", "--allow-memory-learning",
    ])
    person_id = capsys.readouterr().out.strip().splitlines()[-1]

    monkeypatch.setattr(
        "builtins.input",
        lambda *a, **k: (_ for _ in ()).throw(EOFError()),
    )
    assert main(["--db", db, "repl", "--person", person_id]) == 0
    assert main(["--db", db, "repl", "--person", person_id]) == 0

    # Only the original person exists; no repl-user identity was created.
    capsys.readouterr()  # flush the REPL greeting output
    main(["--db", db, "person", "list"])
    out = capsys.readouterr().out
    assert "repl-user" not in out
    lines = out.strip().splitlines()
    assert len(lines) == 1  # exactly one registered person
    assert person_id in lines[0]
