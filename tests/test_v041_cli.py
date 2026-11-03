"""V0.4.1 admin CLI for the character core."""

from __future__ import annotations

from companion.cli import main


def test_persona_core_show_defaults_then_set_and_revision(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])

    assert main(["--db", db, "agent", "persona-core", "show"]) == 0
    out = capsys.readouterr().out
    assert "revision: 1" in out
    assert "updated_by: system" in out
    assert "不编造个人经历" in out

    assert main([
        "--db", db, "agent", "persona-core", "set",
        "--json",
        '{"core_values": ["诚实", "具体"], "quirks": ["会追问含糊之处"],'
        ' "temperament": {"humor": "dry"}, "boundaries": ["不编造经历"]}',
        "--reason", "test setup",
    ]) == 0
    out = capsys.readouterr().out
    assert "revision 2" in out
    assert '"诚实"' in out
    assert "不是 JSON" not in out

    assert main(["--db", db, "agent", "persona-core", "show"]) == 0
    out = capsys.readouterr().out
    assert "revision: 2" in out
    assert "updated_by: admin" in out
    assert '["诚实", "具体"]' in out
    assert '["不编造经历"]' in out


def test_persona_core_set_rejects_bad_json_without_writing(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])

    assert main([
        "--db", db, "agent", "persona-core", "set", "--json", "{not json",
    ]) == 1
    assert "invalid JSON" in capsys.readouterr().out

    assert main([
        "--db", db, "agent", "persona-core", "show",
    ]) == 0
    assert "revision: 1" in capsys.readouterr().out


def test_persona_core_set_rejects_unknown_fields(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])

    assert main([
        "--db", db, "agent", "persona-core", "set",
        "--json", '{"not_a_field": true}',
    ]) == 1
    assert "invalid persona core" in capsys.readouterr().out
