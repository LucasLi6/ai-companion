"""Phase 7 — CLI/REPL behaviour for the real provider switch.

These are CLI-level tests (``main([...])``). They never touch the network:
the DeepSeek path is only exercised for its *configuration* failures, and the
interactive chat test uses the default mock provider.
"""

from __future__ import annotations

from companion.cli import main


def _feed(lines, monkeypatch):
    iterator = iter(lines)

    def fake_input(*args, **kwargs):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError()

    monkeypatch.setattr("builtins.input", fake_input)


def _add_person(capsys, db, *extra):
    main([
        "--db", db, "person", "add", "--name", "Owner", "--role", "owner",
        "--allow-processing", "--allow-persistence", "--allow-memory-learning",
        *extra,
    ])
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_repl_mock_chat_and_commands(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    person_id = _add_person(capsys, db)

    _feed(["你好", "/status", "/quiet on", "/to human", "自言自语", "/quit"], monkeypatch)
    assert main(["--db", db, "repl", "--person", person_id]) == 0

    out = capsys.readouterr().out
    assert "provider: mock" in out
    assert "assistant> 模拟回复：我收到你的话了。" in out
    assert "quiet_mode: off" in out
    assert "quiet_mode=on" in out
    assert "addressee -> human" in out
    # Explicit human addressee is IGNORE, independent of the model.
    assert "[IGNORE (direct_human_address)]" in out


def test_repl_deepseek_without_key_fails_loudly(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    person_id = _add_person(capsys, db)

    _feed([], monkeypatch)
    code = main(["--db", db, "repl", "--person", person_id, "--provider", "deepseek"])

    assert code == 1
    out = capsys.readouterr().out
    assert "deepseek error" in out
    assert "DEEPSEEK_API_KEY" in out
    # Never falls back to a mock reply.
    assert "assistant>" not in out


def test_say_deepseek_without_key_fails_loudly(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    person_id = _add_person(capsys, db)

    code = main([
        "--db", db, "say", "--person", person_id, "--to", "agent",
        "--text", "你好", "--provider", "deepseek",
    ])

    assert code == 1
    out = capsys.readouterr().out
    assert "deepseek error" in out
    assert "DEEPSEEK_API_KEY" in out


def test_deepseek_check_offline_never_prints_the_key(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-super-secret-do-not-print")
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])

    assert main(["--db", db, "deepseek", "check"]) == 0

    out = capsys.readouterr().out
    assert "api_key: configured" in out
    assert "model: deepseek-flash" in out
    assert "config check passed" in out
    assert "sk-super-secret-do-not-print" not in out


def test_deepseek_check_offline_without_key_fails(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])

    assert main(["--db", db, "deepseek", "check"]) == 1
    out = capsys.readouterr().out
    assert "api_key: MISSING" in out
    assert "DEEPSEEK_API_KEY" in out


def test_person_list_shows_external_ai_grant(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "init"])
    _add_person(capsys, db, "--allow-external-ai")

    main(["--db", db, "person", "list"])
    out = capsys.readouterr().out
    assert "external_ai=1" in out
