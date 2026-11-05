"""Smoke tests for the NiceGUI management UI.

These only verify that the app builds and registers every page, plus that the
page handlers are callable async functions — they do not start a real server.
Skipped when NiceGUI is not installed.
"""

from __future__ import annotations

import inspect

import pytest

nicegui = pytest.importorskip("nicegui")

from companion.ui.app import create_app  # noqa: E402


def test_all_pages_registered(tmp_path):
    app = create_app(tmp_path / "ui.db")

    paths = {r.path for r in app.routes}
    for expected in (
        "/",
        "/chat",
        "/monitor",
        "/memory",
        "/memory/{memory_id}",
        "/people",
        "/research",
        "/audit",
        "/settings",
    ):
        assert expected in paths, f"missing route {expected}"


def test_page_handlers_are_async(tmp_path):
    create_app(tmp_path / "ui.db")
    from companion.ui import app as ui_app

    for name in (
        "dashboard",
        "chat_page",
        "monitor_page",
        "memory_explorer",
        "memory_detail",
        "person_manager",
        "research_manager",
        "audit_log",
        "settings_page",
    ):
        handler = getattr(ui_app, name)
        assert inspect.iscoroutinefunction(handler), f"{name} must be async"
