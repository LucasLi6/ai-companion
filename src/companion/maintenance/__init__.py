"""Guarded maintenance commands that never run implicitly.

Every command in this package is dry-run by default and requires an explicit
``--apply``. They operate on one SQLite file with the engine's own backup API
and never touch a live database behind the user's back.

Exports are lazy so that ``python -m companion.maintenance.reset_user_data``
does not import the module twice (which runpy reports as a warning).
"""

from typing import Any

__all__ = ["ResetReport", "reset_user_data"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import companion.maintenance.reset_user_data as module

        return getattr(module, name)
    raise AttributeError(name)
