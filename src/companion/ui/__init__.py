"""Visual Memory Manager (NiceGUI) — the local management web interface.

Start it with ``python -m companion ui`` (served on 127.0.0.1:8080 by
default). The UI is a *management* surface: it operates with an
``is_admin`` :class:`~companion.memory.models.AccessContext`, so it can see
and edit every memory, person and setting. It never sends SQL directly —
every read and write goes through the same repositories and the
:class:`~companion.memory.service.MemoryService` that the CLI uses.
"""

from companion.ui.app import create_app, serve

__all__ = ["create_app", "serve"]
