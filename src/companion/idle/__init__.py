"""Idle cognitive & research engines (Phase D).

The idle pipeline runs *off* the same memory service, research store and
providers the rest of the system uses, and is gated by the ``idle.enabled``
setting. Every output is mock-driven and labelled as such, and nothing an AI
produces is ever written as a human-confirmed fact: reflections and research
summaries enter as ``CANDIDATE`` / ``NEEDS_REVIEW`` records for a human to
approve.
"""

from __future__ import annotations

from companion.memory.models import AccessContext

# The idle engines read the whole store as the trusted system admin, but act
# (propose/queue) as a non-admin "idle" actor so they can never self-confirm.
# Defined before the submodule imports below so the engines can ``from . import``
# them without a circular-import failure.
SYSTEM_ADMIN = AccessContext("admin", is_admin=True, role="idle")
IDLE_AGENT = AccessContext("idle", is_admin=False, role="idle")

from companion.providers.deepseek import DeepSeekProvider  # noqa: E402
from companion.providers.websearch import MockWebSearchProvider  # noqa: E402

from .curiosity import CuriosityEngine  # noqa: E402
from .reflection import ReflectionEngine  # noqa: E402
from .research import ResearchEngine  # noqa: E402
from .scheduler import IdleScheduler  # noqa: E402


def build_idle_scheduler(runtime, *, deepseek=None, web_search=None) -> IdleScheduler:
    """Wire the idle pipeline over a running :class:`~companion.runtime.Runtime`.

    ``deepseek`` and ``web_search`` default to mock providers; callers may pass
    real ones (e.g. a settings-gated web search) without touching the engines.
    """
    deepseek = deepseek or DeepSeekProvider()
    web_search = web_search or MockWebSearchProvider()
    reflection = ReflectionEngine(runtime.memory_service, deepseek)
    curiosity = CuriosityEngine(runtime.memory_service, deepseek, runtime.repos.research)
    research = ResearchEngine(
        runtime.memory_service, runtime.repos.research, web_search, deepseek
    )
    return IdleScheduler(runtime.repos.settings, reflection, curiosity, research)


__all__ = [
    "build_idle_scheduler",
    "IdleScheduler",
    "ReflectionEngine",
    "CuriosityEngine",
    "ResearchEngine",
    "SYSTEM_ADMIN",
    "IDLE_AGENT",
]
