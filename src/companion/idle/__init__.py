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

from .context import IdleMemoryContextBuilder  # noqa: E402
from .curiosity import CuriosityEngine  # noqa: E402
from .reflection import ReflectionEngine  # noqa: E402
from .research import ResearchEngine  # noqa: E402
from .scheduler import IdleScheduler  # noqa: E402


def build_idle_scheduler(
    runtime, *, deepseek=None, web_search=None, settings=None
) -> IdleScheduler:
    """Wire the idle pipeline over a running :class:`~companion.runtime.Runtime`.

    The runtime owns exactly one research engine and has already registered
    its ``RESEARCH_TASK`` handler with the worker. This builder reuses that
    engine unconditionally; injecting ``deepseek``/``web_search`` only affects
    the foreground reflection/curiosity providers used by tests, never the
    executing Worker's global handler. That avoids a second research engine
    (and a second provider) racing the real one.
    """
    reflection_provider = deepseek or DeepSeekProvider()
    settings = settings or runtime.repos.settings
    builder = IdleMemoryContextBuilder(
        runtime.memory_service, runtime.registry, audit=runtime.repos.audit
    )
    task_service = getattr(runtime, "task_service", None)
    research = getattr(runtime, "research_engine", None)
    if research is None:
        # Only a runtime built without the normal wiring needs a fallback.
        # This is not registered over an existing handler.
        from .sources import ResearchSourceGuard

        research = ResearchEngine(
            runtime.memory_service,
            runtime.repos.research,
            web_search or MockWebSearchProvider(),
            reflection_provider,
            settings=settings,
            state=runtime.repos.research_state,
            source_guard=ResearchSourceGuard(
                runtime.memory_service, runtime.registry
            ),
        )
    reflection = ReflectionEngine(
        runtime.memory_service,
        reflection_provider,
        builder,
        learning=runtime.repos.learning,
    )
    curiosity = CuriosityEngine(
        runtime.memory_service,
        reflection_provider,
        runtime.repos.research,
        builder,
        learning=runtime.repos.learning,
        task_service=task_service,
        source_guard=getattr(research, "source_guard", None),
        settings=settings,
    )
    return IdleScheduler(
        runtime.repos.settings,
        reflection,
        curiosity,
        research,
        learning=runtime.repos.learning,
        task_service=task_service,
    )


__all__ = [
    "build_idle_scheduler",
    "IdleScheduler",
    "ReflectionEngine",
    "CuriosityEngine",
    "ResearchEngine",
    "IdleMemoryContextBuilder",
    "SYSTEM_ADMIN",
    "IDLE_AGENT",
]
