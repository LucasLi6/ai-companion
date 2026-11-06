"""V0.9.8 one-shot proactive conversation.

The package contains exactly the two logical components from the design:
:class:`~companion.proactive.manager.ProactiveManager` (timing, staleness,
delivery) and :class:`~companion.proactive.judge.ProactiveJudge` (whether to
speak). Imports are kept lazy here so provider modules can import the judge
models without a package-level cycle.
"""

__all__ = ["ProactiveManager", "ProactiveJudge"]


def __getattr__(name):
    if name == "ProactiveManager":
        from .manager import ProactiveManager

        return ProactiveManager
    if name == "ProactiveJudge":
        from .judge import ProactiveJudge

        return ProactiveJudge
    raise AttributeError(name)
