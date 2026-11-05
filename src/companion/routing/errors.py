"""Explicit router failure classes.

A Semantic-Mode failure is never reported as a successful silence and never
falls back to text/keyword routing.
"""

from __future__ import annotations


class RouterError(Exception):
    """Base class for routing failures."""


class RouterTimeout(RouterError):
    pass


class RouterUnavailable(RouterError):
    pass


class RouterInvalidOutput(RouterError):
    """The model returned something that is not a router decision."""


class RouterValidationError(RouterError):
    """A structurally valid payload violated a local trust constraint."""


class ToolExecutionError(RouterError):
    pass
