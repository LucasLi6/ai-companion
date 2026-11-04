"""Local routing between the Fast Path and the Deliberation Path (V0.6).

The Fast Path exists to remove one full model call from ordinary, clearly
addressed conversation:

    input -> Runtime Snapshot -> local Policy -> Context Assembler -> Response LLM

It is selected from **trusted event metadata**, never from the text. In
particular, a message merely containing the companion's name does not bypass
addressee analysis; only an explicit trusted ``addressee_hint == "agent"``
does. The Fast Path still runs every hard gate before generation: speaker
registration, consent (processing / persistence / learning / external AI),
explicit silence, safety, quiet mode, and permission denial.

The Deliberation Path (the existing cognitive model call) is kept for cases the
local metadata cannot settle:

- the addressee is unknown, especially in a multi-person conversation;
- an explicit "remember this" request needs a real memory candidate and a
  confirmed write before the reply can acknowledge it;
- an unknown/unreliable speaker state (unregistered speakers never reach here).

A local decision is *not* a fake model analysis. It is recorded with
``provider="local_fast_path"`` and an explicit local rule name, and it never
claims that a model interpreted the turn.
"""

from __future__ import annotations

from dataclasses import dataclass

LOCAL_PROVIDER = "local_fast_path"

ROUTE_FAST = "fast_path"
ROUTE_DELIBERATION = "deliberation"

REASON_DIRECT_ADDRESS = "trusted_direct_agent_address"
REASON_EXPLICIT_MEMORY = "explicit_memory_requires_deliberation"
REASON_ADDRESSEE_UNKNOWN = "addressee_not_locally_certain"
REASON_HUMAN_ADDRESS = "explicit_human_addressee"
REASON_ROUTING_DISABLED = "fast_path_disabled"


@dataclass(frozen=True)
class FastPathDecision:
    route: str
    reason: str

    @property
    def is_fast(self) -> bool:
        return self.route == ROUTE_FAST


def decide_route(
    *,
    addressee_hint: str,
    explicit_remember: bool,
    fast_path_enabled: bool = True,
) -> FastPathDecision:
    """Choose the route from trusted metadata only.

    ``addressee_hint`` is the local, trusted event metadata (``agent``,
    ``human`` or ``unknown``). The message text is deliberately not an input.
    """
    if not fast_path_enabled:
        return FastPathDecision(ROUTE_DELIBERATION, REASON_ROUTING_DISABLED)

    hint = str(addressee_hint or "unknown").strip().lower()
    if hint == "human":
        return FastPathDecision(ROUTE_DELIBERATION, REASON_HUMAN_ADDRESS)
    if hint != "agent":
        return FastPathDecision(ROUTE_DELIBERATION, REASON_ADDRESSEE_UNKNOWN)
    if explicit_remember:
        # The acknowledgement must follow a real, durable write; the cognitive
        # call is what produces the candidate that the writer consolidates.
        return FastPathDecision(ROUTE_DELIBERATION, REASON_EXPLICIT_MEMORY)
    return FastPathDecision(ROUTE_FAST, REASON_DIRECT_ADDRESS)
