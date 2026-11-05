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

import re
from dataclasses import dataclass

from companion.domain.enums import CognitiveAction, ReasonCode, ReplyNecessity

LOCAL_PROVIDER = "local_fast_path"

ROUTE_FAST = "fast_path"
ROUTE_DELIBERATION = "deliberation"

REASON_DIRECT_ADDRESS = "trusted_direct_agent_address"
REASON_EXPLICIT_MEMORY = "explicit_memory_requires_deliberation"
REASON_ADDRESSEE_UNKNOWN = "addressee_not_locally_certain"
REASON_HUMAN_ADDRESS = "explicit_human_addressee"
REASON_ROUTING_DISABLED = "fast_path_disabled"

# A local reply-necessity value is metadata, not a router. These markers only
# distinguish "an answer/task is explicitly requested" (REQUIRED) from an
# ordinary greeting or statement (USEFUL); they never decide silence.
_QUESTION_RE = re.compile(r"[?？]\s*$")
_REQUEST_RE = re.compile(
    r"(?:请|麻烦|帮我|帮忙|能不能|可不可以|能否|要不要|是不是|为什么|怎么|如何|什么|哪)"
)
_ENDING_PARTICLE_RE = re.compile(r"(?:吗|呢|么)[？?]?\s*$")
# The speaker asks for a pause / says they are waiting for someone else. The
# match must cover the whole utterance and there must be no question mark, so
# "等一下，这个怎么弄？" still reaches the direct-question path.
_WAIT_RE = re.compile(
    r"(?:(?:先|稍|请)?等(?:我|我们)?(?:一下|一会儿?|会|等)(?:再|先)?"
    r"(?:说|聊|继续)?"
    r"|(?:我|我们)(?:先|还在|在)等(?:别人|朋友|其他人)"
    r"|(?:我|我们)(?:还)?(?:没|没有)说完"
    r"|(?:先)?别急(?:着)?(?:回|回复|说)?)\s*[。.!！…]*\s*$"
)


def looks_like_wait_request(text: str | None) -> bool:
    """Whether the utterance itself asks Ava to hold its reply for now."""
    value = " ".join(str(text or "").split())
    if not value or "?" in value or "？" in value:
        return False
    return bool(_WAIT_RE.search(value))


def looks_like_direct_request(text: str | None) -> bool:
    """Whether the utterance explicitly asks for an answer or an action.

    Deliberately small and used only to label reply necessity. It cannot make
    a turn silent: a direct, trusted agent address still defaults to a reply.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return False
    return bool(
        _QUESTION_RE.search(value)
        or _ENDING_PARTICLE_RE.search(value)
        or _REQUEST_RE.search(value)
    )


@dataclass(frozen=True)
class FastPathDecision:
    route: str
    reason: str

    @property
    def is_fast(self) -> bool:
        return self.route == ROUTE_FAST


@dataclass(frozen=True)
class TurnAssessment:
    """A local, auditable answer to two separate questions.

    ``addressee`` says who the utterance targets; ``reply_necessity`` says
    whether Ava needs to speak. They are never derived from one another:
    a confirmed agent address is not automatically ``REQUIRED``, and a
    ``WAIT``/``IGNORE`` decision is not a missing reply.

    ``needs_model`` means the trusted local signals cannot settle the turn
    (an unclear addressee, an explicit memory request, or deliberation
    disabled/unauthorised). The caller then uses the existing Deliberation
    Path instead of inventing a local answer.
    """

    action: CognitiveAction
    reason: ReasonCode
    reply_necessity: ReplyNecessity
    rule: str
    addressee: str = "unknown"
    needs_model: bool = False

    @property
    def is_local(self) -> bool:
        return not self.needs_model


def assess_turn(
    *,
    addressee_hint: str,
    text: str | None,
    explicit_silence: bool,
    quiet_mode: bool,
    explicit_remember: bool,
    fast_path_enabled: bool = True,
    model_available: bool = True,
) -> TurnAssessment:
    """Assess addressee and reply necessity from trusted local signals.

    Priority mirrors the policy's hard gates, so a local decision can never
    disagree with the final policy: explicit silence, an explicit human
    addressee and quiet mode are always local. Only a confirmed agent address
    is eligible for a local RESPOND, and only when generation is authorised.
    """
    hint = str(addressee_hint or "unknown").strip().lower()
    if hint not in ("agent", "human", "unknown"):
        hint = "unknown"

    # 1. The speaker explicitly asked for no reply (local detector only).
    if explicit_silence:
        return TurnAssessment(
            action=CognitiveAction.IGNORE,
            reason=ReasonCode.EXPLICIT_SILENCE_REQUEST,
            reply_necessity=ReplyNecessity.UNNECESSARY,
            rule="local_explicit_silence",
            addressee=hint,
        )

    # 2. A confirmed human-to-human utterance is not addressed to Ava.
    if hint == "human":
        return TurnAssessment(
            action=CognitiveAction.IGNORE,
            reason=ReasonCode.DIRECT_HUMAN_ADDRESS,
            reply_necessity=ReplyNecessity.UNNECESSARY,
            rule="local_human_addressee",
            addressee=hint,
        )

    # 3. Quiet mode asks Ava to follow without speaking.
    if quiet_mode:
        return TurnAssessment(
            action=CognitiveAction.WAIT,
            reason=ReasonCode.QUIET_MODE,
            reply_necessity=ReplyNecessity.UNNECESSARY,
            rule="local_quiet_mode",
            addressee=hint,
        )

    # 3b. The speaker explicitly asks for a pause or is waiting for someone
    # else. This is a deliberate, local WAIT: no reply text is produced.
    if looks_like_wait_request(text):
        return TurnAssessment(
            action=CognitiveAction.WAIT,
            reason=ReasonCode.RESPONSE_NOT_NECESSARY,
            reply_necessity=ReplyNecessity.UNNECESSARY,
            rule="local_wait_request",
            addressee=hint,
        )

    # 4. Anything the local metadata cannot settle is deliberated.
    if hint != "agent":
        return _deferred(ReasonCode.UNKNOWN_ADDRESSEE, hint)
    if explicit_remember:
        # The acknowledgement must follow a real, durable write, which only the
        # deliberation path can produce.
        return _deferred(ReasonCode.UNKNOWN, hint, rule="local_explicit_memory")
    if not fast_path_enabled:
        return _deferred(ReasonCode.UNKNOWN, hint, rule="local_fast_path_disabled")
    if not model_available:
        # No authorised generation path: let the standard permission gate
        # record PERMISSION_DENIED instead of speaking locally without consent.
        return _deferred(ReasonCode.PERMISSION_DENIED, hint, rule="local_no_provider")

    # 5. Confirmed direct address: Ava should take part. Necessity is a
    # truthful label, not a forced REQUIRED for every greeting.
    requirement = (
        ReplyNecessity.REQUIRED
        if looks_like_direct_request(text)
        else ReplyNecessity.USEFUL
    )
    return TurnAssessment(
        action=CognitiveAction.RESPOND,
        reason=ReasonCode.DIRECT_AGENT_ADDRESS,
        reply_necessity=requirement,
        rule="local_direct_address",
        addressee=hint,
    )


def _deferred(
    reason: ReasonCode, hint: str, *, rule: str = "local_deliberation"
) -> TurnAssessment:
    return TurnAssessment(
        action=CognitiveAction.WAIT,
        reason=reason,
        reply_necessity=ReplyNecessity.OPTIONAL,
        rule=rule,
        addressee=hint,
        needs_model=True,
    )


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
