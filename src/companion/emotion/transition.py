"""Dialogue-driven affect transitions (V0.4.2).

V0.4.1 could already apply an appraisal rule, but time decay was the dominant
mechanism: a strong old activation simply outranked a new event, so Ava could
still *look* frustrated after a clearly warm or playful turn. V0.4.2 makes the
conversation event the primary cause of change and time decay an auxiliary
cleanup:

- a validated event applies its full stimulus;
- old activations that belong to a different scene/frame are suppressed by
  *scene relevance* before the new event is composed;
- a sufficiently strong new event can re-rank a stale dominant emotion instead
  of being permanently outvoted by ``max(activations)``;
- a pragmatic frame change (playful → conflict, conflict → repair, etc.) can
  move affect even when no discrete appraisal event fired;
- repair lowers relationship tension.

Everything here is a pure function over an :class:`EmotionState`. It never
calls a model, writes storage, changes an action or grants permission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Mapping, Optional

from companion.emotion.dynamics import (
    apply_emotional_stimulus,
    clamp,
    derive_acute,
)
from companion.emotion.models import AppraisalRule, EmotionProfile, EmotionState
from companion.emotion.parameters import DEFAULT_CONFIG, EmotionDynamicsConfig

EVENT = "EVENT"
FRAME_CHANGE = "FRAME_CHANGE"
SAME_FRAME_EVENT = "SAME_FRAME_EVENT"
REPAIR = "REPAIR"
NONE = "NONE"

_TOPIC_RELEVANCE = {
    "CONTINUE": 1.0,
    "RETURN": 0.85,
    "SHIFT": 0.25,
    "UNCERTAIN": 0.60,
}

# Frame -> (emotion, base intensity, tension delta). Tension delta is applied
# relative to the remaining headroom (positive) or to the current value
# (negative), and is always bounded.
_FRAME_STIMULI = {
    "CONFLICT": ("FRUSTRATION", 0.20, 0.12),
    "REPAIR": ("RELIEF", 0.16, -0.20),
    "SUPPORT": ("CONCERN", 0.14, 0.0),
    "PLAYFUL": ("JOY", 0.12, -0.02),
    "SERIOUS": ("SERIOUS", 0.14, 0.0),
}


@dataclass(frozen=True)
class ConversationalAffect:
    """The trusted conversational inputs to one affect transition.

    Built by the runtime from the validated pragmatic reading and the trusted
    scene; never from unchecked model prose. ``previous_frame`` is the frame
    before this turn, so the transition can distinguish an unchanged frame from
    a real frame shift.
    """

    event_type: Optional[str] = None
    primary_act: str = "OTHER"
    interaction_frame: str = "UNKNOWN"
    previous_frame: str = "UNKNOWN"
    refers_to_previous_turn: bool = False
    setup_payoff_role: str = "NONE"
    topic_transition: str = "UNCERTAIN"
    target: str = "UNKNOWN"
    social_intent: str = "UNCERTAIN"
    confidence: float = 0.0
    actor_person_id: Optional[str] = None

    @property
    def frame_changed(self) -> bool:
        current = str(self.interaction_frame or "UNKNOWN").upper()
        previous = str(self.previous_frame or "UNKNOWN").upper()
        return current not in ("UNKNOWN", "") and current != previous

    def scene_relevance(self) -> float:
        """How much this turn belongs to the scene the old emotion lived in."""
        key = str(self.topic_transition or "UNCERTAIN").upper()
        relevance = _TOPIC_RELEVANCE.get(key, 0.60)
        if self.refers_to_previous_turn:
            relevance += 0.15
        if str(self.setup_payoff_role or "NONE").upper() == "PAYOFF":
            relevance += 0.10
        if self.frame_changed:
            # Even a continuing topic does not make an old conflict reaction
            # relevant once the interaction frame has moved on.
            relevance *= 0.85
        return clamp(relevance, 0.0, 1.0)


@dataclass(frozen=True)
class EmotionalTransitionCause:
    """Small, auditable explanation of what moved the affect."""

    code: str
    reason: str
    previous_frame: str
    frame: str
    event_type: Optional[str] = None
    rule_name: Optional[str] = None
    scene_relevance: float = 0.0

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "reason": self.reason,
            "previous_frame": self.previous_frame,
            "frame": self.frame,
            "event_type": self.event_type,
            "rule_name": self.rule_name,
            "scene_relevance": round(self.scene_relevance, 4),
        }


@dataclass(frozen=True)
class AffectTransition:
    state: EmotionState
    cause: EmotionalTransitionCause
    changed: bool = False


def build_conversational_affect(
    *,
    pragmatic=None,
    signal=None,
    previous_frame: str = "UNKNOWN",
    actor_person_id: Optional[str] = None,
) -> ConversationalAffect:
    """Normalize trusted pragmatic/event hints for the transition function."""
    if pragmatic is not None:
        return ConversationalAffect(
            event_type=getattr(signal, "event_type", None),
            primary_act=str(getattr(pragmatic, "primary_act", "OTHER")),
            interaction_frame=str(
                getattr(pragmatic, "interaction_frame", "UNKNOWN")
            ),
            previous_frame=previous_frame or "UNKNOWN",
            refers_to_previous_turn=bool(
                getattr(pragmatic, "refers_to_previous_turn", False)
            ),
            setup_payoff_role=str(
                getattr(pragmatic, "setup_payoff_role", "NONE")
            ),
            topic_transition=str(
                getattr(pragmatic, "topic_transition", "UNCERTAIN")
            ),
            target=str(getattr(signal, "target", "UNKNOWN") or "UNKNOWN"),
            social_intent=str(
                getattr(signal, "social_intent", "UNCERTAIN") or "UNCERTAIN"
            ),
            confidence=float(getattr(pragmatic, "confidence", 0.0) or 0.0),
            actor_person_id=actor_person_id,
        )
    return ConversationalAffect(
        event_type=getattr(signal, "event_type", None),
        previous_frame=previous_frame or "UNKNOWN",
        target=str(getattr(signal, "target", "UNKNOWN") or "UNKNOWN"),
        social_intent=str(
            getattr(signal, "social_intent", "UNCERTAIN") or "UNCERTAIN"
        ),
        confidence=float(getattr(signal, "confidence", 0.0) or 0.0),
        actor_person_id=actor_person_id,
    )


def transition_affect(
    current: EmotionState,
    affect: ConversationalAffect,
    *,
    rule: Optional[AppraisalRule] = None,
    profile: Optional[EmotionProfile] = None,
    traits: Optional[Mapping[str, int]] = None,
    now: Optional[datetime] = None,
    config: Optional[EmotionDynamicsConfig] = None,
    dispositions: Optional[Mapping[str, float]] = None,
) -> AffectTransition:
    """Compose one conversational event/frame change with the current state.

    ``rule`` is an already-selected, locally validated appraisal rule. When it
    is ``None`` only a real frame change or an evidence-backed same-frame
    development can move affect. The function is pure aside from reading its
    arguments; callers persist the returned state.

    Since V0.4.3 the conversation-driven mechanisms are independent of the
    time-decay mode: they run in OFF, AUXILIARY and LEGACY alike when
    ``conversation_transition_enabled`` is set.
    """
    cfg = config or DEFAULT_CONFIG
    moment = now or datetime.now(timezone.utc)
    relevance = affect.scene_relevance()
    previous_frame = str(affect.previous_frame or "UNKNOWN").upper()
    frame = str(affect.interaction_frame or "UNKNOWN").upper()

    frame_confidence = float(getattr(affect, "confidence", 0.0))
    frame_spec = (
        _FRAME_STIMULI.get(frame)
        if affect.frame_changed
        and cfg.frame_transition_enabled
        and frame_confidence >= cfg.frame_transition_min_confidence
        else None
    )
    same_frame_spec = (
        _same_frame_stimulus(affect, cfg)
        if frame_spec is None and rule is None
        else None
    )

    prepared = current
    conversation_enabled = bool(
        getattr(cfg, "conversation_transition_enabled", True)
    )
    active_spec = frame_spec or same_frame_spec
    if (
        conversation_enabled
        and (rule is not None or active_spec is not None)
        and getattr(cfg, "scene_relevance_suppression", 0.0) > 0.0
    ):
        # Old, unrelated activations belong to the previous scene. They are
        # reduced before the new event is added; the event's own category is
        # kept so a repeated insult can still compound.
        keep = rule.emotion if rule is not None else active_spec[0]
        retention = 1.0 - (1.0 - relevance) * cfg.scene_relevance_suppression
        if affect.frame_changed:
            retention *= 1.0 - cfg.frame_change_suppression
        prepared = _scale_activations(
            prepared,
            retention,
            keep=keep,
            config=cfg,
            now=moment,
        )
        if rule is not None:
            impact = clamp(rule.intensity, 0.0, 1.0)
            dominant = _dominant(prepared)
            if (
                dominant not in (None, "NEUTRAL", rule.emotion)
                and impact >= cfg.ranking_override_min_impact
            ):
                # A clear new event must be able to become the expressed
                # emotion instead of being outvoted forever by a stale max.
                prepared = _scale_activations(
                    prepared,
                    1.0 - impact * cfg.ranking_override_strength,
                    keep=rule.emotion,
                    config=cfg,
                    now=moment,
                )

    if rule is not None:
        state = apply_emotional_stimulus(
            prepared,
            rule,
            profile,
            traits,
            moment,
            config=cfg,
            dispositions=dispositions,
        )
        cause = EmotionalTransitionCause(
            code=EVENT,
            reason=rule.name,
            previous_frame=previous_frame,
            frame=frame,
            event_type=affect.event_type,
            rule_name=rule.name,
            scene_relevance=relevance,
        )
        # A repair frame lowers tension even when an apology rule also fired.
        if affect.frame_changed and frame == "REPAIR":
            state = _adjust_tension(state, -cfg.tension_decrease * 0.5, moment)
        return AffectTransition(state=state, cause=cause, changed=True)

    spec = frame_spec or same_frame_spec
    if spec is not None:
        emotion, intensity, tension_delta = spec
        frame_rule = AppraisalRule(
            rule_id=f"frame_{frame.lower()}",
            agent_id="frame_transition",
            name=f"frame_{frame.lower()}",
            trigger_type="frame",
            pattern={},
            emotion=emotion,
            intensity=clamp(
                intensity
                * (cfg.frame_transition_scale if frame_spec is not None else 1.0),
                0.0,
                1.0,
            ),
            priority=0,
            status="ACTIVE",
            source="system",
            created_at=moment,
            updated_at=moment,
        )
        state = apply_emotional_stimulus(
            prepared,
            frame_rule,
            profile,
            traits,
            moment,
            config=cfg,
            dispositions=dispositions,
        )
        if tension_delta:
            state = _adjust_tension(state, tension_delta, moment)
        code = (
            SAME_FRAME_EVENT
            if frame_spec is None
            else REPAIR
            if frame == "REPAIR"
            else FRAME_CHANGE
        )
        cause = EmotionalTransitionCause(
            code=code,
            reason=(
                "same_frame_progress"
                if frame_spec is None
                else f"frame_{frame.lower()}"
            ),
            previous_frame=previous_frame,
            frame=frame,
            event_type=affect.event_type,
            rule_name=None,
            scene_relevance=relevance,
        )
        return AffectTransition(state=state, cause=cause, changed=True)

    return AffectTransition(
        state=current,
        cause=EmotionalTransitionCause(
            code=NONE,
            reason="no_conversational_change",
            previous_frame=previous_frame,
            frame=frame,
            event_type=affect.event_type,
            scene_relevance=relevance,
        ),
        changed=False,
    )


def _same_frame_stimulus(
    affect: ConversationalAffect, cfg: EmotionDynamicsConfig
) -> Optional[tuple[str, float, float]]:
    """A new development inside an unchanged frame, not mere repetition.

    Only a validated structural signal counts: a setup/payoff turn that refers
    back to a visible previous turn. Three PLAYFUL turns in a row add nothing
    unless the conversation actually develops (the joke lands, a clarification
    changes the reading).
    """
    if not bool(getattr(cfg, "frame_same_event_enabled", True)):
        return None
    frame = str(affect.interaction_frame or "UNKNOWN").upper()
    spec = _FRAME_STIMULI.get(frame)
    if spec is None:
        return None
    if str(affect.setup_payoff_role or "NONE").upper() not in ("SETUP", "PAYOFF"):
        return None
    if not affect.refers_to_previous_turn:
        return None
    try:
        confidence = float(affect.confidence or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    threshold = max(
        float(getattr(cfg, "frame_transition_min_confidence", 0.55)),
        float(getattr(cfg, "frame_same_event_min_confidence", 0.60)),
    )
    if confidence < threshold:
        return None
    scale = clamp(
        float(getattr(cfg, "frame_same_event_scale", 0.4)), 0.0, 1.0
    )
    emotion, intensity, tension_delta = spec
    return emotion, intensity * scale, tension_delta * scale


def _dominant(state: EmotionState) -> Optional[str]:
    activations = state.activations or {}
    if not activations:
        return None
    return max(activations, key=activations.get)


def _scale_activations(
    state: EmotionState,
    factor: float,
    *,
    keep: str,
    config: EmotionDynamicsConfig,
    now: datetime,
) -> EmotionState:
    factor = clamp(factor, 0.0, 1.0)
    if factor >= 1.0:
        return state
    activations = {
        key: float(value) * factor
        for key, value in (state.activations or {}).items()
        if key != keep and float(value) * factor > config.neutral_threshold
    }
    # Preserve the event's own prior activation unchanged; repeated same-event
    # stimuli should still compound.
    if keep in (state.activations or {}):
        activations[keep] = float(state.activations[keep])
    if not activations:
        return replace(
            state,
            activations={},
            emotion="NEUTRAL",
            valence=0.0,
            arousal=0.0,
            intensity=0.0,
            updated_at=now,
        )
    emotion, valence, arousal, intensity, clean = derive_acute(
        activations,
        cap=config.acute_intensity_cap,
        neutral_threshold=config.neutral_threshold,
    )
    return replace(
        state,
        activations=clean,
        emotion=emotion,
        valence=valence,
        arousal=arousal,
        intensity=intensity,
        updated_at=now,
    )


def _adjust_tension(
    state: EmotionState, delta: float, now: datetime
) -> EmotionState:
    current = state.relationship_tension
    if delta >= 0:
        tension = current + delta * (1.0 - current)
    else:
        tension = current * (1.0 + delta)
    return replace(
        state,
        relationship_tension=clamp(tension, 0.0, 1.0),
        tension_updated_at=now,
    )
