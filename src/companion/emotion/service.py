"""The emotion service: the single gateway for the persistent emotion framework.

It owns four layers and keeps them strictly separated:

- **Stable personality** — the per-agent :class:`EmotionProfile` (configured,
  never inferred from one turn).
- **Persistent appraisal rules** — versioned, audited, seeded with safe
  defaults; learned rules must be approved before they change the profile.
- **Temporary state** — the decaying :class:`EmotionState`, checkpointed lazily
  in bounded batches (no continuous DB-writing timer) and scoped per person so
  two people's interactions never mix private emotional context.
- **Emotion-aware response** — a :class:`SpeechStylePlan` that only shapes
  *how* a reply is delivered, never *whether* it is sent.

Important event/appraisal links are written as durable :class:`EmotionState`
events with ``source_event_id`` and (when applicable) ``memory_id``: the emotion
decays, but the cause and reason persist as audit trail. Decay never deletes an
event or a rule.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from companion.emotion.appraisal import AppraisalEngine
from companion.emotion.dynamics import decay
from companion.emotion.models import (
    AppraisalResult,
    AppraisalRule,
    EmotionProfile,
    EmotionState,
    SpeechStylePlan,
    VoiceProfile,
)
from companion.emotion.parameters import (
    DEFAULT_CONFIG,
    EmotionDynamicsConfig,
)
from companion.emotion.repository import EmotionRepository
from companion.emotion.speech import SpeechPlanner
from companion.storage.repositories import AuditRepository

DEFAULT_VOICE_NAME = "default"


class EmotionProfileConflict(RuntimeError):
    """Raised when a profile update loses an optimistic revision check."""


class EmotionService:
    def __init__(
        self,
        repo: EmotionRepository,
        *,
        audit: Optional[AuditRepository] = None,
        logger: Optional[logging.Logger] = None,
        clock=None,
        config: EmotionDynamicsConfig | None = None,
    ) -> None:
        self.repo = repo
        self.audit = audit
        self.logger = logger or logging.getLogger("companion.emotion")
        # ``clock`` is injectable for deterministic decay tests; default is UTC.
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.config = config or DEFAULT_CONFIG
        self.appraisal = AppraisalEngine(logger=logger, config=self.config)
        self.speech = SpeechPlanner(logger=logger)

    # -- profile ------------------------------------------------------------

    async def ensure_defaults(self, agent_id: str) -> EmotionProfile:
        """Seed a default profile and rules for an agent that lacks them.

        Concurrent-safe: the profile insert is ``ON CONFLICT DO NOTHING`` and
        each default rule is ``INSERT OR IGNORE`` against the unique
        ``(agent_id, name)`` index. Existing rows — including user-disabled
        rules — are never overwritten or re-enabled. Everything happens in one
        immediate transaction so two first turns cannot interleave their seeds.
        """
        now = self.clock()
        default = EmotionProfile(
            agent_id=agent_id,
            baseline_valence=0.0,
            baseline_arousal=0.0,
            default_half_life_seconds=int(self.config.acute_half_life_seconds),
            responsiveness=self.config.responsiveness,
            expression_style={},
            revision=1,
            created_at=now,
            updated_at=now,
        )
        async with self.repo.db.transaction(immediate=True):
            await self.repo.insert_profile_if_absent(default)
            await self.seed_default_rules(agent_id, now=now)
            profile = await self.repo.get_profile(agent_id)
        return profile or default

    async def seed_default_rules(
        self, agent_id: str, *, now: datetime | None = None
    ) -> None:
        """Insert any missing default rules without touching existing ones."""
        from companion.domain.models import new_id
        from companion.emotion.appraisal import DEFAULT_RULES

        moment = now or self.clock()
        for spec in DEFAULT_RULES:
            await self.repo.insert_rule(
                AppraisalRule(
                    rule_id=new_id("emrule"),
                    agent_id=agent_id,
                    name=spec["name"],
                    trigger_type=spec["trigger_type"],
                    pattern=spec["pattern"],
                    emotion=spec["emotion"],
                    valence_delta=spec["valence_delta"],
                    arousal_delta=spec["arousal_delta"],
                    intensity=spec["intensity"],
                    priority=spec["priority"],
                    status="ACTIVE",
                    source="system",
                    created_at=moment,
                    updated_at=moment,
                )
            )

    async def get_profile(self, agent_id: str) -> EmotionProfile | None:
        return await self.repo.get_profile(agent_id)

    async def update_profile(
        self,
        agent_id: str,
        *,
        baseline_valence: float | None = None,
        baseline_arousal: float | None = None,
        default_half_life_seconds: int | None = None,
        responsiveness: float | None = None,
        changed_by: str = "admin",
    ) -> EmotionProfile:
        """Update the stable profile, versioned and audited.

        The write is guarded by an optimistic revision check
        (``WHERE agent_id=? AND revision=?``): if another writer advanced the
        revision between our read and our write, this raises
        :class:`EmotionProfileConflict` instead of silently overwriting it.
        """
        await self.ensure_defaults(agent_id)
        profile = await self.repo.get_profile(agent_id)
        if profile is None:  # pragma: no cover - ensure_defaults creates it
            raise EmotionProfileConflict(f"profile for {agent_id!r} is missing")
        new_valence = (
            profile.baseline_valence
            if baseline_valence is None
            else max(-1.0, min(1.0, float(baseline_valence)))
        )
        new_arousal = (
            profile.baseline_arousal
            if baseline_arousal is None
            else max(0.0, min(1.0, float(baseline_arousal)))
        )
        new_half_life = (
            profile.default_half_life_seconds
            if default_half_life_seconds is None
            else max(1, int(default_half_life_seconds))
        )
        new_responsiveness = (
            profile.responsiveness
            if responsiveness is None
            else max(0.0, min(1.0, float(responsiveness)))
        )
        updated = profile.model_copy(
            update={
                "baseline_valence": new_valence,
                "baseline_arousal": new_arousal,
                "default_half_life_seconds": new_half_life,
                "responsiveness": new_responsiveness,
                "updated_at": self.clock(),
                "revision": profile.revision + 1,
            }
        )
        written = await self.repo.update_profile_optimistic(
            updated, expected_revision=profile.revision
        )
        if not written:
            raise EmotionProfileConflict(
                f"profile for {agent_id!r} changed concurrently"
                f" (expected revision {profile.revision})"
            )
        await self._audit(agent_id, "emotion_profile_update", agent_id,
                          f"revision={updated.revision}")
        self.logger.info(
            "emotion_profile_updated agent_id=%s revision=%d", agent_id, updated.revision
        )
        return updated

    # -- rules --------------------------------------------------------------

    async def list_rules(self, agent_id: str) -> list[AppraisalRule]:
        await self.ensure_defaults(agent_id)
        return await self.repo.list_rules(agent_id)

    async def set_rule_status(
        self, agent_id: str, rule_id: str, status: str, *, changed_by: str = "admin"
    ) -> bool:
        """Enable/disable an existing rule (audited). Never mutates its logic."""
        if status not in ("ACTIVE", "DISABLED"):
            raise ValueError(f"invalid rule status {status!r}")
        changed = await self.repo.update_rule_status(
            agent_id, rule_id, status, changed_by=changed_by
        )
        if changed:
            await self._audit(agent_id, "emotion_rule_status", rule_id,
                              f"status={status}")
        return changed

    async def propose_rule(
        self,
        agent_id: str,
        *,
        name: str,
        trigger_type: str,
        pattern: dict,
        emotion: str,
        valence_delta: float,
        arousal_delta: float,
        intensity: float,
        evidence_ref: str | None,
    ) -> AppraisalRule:
        """Record a learned rule as a *disabled proposal*.

        A reflection engine may suggest a rule, but it is never applied to the
        active profile without human approval. The proposal is stored DISABLED
        with its evidence reference so a human can review and enable it.
        """
        from companion.domain.models import new_id

        now = self.clock()
        rule = AppraisalRule(
            rule_id=new_id("emrule"),
            agent_id=agent_id,
            name=name,
            trigger_type=trigger_type,
            pattern=pattern,
            emotion=emotion,
            valence_delta=max(-2.0, min(2.0, float(valence_delta))),
            arousal_delta=max(-1.0, min(1.0, float(arousal_delta))),
            intensity=max(0.0, min(1.0, float(intensity))),
            priority=0,
            status="DISABLED",
            source="reflection",
            evidence_ref=evidence_ref,
            created_at=now,
            updated_at=now,
        )
        await self.repo.insert_rule(rule)
        await self._audit(agent_id, "emotion_rule_proposed", rule.rule_id,
                          f"source=reflection evidence={evidence_ref}")
        self.logger.info("emotion_rule_proposed agent_id=%s rule_id=%s",
                         agent_id, rule.rule_id)
        return rule

    # -- temporary state ----------------------------------------------------

    async def get_state(
        self, agent_id: str, person_id: str, *, persist_checkpoint: bool = False
    ) -> EmotionState:
        """Return the current (lazily decayed) emotion state for a person.

        The stored snapshot is read, advanced to ``now`` by the decay function,
        and optionally checkpointed back (bounded, no timer). If no snapshot
        exists the neutral baseline is returned.
        """
        await self.ensure_defaults(agent_id)
        profile = await self.repo.get_profile(agent_id)
        baseline_valence = profile.baseline_valence if profile else 0.0
        baseline_arousal = profile.baseline_arousal if profile else 0.0

        snap = await self.repo.get_snapshot(agent_id, person_id)
        state = self._state_from_snapshot(
            snap,
            profile,
            now=self.clock(),
            baseline_valence=baseline_valence,
            baseline_arousal=baseline_arousal,
        )
        state = decay(
            state,
            now=self.clock(),
            baseline_valence=baseline_valence,
            baseline_arousal=baseline_arousal,
            config=self.config,
        )
        if persist_checkpoint:
            await self._checkpoint(agent_id, person_id, state)
        return state

    async def process_turn(
        self,
        agent_id: str,
        person_id: str,
        *,
        text: str,
        signal=None,
        memories: list[dict] | None = None,
        agent_context: dict | None = None,
        conversation_id: str | None = None,
        source_event_id: str | None = None,
        persist: bool = True,
        addressee_hint: str | None = None,
        actor_person_id: str | None = None,
        conversational_affect=None,
        dispositions: Optional[dict[str, float]] = None,
    ) -> EmotionState:
        """Appraise and apply one turn atomically, idempotently, at one instant.

        The whole read-modify-write happens inside one ``BEGIN IMMEDIATE``
        transaction: the receipt is checked, profile and snapshot are read, the
        state is decayed at a single captured ``now``, the local appraisal runs,
        and then either the emotion event + snapshot + ``APPLIED`` receipt are
        written together, or only a ``NO_CHANGE`` receipt is written. A crash at
        any point leaves no half-applied turn. No external model is called here.

        ``persist=False`` performs the same computation with no writes at all.
        """
        memories = memories or []
        traits = None
        if isinstance(agent_context, dict):
            traits = agent_context.get("traits")
        await self.ensure_defaults(agent_id)
        now = self._now()

        if not persist:
            profile = await self.repo.get_profile(agent_id)
            snapshot = await self.repo.get_snapshot(agent_id, person_id)
            current = self._decayed_state(profile, snapshot, now)
            rules = await self.repo.list_rules(agent_id)
            result = self.appraisal.appraise(
                text,
                rules=rules,
                signal=signal,
                memories=memories,
                agent_context=agent_context,
                current_state=current,
                profile=profile,
                traits=traits,
                now=now,
                config=self.config,
                addressee_hint=addressee_hint,
                actor_person_id=actor_person_id,
                conversational_affect=conversational_affect,
                dispositions=dispositions,
            )
            return self._state_from_result(result, profile, now)

        async with self.repo.db.transaction(immediate=True):
            if source_event_id is not None:
                receipt = await self.repo.get_receipt(source_event_id)
                if receipt is None and await self.repo.has_event_for_source(
                    source_event_id
                ):
                    # A pre-V0.3.1 event without a receipt: adopt it rather than
                    # applying the same shift again.
                    await self.repo.insert_receipt(
                        source_event_id=source_event_id,
                        agent_id=agent_id,
                        person_id=person_id,
                        outcome="APPLIED",
                        processed_at=now,
                    )
                    receipt = await self.repo.get_receipt(source_event_id)
                if receipt is not None:
                    profile = await self.repo.get_profile(agent_id)
                    snapshot = await self.repo.get_snapshot(agent_id, person_id)
                    return self._decayed_state(profile, snapshot, now)

            profile = await self.repo.get_profile(agent_id)
            snapshot = await self.repo.get_snapshot(agent_id, person_id)
            current = self._decayed_state(profile, snapshot, now)
            rules = await self.repo.list_rules(agent_id)
            result = self.appraisal.appraise(
                text,
                rules=rules,
                signal=signal,
                memories=memories,
                agent_context=agent_context,
                current_state=current,
                profile=profile,
                traits=traits,
                now=now,
                config=self.config,
                addressee_hint=addressee_hint,
                actor_person_id=actor_person_id,
                conversational_affect=conversational_affect,
                dispositions=dispositions,
            )

            if not result.applied and not result.transitioned:
                if source_event_id is not None:
                    await self.repo.insert_receipt(
                        source_event_id=source_event_id,
                        agent_id=agent_id,
                        person_id=person_id,
                        outcome="NO_CHANGE",
                        processed_at=now,
                        reason=result.reason,
                        detail=result.diagnostic,
                    )
                self._log_evaluation(
                    result, agent_id=agent_id, person_id=person_id
                )
                return current

            state = self._state_from_result(result, profile, now)
            await self.repo.insert_event(
                agent_id=agent_id,
                conversation_id=conversation_id,
                person_id=person_id,
                source_event_id=source_event_id,
                appraisal_rule_id=result.trigger_rule_id,
                memory_id=result.matched_memory_id,
                emotion=state.emotion,
                valence=state.valence,
                arousal=state.arousal,
                intensity=state.intensity,
                appraisal_reason=result.reason,
                created_at=now,
                event_type=result.event_type,
                target=result.target,
                social_intent=result.social_intent,
                validation=result.validation,
                acute_before=result.acute_before,
                acute_after=result.acute_after,
                mood_after=result.mood_after,
                tension_after=result.tension_after,
            )
            await self._checkpoint(
                agent_id, person_id, state, updated_at=now
            )
            if source_event_id is not None:
                await self.repo.insert_receipt(
                    source_event_id=source_event_id,
                    agent_id=agent_id,
                    person_id=person_id,
                    outcome="APPLIED",
                    processed_at=now,
                    reason=result.reason,
                    detail=result.event_type,
                )
            self._log_evaluation(
                result, agent_id=agent_id, person_id=person_id
            )
            return state

    async def apply_result(
        self,
        agent_id: str,
        person_id: str,
        result: AppraisalResult,
        *,
        conversation_id: str | None = None,
        source_event_id: str | None = None,
        persist: bool = True,
    ) -> EmotionState:
        """Compatibility path for an already-computed :class:`AppraisalResult`.

        Uses the same single transaction and receipt semantics as
        :meth:`process_turn`, so callers cannot double-apply a result.
        """
        await self.ensure_defaults(agent_id)
        now = self._now()
        profile = await self.repo.get_profile(agent_id)
        half_life = profile.default_half_life_seconds if profile else 3600.0
        state = EmotionState(
            valence=result.valence,
            arousal=result.arousal,
            intensity=result.intensity,
            emotion=result.emotion,
            updated_at=now,
            half_life_seconds=float(half_life),
        )
        if not persist:
            return state

        async with self.repo.db.transaction(immediate=True):
            if source_event_id is not None:
                if await self.repo.get_receipt(source_event_id) is not None or (
                    await self.repo.has_event_for_source(source_event_id)
                ):
                    snapshot = await self.repo.get_snapshot(agent_id, person_id)
                    return self._decayed_state(profile, snapshot, now)

            if result.trigger_rule_id is not None:
                await self.repo.insert_event(
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    person_id=person_id,
                    source_event_id=source_event_id,
                    appraisal_rule_id=result.trigger_rule_id,
                    memory_id=result.matched_memory_id,
                    emotion=state.emotion,
                    valence=state.valence,
                    arousal=state.arousal,
                    intensity=state.intensity,
                    appraisal_reason=result.reason,
                    created_at=now,
                )
                await self._checkpoint(agent_id, person_id, state, updated_at=now)
            if source_event_id is not None:
                await self.repo.insert_receipt(
                    source_event_id=source_event_id,
                    agent_id=agent_id,
                    person_id=person_id,
                    outcome=(
                        "APPLIED"
                        if result.trigger_rule_id is not None
                        else "NO_CHANGE"
                    ),
                    processed_at=now,
                )
            return state

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now

    def _decayed_state(
        self,
        profile: EmotionProfile | None,
        snapshot: dict | None,
        now: datetime,
    ) -> EmotionState:
        baseline_valence = profile.baseline_valence if profile else 0.0
        baseline_arousal = profile.baseline_arousal if profile else 0.0
        state = self._state_from_snapshot(
            snapshot,
            profile,
            now=now,
            baseline_valence=baseline_valence,
            baseline_arousal=baseline_arousal,
        )
        return decay(
            state,
            now=now,
            baseline_valence=baseline_valence,
            baseline_arousal=baseline_arousal,
            config=self.config,
        )

    def _state_from_snapshot(
        self,
        snapshot: dict | None,
        profile: EmotionProfile | None,
        *,
        now: datetime,
        baseline_valence: float,
        baseline_arousal: float,
    ) -> EmotionState:
        """Load a snapshot, preserving legacy rows without inventing layers.

        A pre-schema-15 row has no activation map, mood or tension columns set;
        it is read as acute-only affect, and the legacy acute fields remain the
        source of truth. Nothing is back-filled or fabricated.
        """
        default_half_life = (
            float(profile.default_half_life_seconds)
            if profile
            else float(self.config.acute_half_life_seconds)
        )
        if snapshot is None:
            return EmotionState(
                valence=baseline_valence,
                arousal=baseline_arousal,
                intensity=0.0,
                emotion="NEUTRAL",
                updated_at=now,
                half_life_seconds=default_half_life,
                mood_half_life_seconds=self.config.mood_half_life_seconds,
                tension_half_life_seconds=self.config.tension_half_life_seconds,
            )
        activations = {}
        raw = snapshot.get("activations_json")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    activations = {
                        str(key): float(value) for key, value in parsed.items()
                    }
            except (json.JSONDecodeError, TypeError, ValueError):
                activations = {}
        return EmotionState(
            valence=snapshot["valence"],
            arousal=snapshot["arousal"],
            intensity=snapshot["intensity"],
            emotion=snapshot["emotion"],
            updated_at=datetime.fromisoformat(snapshot["updated_at"]),
            half_life_seconds=snapshot["half_life_seconds"],
            activations=activations,
            mood_valence=float(snapshot.get("mood_valence") or 0.0),
            mood_arousal=float(snapshot.get("mood_arousal") or 0.0),
            mood_intensity=float(snapshot.get("mood_intensity") or 0.0),
            mood_emotion=snapshot.get("mood_emotion") or "NEUTRAL",
            mood_updated_at=self._optional_dt(snapshot.get("mood_updated_at")),
            mood_half_life_seconds=float(
                snapshot.get("mood_half_life_seconds")
                or self.config.mood_half_life_seconds
            ),
            relationship_tension=float(
                snapshot.get("relationship_tension") or 0.0
            ),
            tension_updated_at=self._optional_dt(
                snapshot.get("tension_updated_at")
            ),
            tension_half_life_seconds=float(
                snapshot.get("tension_half_life_seconds")
                or self.config.tension_half_life_seconds
            ),
            last_event_type=snapshot.get("last_event_type"),
            last_event_at=self._optional_dt(snapshot.get("last_event_at")),
        )

    @staticmethod
    def _optional_dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    @staticmethod
    def _state_from_result(
        result: AppraisalResult,
        profile: EmotionProfile | None,
        now: datetime,
    ) -> EmotionState:
        if result.state is not None:
            return result.state
        return EmotionState(
            valence=result.valence,
            arousal=result.arousal,
            intensity=result.intensity,
            emotion=result.emotion,
            updated_at=now,
            half_life_seconds=(
                float(profile.default_half_life_seconds) if profile else 300.0
            ),
        )

    async def _checkpoint(
        self,
        agent_id: str,
        person_id: str,
        state: EmotionState,
        *,
        updated_at: datetime | None = None,
    ) -> None:
        moment = updated_at or state.updated_at or self.clock()
        await self.repo.upsert_snapshot(
            agent_id=agent_id,
            person_id=person_id,
            valence=state.valence,
            arousal=state.arousal,
            intensity=state.intensity,
            emotion=state.emotion,
            half_life_seconds=state.half_life_seconds,
            updated_at=moment,
            activations=state.activations,
            mood_valence=state.mood_valence,
            mood_arousal=state.mood_arousal,
            mood_intensity=state.mood_intensity,
            mood_emotion=state.mood_emotion,
            mood_half_life_seconds=state.mood_half_life_seconds,
            mood_updated_at=state.mood_updated_at or moment,
            relationship_tension=state.relationship_tension,
            tension_half_life_seconds=state.tension_half_life_seconds,
            tension_updated_at=state.tension_updated_at or moment,
            last_event_type=state.last_event_type,
            last_event_at=state.last_event_at,
        )

    def _log_evaluation(
        self,
        result: AppraisalResult,
        *,
        agent_id: str,
        person_id: str,
    ) -> None:
        """One structured, PII-free diagnostic line per evaluated turn."""
        self.logger.info(
            "emotion_evaluated agent_id=%s person_id=%s event_type=%s"
            " target=%s validation=%s rule=%s outcome=%s"
            " acute_before=%.4f acute_after=%.4f mood=%.4f tension=%.4f",
            agent_id,
            person_id,
            result.event_type or "none",
            result.target or "UNKNOWN",
            result.validation,
            result.reason or "none",
            result.diagnostic,
            result.acute_before,
            result.acute_after,
            result.mood_after,
            result.tension_after,
        )

    # -- speech plan --------------------------------------------------------

    async def plan_speech(
        self, agent_id: str, *, state: EmotionState, text: str = ""
    ) -> SpeechStylePlan:
        """Build the emotion-aware delivery plan (no real TTS)."""
        voice = None
        profiles = await self.repo.list_voice_profiles(agent_id)
        if profiles:
            voice = profiles[0]
        return self.speech.plan(state=state, voice_profile=voice, text=text)

    # -- feedback -----------------------------------------------------------

    async def record_feedback(
        self,
        agent_id: str,
        *,
        feedback_text: str,
        person_id: str | None = None,
        conversation_id: str | None = None,
        source_event_id: str | None = None,
        suggestion_type: str | None = None,
        suggestion_payload: dict | None = None,
    ) -> str:
        return await self.repo.insert_feedback(
            agent_id=agent_id,
            person_id=person_id,
            conversation_id=conversation_id,
            source_event_id=source_event_id,
            feedback_text=feedback_text,
            suggestion_type=suggestion_type,
            suggestion_payload=suggestion_payload or {},
        )

    async def list_feedback(self, agent_id: str) -> list[dict]:
        return await self.repo.list_feedback(agent_id)

    async def list_events(self, agent_id: str, limit: int = 100) -> list[dict]:
        return await self.repo.list_events(agent_id, limit=limit)

    async def list_snapshots(self, agent_id: str) -> list[dict]:
        return await self.repo.list_snapshots(agent_id)

    async def get_voice_profiles(self, agent_id: str) -> list[VoiceProfile]:
        await self.ensure_defaults(agent_id)
        return await self.repo.list_voice_profiles(agent_id)

    # -- helpers ------------------------------------------------------------

    async def _audit(
        self, agent_id: str, action: str, entity_id: str | None, detail: str | None = None
    ) -> None:
        if self.audit is None:
            return
        await self.audit.record(
            actor="admin",
            action=action,
            entity_type="emotion",
            entity_id=entity_id,
            detail=detail,
        )
