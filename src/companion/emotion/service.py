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
from companion.emotion.repository import EmotionRepository
from companion.emotion.speech import SpeechPlanner
from companion.storage.repositories import AuditRepository

DEFAULT_VOICE_NAME = "default"


class EmotionService:
    def __init__(
        self,
        repo: EmotionRepository,
        *,
        audit: Optional[AuditRepository] = None,
        logger: Optional[logging.Logger] = None,
        clock=None,
    ) -> None:
        self.repo = repo
        self.audit = audit
        self.logger = logger or logging.getLogger("companion.emotion")
        # ``clock`` is injectable for deterministic decay tests; default is UTC.
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.appraisal = AppraisalEngine(logger=logger)
        self.speech = SpeechPlanner(logger=logger)

    # -- profile ------------------------------------------------------------

    async def ensure_defaults(self, agent_id: str) -> EmotionProfile:
        """Seed a default profile and rules for an agent that lacks them."""
        profile = await self.repo.get_profile(agent_id)
        if profile is None:
            now = self.clock()
            profile = EmotionProfile(
                agent_id=agent_id,
                baseline_valence=0.0,
                baseline_arousal=0.0,
                default_half_life_seconds=3600,
                responsiveness=0.5,
                expression_style={},
                revision=1,
                created_at=now,
                updated_at=now,
            )
            await self.repo.upsert_profile(profile)
        if await self.repo.count_rules(agent_id) == 0:
            await self.seed_default_rules(agent_id)
        return profile

    async def seed_default_rules(self, agent_id: str) -> None:
        from companion.domain.models import new_id
        from companion.emotion.appraisal import DEFAULT_RULES

        now = self.clock()
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
                    created_at=now,
                    updated_at=now,
                )
            )
        self.logger.info("emotion_defaults_seeded agent_id=%s rules=%d",
                         agent_id, len(DEFAULT_RULES))

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
        """Update the stable profile, versioned and audited."""
        profile = await self.repo.get_profile(agent_id) or EmotionProfile(
            agent_id=agent_id,
            created_at=self.clock(),
            updated_at=self.clock(),
            revision=0,
        )
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
        await self.repo.upsert_profile(updated)
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
        state = EmotionState(
            valence=baseline_valence,
            arousal=baseline_arousal,
            intensity=0.0,
            emotion="NEUTRAL",
            updated_at=self.clock(),
            half_life_seconds=profile.default_half_life_seconds if profile else 3600.0,
        )
        if snap is not None:
            state = EmotionState(
                valence=snap["valence"],
                arousal=snap["arousal"],
                intensity=snap["intensity"],
                emotion=snap["emotion"],
                updated_at=datetime.fromisoformat(snap["updated_at"]),
                half_life_seconds=snap["half_life_seconds"],
            )
        state = decay(
            state,
            now=self.clock(),
            baseline_valence=baseline_valence,
            baseline_arousal=baseline_arousal,
        )
        if persist_checkpoint:
            await self._checkpoint(agent_id, person_id, state)
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
        """Apply an appraisal result: persist the event + snapshot, return state.

        ``persist=False`` computes the state in memory without writing events or
        snapshots (strict no-persistence); the caller is responsible for not
        claiming anything was saved.
        """
        await self.ensure_defaults(agent_id)

        # Idempotency anchor: re-processing a turn must not apply the same
        # emotional shift twice. If a previous (possibly crashed) attempt already
        # recorded an event for this source utterance, return the current decayed
        # state instead of re-adding the appraisal delta.
        if persist and source_event_id is not None:
            if await self.repo.has_event_for_source(source_event_id):
                return await self.get_state(agent_id, person_id)

        profile = await self.repo.get_profile(agent_id)
        half_life = profile.default_half_life_seconds if profile else 3600.0

        state = EmotionState(
            valence=result.valence,
            arousal=result.arousal,
            intensity=result.intensity,
            emotion=result.emotion,
            updated_at=self.clock(),
            half_life_seconds=float(half_life),
        )

        if persist:
            # Only a real appraisal (a rule actually fired) is recorded as a
            # durable emotion event. A turn where nothing matched carries no
            # emotional shift, so writing an event for it would only bloat the
            # history; the decayed state is still checkpointed below.
            if result.trigger_rule_id is not None:
                await self.repo.insert_event(
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    person_id=person_id,
                    source_event_id=source_event_id,
                    appraisal_rule_id=result.trigger_rule_id,
                    memory_id=result.matched_memory_id,
                    emotion=result.emotion,
                    valence=result.valence,
                    arousal=result.arousal,
                    intensity=result.intensity,
                    appraisal_reason=result.reason,
                )
            await self._checkpoint(agent_id, person_id, state)
        return state

    async def _checkpoint(self, agent_id: str, person_id: str, state: EmotionState) -> None:
        await self.repo.upsert_snapshot(
            agent_id=agent_id,
            person_id=person_id,
            valence=state.valence,
            arousal=state.arousal,
            intensity=state.intensity,
            emotion=state.emotion,
            half_life_seconds=state.half_life_seconds,
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
