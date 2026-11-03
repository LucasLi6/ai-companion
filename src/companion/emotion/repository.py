"""Data access for the emotion tables.

Thin, parameterised SQL. No business rules live here — authorisation, decay
and audit belong to :class:`EmotionService`. All rows are keyed by ``agent_id``
so different companion identities (or a clone) never mix private emotional
history.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from companion.domain.models import new_id, utcnow
from companion.emotion.models import AppraisalRule, EmotionProfile, VoiceProfile
from companion.storage.database import Database


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _json(value: str | None) -> dict:
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


class EmotionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # -- profile ------------------------------------------------------------

    async def get_profile(self, agent_id: str) -> EmotionProfile | None:
        cur = await self.db.execute(
            "SELECT agent_id, baseline_valence, baseline_arousal,"
            " default_half_life_seconds, responsiveness, expression_style,"
            " revision, created_at, updated_at FROM emotion_profiles"
            " WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return EmotionProfile(
            agent_id=row["agent_id"],
            baseline_valence=row["baseline_valence"],
            baseline_arousal=row["baseline_arousal"],
            default_half_life_seconds=row["default_half_life_seconds"],
            responsiveness=row["responsiveness"],
            expression_style=_json(row["expression_style"]),
            revision=row["revision"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    async def upsert_profile(
        self,
        profile: EmotionProfile,
    ) -> None:
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO emotion_profiles(agent_id, baseline_valence,"
            " baseline_arousal, default_half_life_seconds, responsiveness,"
            " expression_style, created_at, updated_at, revision)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(agent_id) DO UPDATE SET"
            " baseline_valence = excluded.baseline_valence,"
            " baseline_arousal = excluded.baseline_arousal,"
            " default_half_life_seconds = excluded.default_half_life_seconds,"
            " responsiveness = excluded.responsiveness,"
            " expression_style = excluded.expression_style,"
            " updated_at = excluded.updated_at, revision = revision + 1",
            (
                profile.agent_id,
                profile.baseline_valence,
                profile.baseline_arousal,
                profile.default_half_life_seconds,
                profile.responsiveness,
                json.dumps(profile.expression_style, ensure_ascii=False),
                now,
                now,
                profile.revision,
            ),
        )

    async def insert_profile_if_absent(self, profile: EmotionProfile) -> bool:
        """Insert the default profile only if no row exists (concurrent-safe).

        Unlike :meth:`upsert_profile`, this never bumps ``revision``: seeding a
        missing profile is not a user edit. Returns True when it inserted.
        """
        now = _iso(utcnow())
        cur = await self.db.execute(
            "INSERT INTO emotion_profiles(agent_id, baseline_valence,"
            " baseline_arousal, default_half_life_seconds, responsiveness,"
            " expression_style, created_at, updated_at, revision)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(agent_id) DO NOTHING",
            (
                profile.agent_id,
                profile.baseline_valence,
                profile.baseline_arousal,
                profile.default_half_life_seconds,
                profile.responsiveness,
                json.dumps(profile.expression_style, ensure_ascii=False),
                now,
                now,
                profile.revision,
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    async def update_profile_optimistic(
        self, profile: EmotionProfile, expected_revision: int
    ) -> bool:
        """Update a profile with an explicit revision check.

        Returns False (no rows changed) if another writer advanced the revision
        first. The caller decides whether to retry or report a conflict.
        """
        now = _iso(utcnow())
        cur = await self.db.execute(
            "UPDATE emotion_profiles SET"
            " baseline_valence = ?, baseline_arousal = ?,"
            " default_half_life_seconds = ?, responsiveness = ?,"
            " expression_style = ?, updated_at = ?, revision = revision + 1"
            " WHERE agent_id = ? AND revision = ?",
            (
                profile.baseline_valence,
                profile.baseline_arousal,
                profile.default_half_life_seconds,
                profile.responsiveness,
                json.dumps(profile.expression_style, ensure_ascii=False),
                now,
                profile.agent_id,
                expected_revision,
            ),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    # -- appraisal rules ----------------------------------------------------

    async def list_rules(
        self, agent_id: str, *, status: str | None = None
    ) -> list[AppraisalRule]:
        sql = (
            "SELECT rule_id, agent_id, name, trigger_type, pattern, emotion,"
            " valence_delta, arousal_delta, intensity, priority, status, source,"
            " evidence_ref, created_at, updated_at FROM emotion_appraisal_rules"
            " WHERE agent_id = ?"
        )
        params: tuple = (agent_id,)
        if status is not None:
            sql += " AND status = ?"
            params = (agent_id, status)
        sql += " ORDER BY priority DESC, created_at ASC"
        cur = await self.db.execute(sql, params)
        return [self._to_rule(dict(r)) for r in await cur.fetchall()]

    async def get_rule(self, agent_id: str, rule_id: str) -> AppraisalRule | None:
        cur = await self.db.execute(
            "SELECT rule_id, agent_id, name, trigger_type, pattern, emotion,"
            " valence_delta, arousal_delta, intensity, priority, status, source,"
            " evidence_ref, created_at, updated_at FROM emotion_appraisal_rules"
            " WHERE agent_id = ? AND rule_id = ?",
            (agent_id, rule_id),
        )
        row = await cur.fetchone()
        return self._to_rule(dict(row)) if row is not None else None

    async def insert_rule(self, rule: AppraisalRule) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO emotion_appraisal_rules(rule_id, agent_id, name,"
            " trigger_type, pattern, emotion, valence_delta, arousal_delta,"
            " intensity, priority, status, source, evidence_ref, created_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rule.rule_id,
                rule.agent_id,
                rule.name,
                rule.trigger_type,
                json.dumps(rule.pattern, ensure_ascii=False, sort_keys=True),
                rule.emotion,
                rule.valence_delta,
                rule.arousal_delta,
                rule.intensity,
                rule.priority,
                rule.status,
                rule.source,
                rule.evidence_ref,
                _iso(rule.created_at),
                _iso(rule.updated_at),
            ),
        )

    async def update_rule_status(
        self, agent_id: str, rule_id: str, status: str, *, changed_by: str
    ) -> bool:
        cur = await self.db.execute(
            "UPDATE emotion_appraisal_rules SET status = ?, updated_at = ?"
            " WHERE agent_id = ? AND rule_id = ?",
            (status, _iso(utcnow()), agent_id, rule_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    async def count_rules(self, agent_id: str) -> int:
        cur = await self.db.execute(
            "SELECT COUNT(*) AS c FROM emotion_appraisal_rules WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cur.fetchone()
        return int(row["c"])

    # -- events -------------------------------------------------------------

    async def insert_event(
        self,
        *,
        agent_id: str,
        conversation_id: str | None,
        person_id: str | None,
        source_event_id: str | None,
        appraisal_rule_id: str | None,
        memory_id: str | None,
        emotion: str,
        valence: float,
        arousal: float,
        intensity: float,
        appraisal_reason: str | None,
        created_at: datetime | None = None,
    ) -> str:
        event_id = new_id("emoev")
        await self.db.execute(
            "INSERT OR IGNORE INTO emotion_events(emotion_event_id, agent_id,"
            " conversation_id, person_id, source_event_id, appraisal_rule_id,"
            " memory_id, emotion, valence, arousal, intensity, appraisal_reason,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                agent_id,
                conversation_id,
                person_id,
                source_event_id,
                appraisal_rule_id,
                memory_id,
                emotion,
                valence,
                arousal,
                intensity,
                appraisal_reason,
                _iso(created_at or utcnow()),
            ),
        )
        return event_id

    async def has_event_for_source(self, source_event_id: str) -> bool:
        """Whether an emotion event already exists for a source utterance.

        This is the idempotency anchor: re-processing a turn must not apply the
        same emotional update twice. Enforced both here and by the unique index.
        """
        cur = await self.db.execute(
            "SELECT 1 FROM emotion_events WHERE source_event_id = ?",
            (source_event_id,),
        )
        return await cur.fetchone() is not None

    async def get_event_for_source(self, source_event_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM emotion_events WHERE source_event_id = ?",
            (source_event_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_events(
        self, agent_id: str, *, limit: int = 100
    ) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM emotion_events WHERE agent_id = ?"
            " ORDER BY created_at DESC LIMIT ?",
            (agent_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]

    # -- state snapshots ----------------------------------------------------

    async def get_snapshot(self, agent_id: str, person_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT agent_id, person_id, valence, arousal, intensity, emotion,"
            " half_life_seconds, updated_at FROM emotion_state_snapshots"
            " WHERE agent_id = ? AND person_id = ?",
            (agent_id, person_id),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def upsert_snapshot(
        self,
        *,
        agent_id: str,
        person_id: str,
        valence: float,
        arousal: float,
        intensity: float,
        emotion: str,
        half_life_seconds: float,
        updated_at: datetime | None = None,
    ) -> None:
        now = _iso(updated_at or utcnow())
        await self.db.execute(
            "INSERT INTO emotion_state_snapshots(agent_id, person_id, valence,"
            " arousal, intensity, emotion, half_life_seconds, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(agent_id, person_id) DO UPDATE SET"
            " valence = excluded.valence, arousal = excluded.arousal,"
            " intensity = excluded.intensity, emotion = excluded.emotion,"
            " half_life_seconds = excluded.half_life_seconds,"
            " updated_at = excluded.updated_at",
            (
                agent_id,
                person_id,
                valence,
                arousal,
                intensity,
                emotion,
                half_life_seconds,
                now,
            ),
        )

    async def list_snapshots(self, agent_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM emotion_state_snapshots WHERE agent_id = ?",
            (agent_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    # -- turn receipts (idempotency) ----------------------------------------

    async def get_receipt(self, source_event_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT source_event_id, agent_id, person_id, outcome, processed_at"
            " FROM emotion_turn_receipts WHERE source_event_id = ?",
            (source_event_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def insert_receipt(
        self,
        *,
        source_event_id: str,
        agent_id: str,
        person_id: str,
        outcome: str,
        processed_at: datetime | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO emotion_turn_receipts(source_event_id, agent_id,"
            " person_id, outcome, processed_at) VALUES (?, ?, ?, ?, ?)",
            (
                source_event_id,
                agent_id,
                person_id,
                outcome,
                _iso(processed_at or utcnow()),
            ),
        )

    # -- feedback / audit ---------------------------------------------------

    async def insert_feedback(
        self,
        *,
        agent_id: str,
        person_id: str | None,
        conversation_id: str | None,
        source_event_id: str | None,
        feedback_text: str,
        suggestion_type: str | None,
        suggestion_payload: dict,
    ) -> str:
        feedback_id = new_id("emofb")
        await self.db.execute(
            "INSERT INTO emotion_feedback(feedback_id, agent_id, person_id,"
            " conversation_id, source_event_id, feedback_text, suggestion_type,"
            " suggestion_payload, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)",
            (
                feedback_id,
                agent_id,
                person_id,
                conversation_id,
                source_event_id,
                feedback_text,
                suggestion_type,
                json.dumps(suggestion_payload, ensure_ascii=False, sort_keys=True),
                _iso(utcnow()),
            ),
        )
        return feedback_id

    async def list_feedback(self, agent_id: str, *, limit: int = 100) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM emotion_feedback WHERE agent_id = ?"
            " ORDER BY created_at DESC LIMIT ?",
            (agent_id, limit),
        )
        out = []
        for r in await cur.fetchall():
            item = dict(r)
            item["suggestion_payload"] = _json(item.get("suggestion_payload"))
            out.append(item)
        return out

    async def set_feedback_status(self, feedback_id: str, status: str) -> bool:
        cur = await self.db.execute(
            "UPDATE emotion_feedback SET status = ? WHERE feedback_id = ?",
            (status, feedback_id),
        )
        return (cur.rowcount if hasattr(cur, "rowcount") else 0) == 1

    # -- voice profiles -----------------------------------------------------

    async def list_voice_profiles(self, agent_id: str) -> list[VoiceProfile]:
        cur = await self.db.execute(
            "SELECT voice_profile_id, agent_id, name, provider, voice_id,"
            " language, speaking_rate, pitch, extra, created_at, updated_at"
            " FROM voice_profiles WHERE agent_id = ? ORDER BY created_at ASC",
            (agent_id,),
        )
        return [self._to_voice(dict(r)) for r in await cur.fetchall()]

    async def get_voice_profile(self, voice_profile_id: str) -> VoiceProfile | None:
        cur = await self.db.execute(
            "SELECT voice_profile_id, agent_id, name, provider, voice_id,"
            " language, speaking_rate, pitch, extra, created_at, updated_at"
            " FROM voice_profiles WHERE voice_profile_id = ?",
            (voice_profile_id,),
        )
        row = await cur.fetchone()
        return self._to_voice(dict(row)) if row is not None else None

    async def upsert_voice_profile(self, profile: VoiceProfile) -> None:
        now = _iso(utcnow())
        await self.db.execute(
            "INSERT INTO voice_profiles(voice_profile_id, agent_id, name, provider,"
            " voice_id, language, speaking_rate, pitch, extra, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(voice_profile_id) DO UPDATE SET"
            " name = excluded.name, provider = excluded.provider,"
            " voice_id = excluded.voice_id, language = excluded.language,"
            " speaking_rate = excluded.speaking_rate, pitch = excluded.pitch,"
            " extra = excluded.extra, updated_at = excluded.updated_at",
            (
                profile.voice_profile_id,
                profile.agent_id,
                profile.name,
                profile.provider,
                profile.voice_id,
                profile.language,
                profile.speaking_rate,
                profile.pitch,
                json.dumps(profile.extra, ensure_ascii=False),
                now,
                now,
            ),
        )

    async def count_voice_profiles(self, agent_id: str) -> int:
        cur = await self.db.execute(
            "SELECT COUNT(*) AS c FROM voice_profiles WHERE agent_id = ?",
            (agent_id,),
        )
        row = await cur.fetchone()
        return int(row["c"])

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _to_rule(row: dict) -> AppraisalRule:
        return AppraisalRule(
            rule_id=row["rule_id"],
            agent_id=row["agent_id"],
            name=row["name"],
            trigger_type=row["trigger_type"],
            pattern=_json(row.get("pattern")),
            emotion=row["emotion"],
            valence_delta=row["valence_delta"],
            arousal_delta=row["arousal_delta"],
            intensity=row["intensity"],
            priority=row["priority"],
            status=row["status"],
            source=row["source"],
            evidence_ref=row.get("evidence_ref"),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    @staticmethod
    def _to_voice(row: dict) -> VoiceProfile:
        return VoiceProfile(
            voice_profile_id=row["voice_profile_id"],
            agent_id=row["agent_id"],
            name=row["name"],
            provider=row["provider"],
            voice_id=row.get("voice_id"),
            language=row["language"],
            speaking_rate=row["speaking_rate"],
            pitch=row.get("pitch"),
            extra=_json(row.get("extra")),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )
