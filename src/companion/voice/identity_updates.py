"""Display-only planning for identity updates that arrive after a turn.

The speaker worker may publish a better identity revision after the ASR final
has already been processed. Such an update may *annotate* the turn; it must
never re-submit the transcript, call the Router again, regenerate speech,
rewrite the submitted ``UtteranceEvent`` or append memory.

This module is intentionally pure: it reads an attribution-ledger entry and a
resolved identity and returns the single action the surface should take. The
caller owns the entry mutations and the terminal write, which keeps the rules
(no silent person switch, no downgrade of a submitted turn, explicit conflict)
testable without a running REPL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

CONFLICT_SUBMITTED = "conflict_submitted"
CONFLICT_PENDING = "conflict_pending"
CONFIRMED = "confirmed"
RELABELED = "relabeled"
TENTATIVE = "tentative"

CONFLICT_SUBMITTED_MESSAGE = (
    "[id] 身份冲突：同一轮语音后来解析为另一人物；"
    "本条已按提交时确认的人物处理，未改写，需人工检查（/speaker status）。\n"
)
CONFLICT_PENDING_MESSAGE = (
    "[id] 身份冲突：同一轮语音先后解析出不同人物；"
    "本条未提交，需人工确认（/speaker status）。\n"
)
CONFIRMED_MESSAGE = (
    "[id] 本轮声音已识别为 {name}（仅更新标注，不会重复回复）\n"
)
RELABELED_MESSAGE = (
    "[id] 声纹标注更新：speaker {speaker}（本条已提交，不重复回复）\n"
)
TENTATIVE_MESSAGE = (
    "[id] 声纹标注更新：speaker {speaker}"
    "（仍无已确认人物，不会重复回复）\n"
)


@dataclass(frozen=True)
class LateIdentityPlan:
    """One display-only action for an already-decided turn."""

    kind: str
    message: str
    state: Optional[str] = None
    speaker_id: Optional[str] = None
    person_id: Optional[str] = None
    display_name: Optional[str] = None


def plan_late_identity(
    entry: Mapping[str, Any], identity: Any
) -> Optional[LateIdentityPlan]:
    """Decide what a late identity revision may change, if anything.

    ``entry`` keys are the attribution ledger's: ``state`` (the displayed
    identity state), ``submitted``, ``identity_conflict``,
    ``last_displayed_state``, ``last_displayed_speaker_id`` and
    ``last_displayed_person_id``.
    """
    state = getattr(identity, "state", None)
    state_value = getattr(state, "value", state)
    person_id = getattr(identity, "person_id", None)
    display_name = getattr(identity, "display_name", None)
    label = (
        getattr(identity, "canonical_speaker_id", None)
        or getattr(identity, "raw_speaker_id", None)
    )
    previous_state = entry.get("last_displayed_state")
    previous_state_value = getattr(previous_state, "value", previous_state)
    previous_speaker = entry.get("last_displayed_speaker_id")
    previous_person = entry.get("last_displayed_person_id")
    # "submitted" means the turn was handed to the Router. A late CONFIRMED
    # state on a refused turn is only an annotation, so it never implies one.
    submitted = bool(entry.get("submitted"))

    if entry.get("identity_conflict"):
        # Once two trusted identities disagreed, only a human can resolve it.
        return None

    if submitted:
        if (
            state_value == "CONFIRMED"
            and person_id
            and previous_person
            and person_id != previous_person
        ):
            return LateIdentityPlan(
                CONFLICT_SUBMITTED, CONFLICT_SUBMITTED_MESSAGE
            )
        # A submitted turn is never downgraded or re-attributed. Only the
        # voice-cluster label may be refined.
        if (
            state_value == "CONFIRMED"
            and label
            and label != previous_speaker
        ):
            return LateIdentityPlan(
                RELABELED,
                RELABELED_MESSAGE.format(speaker=label),
                speaker_id=label,
            )
        return None

    if state_value == "CONFIRMED":
        if (
            previous_person
            and person_id
            and person_id != previous_person
        ):
            return LateIdentityPlan(
                CONFLICT_PENDING, CONFLICT_PENDING_MESSAGE
            )
        if (
            previous_state_value == "CONFIRMED"
            and previous_person == person_id
        ):
            if label and label != previous_speaker:
                return LateIdentityPlan(
                    RELABELED,
                    RELABELED_MESSAGE.format(speaker=label),
                    speaker_id=label,
                )
            return None
        return LateIdentityPlan(
            CONFIRMED,
            CONFIRMED_MESSAGE.format(name=display_name or person_id or "-"),
            state="CONFIRMED",
            speaker_id=label,
            person_id=person_id,
            display_name=display_name,
        )

    if state_value == "TENTATIVE" and label and label != previous_speaker:
        # Successive upgrades are shown as they arrive
        # (unknown -> speaker TEMP-xxxx -> speaker S-xxxx).
        return LateIdentityPlan(
            TENTATIVE,
            TENTATIVE_MESSAGE.format(speaker=label),
            state="TENTATIVE",
            speaker_id=label,
        )

    # A later UNKNOWN verdict never erases a displayed tentative label.
    return None


__all__ = [
    "CONFIRMED",
    "CONFLICT_PENDING",
    "CONFLICT_PENDING_MESSAGE",
    "CONFLICT_SUBMITTED",
    "CONFLICT_SUBMITTED_MESSAGE",
    "LateIdentityPlan",
    "RELABELED",
    "TENTATIVE",
    "plan_late_identity",
]
