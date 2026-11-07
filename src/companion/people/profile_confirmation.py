"""Direct, trusted handling of explicit first-person profile statements.

V0.9.9.6 removed the candidate/confirmation queue. When a person says
"我叫李约" or "以后叫我 Lucas", that explicit statement is itself the
authorization. This service detects it deterministically and hands it to
``PersonProfileService.apply_direct_self_report``, which re-reads the original
event before writing. Nothing is asked, timed, or parked for later.

It never interprets a reply as confirmation of an earlier model suggestion: a
self-report is written only from the person's own current words. Silence,
timeouts and model inference are not evidence and produce no write.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.people.profile_self_report import detect_self_reports
from companion.people.profile_service import PersonProfileService

_LOGGER = logging.getLogger("companion.profile_confirmation")


class ProfileDirectSelfReportService:
    """Write explicit self-reports immediately; ask no follow-up question."""

    def __init__(
        self,
        *,
        profile_service: PersonProfileService,
        repos=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.profile_service = profile_service
        self.repos = repos
        self.logger = logger or _LOGGER

    async def begin_turn(self, *, event, person_id: str):
        """Detect and directly apply explicit self-reports in this turn.

        Returns ``None`` by design: there is no confirmation question to add to
        the reply. Failures are logged and never affect the conversation.
        """
        text = str(getattr(event, "text", "") or "")
        if not text:
            return None
        for report in detect_self_reports(text):
            try:
                result = await self.profile_service.apply_direct_self_report(
                    person_id=person_id,
                    field_key=report.field_key,
                    value=report.value,
                    source_event_id=event.event_id,
                    quote=report.quote,
                    actor="profile_confirmation",
                )
                if result.status == "REJECTED":
                    self.logger.info(
                        "profile_self_report_rejected person_id=%s field=%s reason=%s",
                        person_id,
                        report.field_key,
                        result.reason,
                    )
            except Exception as exc:  # pragma: no cover - safety net
                self.logger.warning(
                    "profile_self_report_failed person_id=%s field=%s error=%s",
                    person_id,
                    report.field_key,
                    type(exc).__name__,
                )
        return None

    async def mark_question_delivered(self, **_: object) -> None:
        """Compatibility no-op: no confirmation question exists any more."""
        return None

    async def resume_pending(self) -> int:
        """Compatibility no-op: there is no persisted question to resume."""
        return 0

    async def shutdown(self) -> None:
        """Compatibility no-op: the service owns no timers or tasks."""
        return None

    async def on_turn_completed(self, **_: object) -> None:
        return None


# Import compatibility for code and tests written before V0.9.9.6.
ProfileConfirmationService = ProfileDirectSelfReportService

__all__ = [
    "ProfileConfirmationService",
    "ProfileDirectSelfReportService",
]
