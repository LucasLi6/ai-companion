"""Real DeepSeek response provider (Phase 6).

Called only when the policy finalises ``RESPOND``. It reuses the prepared
:class:`CognitiveContext` (the same history + memories the cognitive provider
saw), so the model never sees a second, differently-filtered view of the
conversation. The external-AI gate is re-checked before the network call as a
defence-in-depth measure.
"""

from __future__ import annotations

import logging
from typing import Optional

from companion.cognition.models import CognitiveContext
from companion.domain.models import CognitiveResult, UtteranceEvent
from companion.prompts.response import build_response_messages

PROVIDER_NAME = "deepseek"


class ResponsePermissionError(Exception):
    """Raised if a response is requested for a speaker with no external-AI grant."""


class DeepSeekResponseProvider:
    """``ResponseProvider`` backed by the real DeepSeek client."""

    name = PROVIDER_NAME

    def __init__(
        self,
        client,
        *,
        privacy_gateway=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = client
        self.privacy_gateway = privacy_gateway
        self.logger = logger or logging.getLogger("companion.deepseek.response")

    async def generate(
        self,
        event: UtteranceEvent,
        analysis: CognitiveResult,
        context,
    ) -> str:
        if self.privacy_gateway is not None:
            auth = await self.privacy_gateway.authorize_event(event.speaker_person_id)
            if not auth.allowed:
                self.logger.warning(
                    "deepseek_response_denied event_id=%s reason=%s",
                    event.event_id,
                    auth.reason,
                )
                raise ResponsePermissionError(
                    f"external AI not permitted: {auth.reason}"
                )

        ctx = self._as_context(event, context, analysis)
        messages = build_response_messages(ctx, analysis)
        response = await self.client.complete(
            messages,
            request_type="response",
            event_id=event.event_id,
            json_mode=False,
        )
        return response.content.strip()

    @staticmethod
    def _as_context(
        event: UtteranceEvent, context, analysis: CognitiveResult
    ) -> CognitiveContext:
        if isinstance(context, CognitiveContext):
            return context
        return CognitiveContext(
            conversation_id=event.conversation_id,
            current_event_id=event.event_id,
            current_speaker_id=event.speaker_person_id,
            current_speaker_label=event.speaker_label
            or event.speaker_person_id
            or "unknown",
            current_text=event.text,
            current_addressee_hint=event.addressee_hint.value,
            quiet_mode=getattr(context, "quiet_mode", False),
            turns=[],
            memories=[],
        )
