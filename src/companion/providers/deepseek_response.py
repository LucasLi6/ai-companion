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
from companion.prompts.layout import RESPONSE_LAYOUT_CHAT, response_layout_for
from companion.prompts.response import (
    build_proactive_response_messages,
    build_response_messages,
)

PROVIDER_NAME = "deepseek"


class ResponseText(str):
    """Reply text that also carries the provider's terminal stop reason.

    A ``str`` subclass keeps the ``ResponseProvider`` protocol unchanged (the
    orchestrator still receives text), while letting the blocking path report
    ``finish_reason == "length"`` as ``TRUNCATED`` instead of claiming the
    reply is complete. Callers may simply use it as a string.
    """

    finish_reason: Optional[str] = None

    def __new__(cls, value: str, finish_reason: Optional[str] = None):
        text = super().__new__(cls, value)
        text.finish_reason = finish_reason
        return text


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
        # Diagnostic metadata for the proactive path only; never prompt data.
        self.last_model = ""
        # Off by default. When enabled, ordinary conversational replies may
        # carry the optional voice-expression labels; structured output tasks
        # never receive the protocol.
        self.speech_expression_enabled = False

    def _speech_expression_for(self, event: UtteranceEvent) -> bool:
        if not self.speech_expression_enabled:
            return False
        return response_layout_for(event.text) == RESPONSE_LAYOUT_CHAT

    async def generate(
        self,
        event: UtteranceEvent,
        analysis: CognitiveResult,
        context,
    ) -> str:
        await self._authorize(event)
        ctx = self._as_context(event, context, analysis)
        messages = build_response_messages(
            ctx, analysis, speech_expression=self._speech_expression_for(event)
        )
        response = await self.client.complete(
            messages,
            request_type="response",
            event_id=event.event_id,
            json_mode=False,
        )
        return ResponseText(
            response.content.strip(), getattr(response, "finish_reason", None)
        )

    async def generate_proactive(
        self,
        event: UtteranceEvent,
        analysis: CognitiveResult,
        context,
        *,
        mode: str = "IMMEDIATE",
    ) -> str:
        """Generate one Ava-initiated utterance with no new user message.

        The caller has already decided *whether* to speak. This path reuses
        the same persona/history/memory/scene context and the same external-AI
        authorization as a normal reply; only the generation-mode block and
        the absence of a current user turn differ.

        ``mode`` is ``IMMEDIATE`` or ``LONG_IDLE`` and only changes the short
        trusted generation note.
        """
        await self._authorize(event)
        ctx = self._as_context(event, context, analysis)
        messages = build_proactive_response_messages(
            ctx,
            analysis,
            mode=mode,
            speech_expression=self.speech_expression_enabled,
        )
        response = await self.client.complete(
            messages,
            request_type="response",
            event_id=event.event_id,
            json_mode=False,
        )
        self.last_model = getattr(response, "model", "") or self.last_model
        return ResponseText(
            response.content.strip(), getattr(response, "finish_reason", None)
        )

    async def generate_stream(
        self,
        event: UtteranceEvent,
        analysis: CognitiveResult,
        context,
    ):
        """Yield real deltas as the provider produces them.

        Each item is a :class:`~companion.providers.deepseek_client.StreamDelta`.
        The caller decides how to display and persist it; this method never
        assembles the whole reply before yielding.
        """
        await self._authorize(event)
        ctx = self._as_context(event, context, analysis)
        messages = build_response_messages(
            ctx, analysis, speech_expression=self._speech_expression_for(event)
        )
        async for delta in self.client.stream_completion(
            messages,
            request_type="response",
            event_id=event.event_id,
            json_mode=False,
        ):
            yield delta

    async def _authorize(self, event: UtteranceEvent) -> None:
        if self.privacy_gateway is None:
            return
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
