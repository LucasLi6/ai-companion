"""V0.7 selective response: addressee and reply necessity are separate.

All decisions are local and offline. No test constructs a real provider client.
"""

from __future__ import annotations

import pytest

from companion.cognition.nonresponse import derive_nonresponse_trace
from companion.cognition.context_builder import ContextBuilder
from companion.core.fast_path import assess_turn, looks_like_wait_request
from companion.core.policy import Policy
from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    ReasonCode,
    ReplyNecessity,
)
from companion.domain.models import ConversationContext, make_utterance_event
from companion.people.profile_models import ProfileFieldKey


def assess(text, *, hint="agent", silence=False, quiet=False, remember=False, **kw):
    return assess_turn(
        addressee_hint=hint,
        text=text,
        explicit_silence=silence,
        quiet_mode=quiet,
        explicit_remember=remember,
        **kw,
    )


def test_direct_question_requires_a_response():
    result = assess("今天天气怎么样？")
    assert result.action == CognitiveAction.RESPOND
    assert result.reply_necessity == ReplyNecessity.REQUIRED
    assert result.addressee == "agent"
    assert not result.needs_model


def test_ordinary_direct_address_is_useful_not_required():
    result = assess("今天挺热的。")
    assert result.action == CognitiveAction.RESPOND
    assert result.reply_necessity == ReplyNecessity.USEFUL


def test_confirmed_human_addressee_is_ignored():
    result = assess("我昨天完成了一个项目。", hint="human")
    assert result.action == CognitiveAction.IGNORE
    assert result.reason == ReasonCode.DIRECT_HUMAN_ADDRESS
    assert result.reply_necessity == ReplyNecessity.UNNECESSARY


def test_ambiguous_multi_speaker_uses_deliberation():
    result = assess("这个我觉得可以。", hint="unknown")
    assert result.needs_model
    assert result.action == CognitiveAction.WAIT
    assert result.reason == ReasonCode.UNKNOWN_ADDRESSEE
    # A conservative default, not a fake local RESPOND.
    assert result.addressee == "unknown"


def test_explicit_silence_is_local_ignore():
    result = assess("这句不用回答。", silence=True)
    assert result.action == CognitiveAction.IGNORE
    assert result.reason == ReasonCode.EXPLICIT_SILENCE_REQUEST
    assert not result.needs_model


def test_quiet_mode_is_local_wait():
    result = assess("你先听着。", quiet=True)
    assert result.action == CognitiveAction.WAIT
    assert result.reason == ReasonCode.QUIET_MODE


def test_waiting_for_others_is_wait_and_keeps_pending_followup():
    result = assess("等我一下。")
    assert result.action == CognitiveAction.WAIT
    assert result.rule == "local_wait_request"
    assert result.reply_necessity == ReplyNecessity.UNNECESSARY

    event = make_utterance_event(
        conversation_id="c1",
        text="等我一下。",
        addressee_hint="agent",
        source=EventSource.TEST,
    )
    analysis = _local_analysis("local_wait_request")
    decision = Policy().decide(analysis, ConversationContext(conversation_id="c1"))
    trace = derive_nonresponse_trace(
        event=event,
        analysis=analysis,
        policy_decision=decision,
        conversation_context=ConversationContext(conversation_id="c1"),
    )
    assert trace is not None
    assert trace.final_action == "WAIT"
    assert trace.cause_source == "LOCAL_POLICY"
    assert trace.pending_followup is True
    assert trace.reentry_condition == "NEXT_TOPIC_CONTINUATION"


def test_wait_phrase_with_a_question_is_not_silenced():
    result = assess("等一下，这个怎么弄？")
    assert result.action == CognitiveAction.RESPOND
    assert result.reply_necessity == ReplyNecessity.REQUIRED


def test_later_direct_address_responds_after_a_wait():
    assert assess("等我一下。").action == CognitiveAction.WAIT
    assert assess("Ava，你刚才看到那条消息了吗？").action == CognitiveAction.RESPOND


def test_explicit_memory_request_uses_deliberation():
    result = assess("请记住我喜欢喝乌龙茶。", remember=True)
    assert result.needs_model
    assert result.rule == "local_explicit_memory"


def test_disabled_fast_path_defers_instead_of_speaking_locally():
    result = assess("你好呀。", fast_path_enabled=False)
    assert result.needs_model


def test_wait_detector_is_narrow():
    assert looks_like_wait_request("等我一下。")
    assert looks_like_wait_request("我还没说完")
    assert not looks_like_wait_request("等一下，这个怎么弄？")
    assert not looks_like_wait_request("我在等你回答这个问题。")


def _local_analysis(rule):
    from companion.core.orchestrator import Orchestrator
    from companion.domain.enums import ReasonCode

    event = make_utterance_event(
        conversation_id="c1",
        text="等我一下。",
        addressee_hint="agent",
        source=EventSource.TEST,
    )
    assessment = assess("等我一下。")
    return Orchestrator._local_turn_analysis(event, assessment)


@pytest.mark.asyncio
async def test_no_response_provider_is_called_for_local_ignore(tmp_path):
    """A locally confirmed human-to-human turn never reaches the reply model."""
    from companion.runtime import build_runtime
    from companion.storage.database import Database

    db = await Database(tmp_path / "s.db").connect()
    await db.initialize()
    rt = build_runtime(db)
    person = await rt.registry.register(
        "Speaker",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    calls = {"n": 0}

    class CountingResponse:
        async def generate(self, *args, **kwargs):
            calls["n"] += 1
            raise AssertionError("response provider must not be called")

    rt.orchestrator.response = CountingResponse()
    rt.orchestrator.context_builder = ContextBuilder(
        repos=rt.repos,
        registry=rt.registry,
        agent_service=rt.agent_service,
        profile_service=rt.profile_service,
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="我昨天完成了一个项目。",
        addressee_hint="human",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)
    assert report.action == CognitiveAction.IGNORE
    assert report.response_text is None
    assert calls["n"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_local_ignore_still_records_the_event_and_learns_profile(tmp_path):
    """Silence is not a lost turn: it is recorded and enters profile review.

    V0.9.7.2: ordinary conversation text can no longer mutate the profile from
    the turn path. The silent turn persists the event, registers exactly one
    recoverable review request, and only the review (AI read -> local validate
    -> atomic commit) may apply the change.
    """
    from companion.runtime import build_runtime
    from companion.storage.database import Database

    db = await Database(tmp_path / "s2.db").connect()
    await db.initialize()
    rt = build_runtime(db)
    person = await rt.registry.register(
        "Silent",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    rt.orchestrator.context_builder = ContextBuilder(
        repos=rt.repos,
        registry=rt.registry,
        agent_service=rt.agent_service,
        profile_service=rt.profile_service,
    )
    event = make_utterance_event(
        conversation_id="c1",
        text="你以后跟我说话不用这么正式。",
        addressee_hint="human",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)
    assert report.action == CognitiveAction.IGNORE
    assert report.response_text is None

    # The event and its action are still persisted.
    stored = await rt.repos.utterance.get(event.event_id)
    assert stored is not None
    action = await rt.repos.analysis.get_action(event.event_id)
    assert action is not None and action.action_type == CognitiveAction.IGNORE

    # Local regex signals are diagnostics only: the profile is untouched.
    profile = await rt.profile_service.get(person.person_id)
    assert profile is not None
    assert profile.communication_value(ProfileFieldKey.FORMALITY) == "UNKNOWN"

    # The turn registered exactly one recoverable review request.
    receipt = await rt.repos.profile.get_review_receipt(
        person.person_id, event.event_id
    )
    assert receipt is not None and receipt["status"] == "PENDING"

    # The AI review path is the only writer; once it runs, the learned value
    # is visible in the profile.
    import logging

    from companion.cognition.privacy import ForegroundPrivacyGateway
    from companion.jobs.profile_review import build_profile_review_handler
    from companion.people.profile_refinement import (
        ProfileChangeProposal,
        ProfileRefinementEngine,
        ProfileReviewContextSource,
        ProfileReviewDecision,
    )
    from companion.providers.deepseek_profile import (
        ScriptedProfileRefinementProvider,
    )

    engine = ProfileRefinementEngine(
        rt.profile_service,
        rt.repos.profile,
        ScriptedProfileRefinementProvider(
            [
                ProfileReviewDecision(
                    person_id=person.person_id,
                    base_revision=profile.metadata.revision,
                    decision="PROPOSE_CHANGES",
                    reason="silent_turn",
                    changes=[
                        ProfileChangeProposal(
                            operation="ADD",
                            field_key="interaction.communication_feedback",
                            proposed_content="希望说话不要太正式，随意自然一点",
                            evidence_event_ids=[event.event_id],
                            confidence=0.95,
                        )
                    ],
                )
            ]
        ),
        ProfileReviewContextSource(
            rt.repos, privacy_gateway=ForegroundPrivacyGateway(rt.registry)
        ),
    )
    await build_profile_review_handler(engine)(
        {"person_id": person.person_id, "source_event_id": event.event_id},
        rt.repos,
        logging.getLogger("test.selective_response"),
    )
    entries = await rt.repos.profile.list_entries(
        person.person_id,
        statuses=("ACTIVE", "TENTATIVE"),
        field_key="interaction.communication_feedback",
    )
    assert len(entries) == 1
    assert "正式" in entries[0]["content"]
    # The enum interaction switch stays as it was: only the review-chosen
    # feedback entry was added.
    learned = await rt.profile_service.get(person.person_id)
    assert learned.communication_value(ProfileFieldKey.FORMALITY) == "UNKNOWN"
    await db.close()
