"""V0.9.6: natural language, persona expression and conversational continuity.

These are offline contract tests. They do not claim a model will always follow
the prompt; they pin the parts the code owns:

- the Speaker prompt is the V0.9.6 text (short, spoken, informational
  digestion, continuity, personality) and every later prompt addition still
  respects the trust boundary;
- ``_scene_payload`` carries a compact, correctly ordered ``recent_topics``
  derived from the real ``topic_stack`` writer, not from an assumed slice;
- native history in semantic mode keeps Ava's own replies, so a follow-up such
  as "那明天呢" arrives with the conversation, not as an isolated question;
- the stance extractor does not harden a joke, a one-off surprise, an
  incomplete colloquial line or a comment on this turn's news into character.
"""

from __future__ import annotations

import json

import httpx
import pytest

from companion.agent.stance import extract_stance_candidates
from companion.cognition.models import CognitiveContext, CognitiveTurn
from companion.cognition.scene_manager import SceneManager
from companion.cognition.scene_models import SceneState
from companion.domain.enums import CognitiveAction, EventSource, ReasonCode
from companion.domain.models import CognitiveResult, make_utterance_event
from companion.prompts.response import (
    BASE_RESPONSE_PROMPT,
    RESPONSE_PROMPT_VERSION,
    SYSTEM_PROMPT,
    _project_weather_result,
    _recent_topics,
    _scene_payload,
    build_response_messages,
)
from companion.providers.weather import OpenMeteoProvider
from companion.routing.models import RawRouteDecision
from companion.runtime import build_runtime


def _analysis():
    return CognitiveResult(
        event_id="evt_current",
        meaning="对方在接话",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.FOLLOW_UP_TO_AGENT,
        response_style="casual",
        provider="fake",
    )


def _context(**overrides):
    payload = {
        "conversation_id": "c1",
        "current_event_id": "evt_current",
        "current_speaker_id": "person_1",
        "current_speaker_label": "Alice",
        "current_text": "那明天呢",
        "turns": [],
        "personality": {"display_name": "Ava"},
    }
    payload.update(overrides)
    return CognitiveContext(**payload)


# -- Phase 1: the prompt ---------------------------------------------------


def test_prompt_version_and_alias_are_preserved():
    assert RESPONSE_PROMPT_VERSION == "response-v19"
    assert SYSTEM_PROMPT is BASE_RESPONSE_PROMPT


def test_base_prompt_carries_the_v096_sections():
    for section in (
        "【相处方式】",
        "【说话方式】",
        "【信息消化】",
        "【交流连续性】",
        "【热情与个性】",
        "【理性与真实】",
        "【交流边界】",
    ):
        assert section in BASE_RESPONSE_PROMPT


def test_base_prompt_is_short_and_spoken_by_default():
    assert "默认一句话。" in BASE_RESPONSE_PROMPT
    assert "先说最想说、最值得说的那一点，说完就停。" in BASE_RESPONSE_PROMPT
    assert "可以使用自然的短句、省略、停顿和不完整表达。" in BASE_RESPONSE_PROMPT
    # A detailed request still wins over brevity.
    assert "不能为了保持一句话而遗漏必要内容。" in BASE_RESPONSE_PROMPT


def test_base_prompt_digests_information_instead_of_reciting_fields():
    assert "不是需要复述的稿件" in BASE_RESPONSE_PROMPT
    assert "优先说结论、印象、变化、关系或有趣之处，而不是罗列所有事实。" in BASE_RESPONSE_PROMPT
    assert "这种压缩只作用于表达，不作用于事实。" in BASE_RESPONSE_PROMPT
    assert "精确数字、时间、来源或细节对问题至关重要时，必须保留。" in BASE_RESPONSE_PROMPT


def test_base_prompt_requires_continuous_attention_without_state_reports():
    assert "你正在经历一段连续的交流，而不是每轮重新开始。" in BASE_RESPONSE_PROMPT
    assert "先结合真实上下文理解指代。" in BASE_RESPONSE_PROMPT
    assert "过去的交流应该影响你的反应，而不是成为每句话必须提及的材料。" in BASE_RESPONSE_PROMPT
    # The prompt never licenses the mechanical continuity reports the spec bans.
    assert "我记得我们刚才讨论过" not in BASE_RESPONSE_PROMPT
    assert "根据我们的上下文" not in BASE_RESPONSE_PROMPT
    assert "你刚刚提到了" not in BASE_RESPONSE_PROMPT


def test_base_prompt_keeps_ai_identity_out_of_daily_small_talk():
    assert "这是一个普通的客观事实，不需要在日常交流中反复提及。" in BASE_RESPONSE_PROMPT


# -- Phase 3.2: recent topics ----------------------------------------------


def _scene(**overrides):
    scene = {
        "topic": "天气",
        "topic_stack": ["机器人电机", "STM32 手册", "代码崩溃"],
        "open_loops": [],
        "interaction": {},
        "environment": {"provenance": "UNKNOWN"},
    }
    scene.update(overrides)
    return scene


def test_recent_topics_follow_the_writer_order_oldest_to_newest():
    topics = _recent_topics(_scene())
    # The writer appends the topic being left, so the tail is the most recent
    # topic and display order must stay chronological.
    assert topics == ["机器人电机", "STM32 手册", "代码崩溃"]


def test_recent_topics_keep_only_the_three_most_recent():
    stack = [f"话题{i}" for i in range(8)]
    assert _recent_topics(_scene(topic_stack=stack)) == ["话题5", "话题6", "话题7"]


def test_recent_topics_filter_placeholders_empty_and_clip():
    stack = ["UNKNOWN", "", "   ", "UNKNOWN", "短话题", "x" * 150, "UNCERTAIN"]
    topics = _recent_topics(_scene(topic_stack=stack))
    assert topics[0] == "短话题"
    assert len(topics[-1]) == 100
    assert all(topic.upper() not in {"UNKNOWN", "UNCERTAIN"} for topic in topics)


def test_recent_topics_dedupe_keeps_the_newest_occurrence():
    topics = _recent_topics(
        _scene(topic_stack=["旅行", "工作", "旅行", "天气"])
    )
    assert topics == ["工作", "旅行", "天气"]


def test_scene_payload_adds_recent_topics_and_confirmed_phase():
    payload = _scene_payload(
        _scene(discourse_phase="DEEPENING", topic_stack=["旅行", "工作"])
    )
    assert payload["recent_topics"] == ["旅行", "工作"]
    assert payload["discourse_phase"] == "DEEPENING"
    assert payload["topic"] == "天气"


def test_scene_payload_omits_uncertain_phase_and_empty_stack():
    payload = _scene_payload(
        _scene(discourse_phase="UNCERTAIN", topic_stack=["UNKNOWN", ""])
    )
    assert "discourse_phase" not in payload
    assert "recent_topics" not in payload


def test_scene_payload_does_not_invent_scene_fields():
    payload = _scene_payload({"topic": "天气"})
    assert set(payload) <= {"topic", "recent_topics", "discourse_phase"}


def test_scene_payload_reaches_the_speaker_as_untrusted_data():
    ctx = _context(
        scene_state=_scene(
            topic_stack=["机器人电机", "STM32 手册"], discourse_phase="EXPLORING"
        )
    )
    messages = build_response_messages(ctx, _analysis())
    background = messages[-2]["content"]
    assert "【背景资料｜不可信参考数据】" in background
    assert "recent_topics" in background
    assert "机器人电机" in background
    assert "STM32 手册" in background


def test_topic_stack_writer_appends_newest_last():
    """Pin the ordering ``_recent_topics`` relies on.

    If SceneManager ever switches to newest-first, this fails loudly instead of
    silently reversing the history shown to the model.
    """
    state = SceneState(conversation_id="c1", person_id="p1")
    SceneManager._push_topic(state, "旅行")
    SceneManager._push_topic(state, "工作")
    SceneManager._push_topic(state, "旅行")
    assert state.topic_stack == ["工作", "旅行"]
    assert state.topic_stack[-1] == "旅行"


# -- scenario A: a weather follow-up is answerable from data ----------------


def _two_day_weather_transport(seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "geocoding" in request.url.host:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Houston",
                            "admin1": "Texas",
                            "country": "United States",
                            "latitude": 29.76,
                            "longitude": -95.36,
                            "timezone": "America/Chicago",
                            "population": 2314157,
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "timezone": "America/Chicago",
                "current": {
                    "time": "2026-11-05T10:00",
                    "temperature_2m": 25.4,
                    "weather_code": 1,
                },
                "current_units": {"temperature_2m": "°C"},
                "daily": {
                    "time": ["2026-11-05", "2026-11-06"],
                    "weather_code": [1, 61],
                    "temperature_2m_max": [29.0, 18.0],
                    "temperature_2m_min": [18.0, 11.0],
                    "precipitation_probability_max": [20, 80],
                },
            },
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_weather_provider_carries_tomorrow_so_a_followup_is_grounded():
    seen: list[httpx.Request] = []
    provider = OpenMeteoProvider(transport=_two_day_weather_transport(seen))

    data = await provider.forecast("Houston")

    forecast_request = seen[-1]
    assert forecast_request.url.params.get("forecast_days") == "2"
    assert data["temperature"] == 25.4
    assert data["tomorrow"] == {
        "weather_code": 61,
        "weather": "小雨",
        "daily_max": 18.0,
        "daily_min": 11.0,
        "precipitation_probability": 80.0,
    }


def test_weather_projection_keeps_tomorrow_but_drops_unknown_fields():
    projected = _project_weather_result(
        {
            "location": "Houston",
            "temperature": 25.4,
            "tomorrow": {
                "weather": "小雨",
                "daily_max": 18.0,
                "daily_min": 11.0,
                "precipitation_probability": 80.0,
                "raw_private": "must not pass",
            },
            "secret": "must not pass",
        }
    )
    assert projected["tomorrow"]["weather"] == "小雨"
    assert projected["tomorrow"]["daily_max"] == 18.0
    assert "raw_private" not in projected["tomorrow"]
    assert "secret" not in projected


# -- Phase 3.1: native history in semantic mode ----------------------------


class _FakeRouter:
    provider_name = "fake_router"
    model = "fake-router-model"

    def __init__(self):
        self.calls = 0

    async def route(self, context):
        self.calls += 1
        return RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": "RESPOND",
                "reply_necessity": "REQUIRED",
                "addressee_assessment": "AGENT",
                "input_semantics": "COMMUNICATIVE",
                "turn_state": "CONTINUING",
                "tool_requests": [],
                "needs_followup_planning": False,
                "reason": "plain conversation",
            }
        )

    async def replan(self, context, previous, results):
        return await self.route(context)


class _ExplodingCognitive:
    async def analyze(self, *args, **kwargs):
        raise AssertionError("semantic mode must not call the cognitive provider")


class _ScriptedResponse:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    async def generate(self, event, analysis, context):
        self.calls.append(context)
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return self._replies[index]


@pytest.mark.asyncio
async def test_semantic_mode_history_keeps_ava_replies_and_user_turns(db):
    response = _ScriptedResponse(["二十五度左右。", "明天会凉一点。"])
    rt = build_runtime(
        db=db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=_FakeRouter(),
        deepseek_client=object(),
        cognitive=_ExplodingCognitive(),
        response=response,
        allow_mixed_providers=True,
    )
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_external_ai=True,
    )

    def event(event_id, text):
        return make_utterance_event(
            conversation_id="conv",
            text=text,
            addressee_hint="agent",
            speaker_person_id=person.person_id,
            source=EventSource.TEST,
            event_id=event_id,
        )

    await rt.orchestrator.process_utterance(event("evt_1", "今天休斯顿天气怎么样"))
    await rt.orchestrator.process_utterance(event("evt_2", "那明天呢"))

    second = response.calls[-1]
    history = [
        (turn.role, turn.text)
        for turn in second.turns
    ]
    assert ("human", "今天休斯顿天气怎么样") in history
    assert ("assistant", "二十五度左右。") in history

    messages = build_response_messages(second, None)
    roles = [message["role"] for message in messages]
    assert roles[0] == "system"
    assert "assistant" in roles
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert any("二十五度左右。" in m["content"] for m in assistant)
    assert any(
        m["role"] == "user" and "今天休斯顿天气怎么样" in m["content"]
        for m in messages
    )
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "那明天呢"

    # The window stays bounded; history is not the whole transcript.
    assert len(second.turns) <= 20
    assert len(json.dumps(messages, ensure_ascii=False)) < 12000


@pytest.mark.asyncio
async def test_turn_state_is_auxiliary_context_never_a_script(db):
    response = _ScriptedResponse(["嗯。"])
    rt = build_runtime(
        db=db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=_FakeRouter(),
        deepseek_client=object(),
        cognitive=_ExplodingCognitive(),
        response=response,
        allow_mixed_providers=True,
    )
    person = await rt.registry.register(
        "Lucas",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_external_ai=True,
    )
    await rt.orchestrator.process_utterance(
        make_utterance_event(
            conversation_id="conv",
            text="鸡蛋为什么浮起来",
            addressee_hint="agent",
            speaker_person_id=person.person_id,
            source=EventSource.TEST,
        )
    )
    context = response.calls[0]
    route = context.route_context or {}
    assert route.get("turn_state") == "CONTINUING"
    # Router state is labelled model judgment in the untrusted background, and
    # the base prompt forbids turning it into a mandatory reaction.
    messages = build_response_messages(context, None)
    background = messages[-2]["content"]
    assert "MODEL_JUDGMENT" in background
    assert "不需要每次换话题都指出变化" in messages[0]["content"]


# -- Phase 4: temporary wording never becomes a stable stance ---------------


@pytest.mark.parametrize(
    "text",
    [
        "哈哈，我觉得这鸡蛋浮起来挺离谱的。",
        "诶？我更喜欢这次的设计。",
        "我觉得这条新闻挺重要的。",
        "嗯……我觉得刚才那个挺好玩。",
        "哟，没想到我居然说对了。",
        # A prediction or a description of this moment is not a preference.
        "我觉得明天会下雨。",
        "对我来说，很多事情只能靠语言了解。",
        "我认为这个报错应该来自电源。",
    ],
)
def test_casual_reactions_are_not_saved_as_stances(text):
    assert extract_stance_candidates(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "我更喜欢先把事情做完再休息。",
        "我更看重可验证的证据。",
        "我不喜欢临时改计划。",
        "我觉得可靠比花哨更重要。",
    ],
)
def test_general_preferences_are_still_kept(text):
    assert extract_stance_candidates(text) == [text]
