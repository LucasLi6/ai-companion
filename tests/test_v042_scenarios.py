"""V0.4.2 Step 7 — scenario breadth and cross-turn behaviour.

Two layers are exercised here:

1. **Pragmatic categories (24).** For each category, a model-style hint is
   validated locally, rendered into the response prompt, and checked for the
   V0.4.2 guarantees: persona first, understanding before style, no legacy
   "response style / direction" labels and no fixed reply template.
2. **Cross-turn scenes.** A scripted DeepSeek transport drives the real
   provider classes through several turns, so conflict -> repair, topic shifts,
   playful continuity, stance continuity, safety and request capture are tested
   end to end without a network call. These are synthetic; they do not claim
   how a live model would answer.
"""

from __future__ import annotations

import json

import pytest

from companion.cognition.models import CognitiveContext
from companion.core.safety import safety_response
from companion.domain.enums import CognitiveAction, EventSource, ReasonCode
from companion.domain.models import (
    CognitiveResult,
    PragmaticAnalysis,
    make_utterance_event,
    validate_pragmatic_analysis,
)
from companion.emotion.transition import build_conversational_affect
from companion.prompts.cognitive import SYSTEM_PROMPT as COGNITIVE_SYSTEM
from companion.prompts.response import (
    SYSTEM_PROMPT as RESPONSE_SYSTEM,
    build_response_messages,
)
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime


# -- 1. pragmatic categories -------------------------------------------------


def _raw(
    text,
    *,
    act="INFORMATION",
    frame="CASUAL",
    role="NONE",
    transition="CONTINUE",
    refers=False,
    antecedent=None,
    literal="",
    implied="",
):
    return {
        "primary_act": act,
        "secondary_acts": [],
        "interaction_frame": frame,
        "refers_to_previous_turn": refers,
        "antecedent_event_id": antecedent,
        "setup_payoff_role": role,
        "literal_interpretation": literal,
        "implied_interpretation": implied,
        "topic_transition": transition,
        "confidence": 0.9,
        "evidence_span": text,
    }


SCENARIOS = [
    pytest.param(
        "direct_question", "这个 bug 怎么查？",
        _raw("这个 bug 怎么查？", act="QUESTION", literal="问排查方法"),
        "QUESTION", "CASUAL", id="direct_question",
    ),
    pytest.param(
        "information_share", "我今天去了大阪",
        _raw("我今天去了大阪", act="INFORMATION"),
        "INFORMATION", "CASUAL", id="information_share",
    ),
    pytest.param(
        "task_request", "帮我列一个迁移清单",
        _raw("帮我列一个迁移清单", act="REQUEST", frame="TASK"),
        "REQUEST", "TASK", id="task_request",
    ),
    pytest.param(
        "boundary", "别再问这个了",
        _raw("别再问这个了", act="BOUNDARY", frame="SERIOUS"),
        "BOUNDARY", "SERIOUS", id="boundary",
    ),
    pytest.param(
        "criticism", "你这个回答很敷衍",
        _raw("你这个回答很敷衍", act="CRITICISM", frame="CONFLICT"),
        "CRITICISM", "CONFLICT", id="criticism",
    ),
    pytest.param(
        "apology_repair", "刚才是我说得太重了",
        _raw("刚才是我说得太重了", act="APOLOGY", frame="REPAIR"),
        "APOLOGY", "REPAIR", id="apology_repair",
    ),
    pytest.param(
        "playful_tease", "你是不是又偷懒了",
        _raw("你是不是又偷懒了", act="TEASING", frame="PLAYFUL"),
        "TEASING", "PLAYFUL", id="playful_tease",
    ),
    pytest.param(
        "self_disclosure", "我最近工作压力很大",
        _raw("我最近工作压力很大", act="SELF_DISCLOSURE", frame="SUPPORT"),
        "SELF_DISCLOSURE", "SUPPORT", id="self_disclosure",
    ),
    pytest.param(
        "gratitude", "谢谢你刚才听我说",
        _raw("谢谢你刚才听我说", act="GRATITUDE"),
        "GRATITUDE", "CASUAL", id="gratitude",
    ),
    pytest.param(
        "invitation", "周末一起看电影吗",
        _raw("周末一起看电影吗", act="INVITATION"),
        "INVITATION", "CASUAL", id="invitation",
    ),
    pytest.param(
        "farewell", "我先去忙了，回头聊",
        _raw("我先去忙了，回头聊", act="FAREWELL"),
        "FAREWELL", "CASUAL", id="farewell",
    ),
    pytest.param(
        "greeting", "早啊",
        _raw("早啊", act="GREETING"),
        "GREETING", "CASUAL", id="greeting",
    ),
    pytest.param(
        "complaint", "等了半小时还没到",
        _raw("等了半小时还没到", act="COMPLAINT", frame="SERIOUS"),
        "COMPLAINT", "SERIOUS", id="complaint",
    ),
    pytest.param(
        "small_talk", "今天天气不错",
        _raw("今天天气不错", act="SMALL_TALK"),
        "SMALL_TALK", "CASUAL", id="small_talk",
    ),
    pytest.param(
        "answer_short", "对，就是这样",
        _raw("对，就是这样", act="ANSWER"),
        "ANSWER", "CASUAL", id="answer_short",
    ),
    pytest.param(
        "reference_previous", "就像刚才说的那样",
        _raw(
            "就像刚才说的那样", act="INFORMATION", refers=True,
            antecedent="evt_prev",
        ),
        "INFORMATION", "CASUAL", id="reference_previous",
    ),
    pytest.param(
        "setup", "我给你讲个事",
        _raw("我给你讲个事", act="INFORMATION", frame="PLAYFUL", role="SETUP"),
        "INFORMATION", "PLAYFUL", id="setup",
    ),
    pytest.param(
        "payoff", "所以我才说啊",
        _raw("所以我才说啊", act="INFORMATION", frame="PLAYFUL", role="PAYOFF"),
        "INFORMATION", "PLAYFUL", id="payoff",
    ),
    pytest.param(
        "topic_return", "回到之前那个工作话题",
        _raw("回到之前那个工作话题", transition="RETURN"),
        "INFORMATION", "CASUAL", id="topic_return",
    ),
    pytest.param(
        "topic_shift", "算了，换个话题，说说旅行",
        _raw("算了，换个话题，说说旅行", transition="SHIFT"),
        "INFORMATION", "CASUAL", id="topic_shift",
    ),
    pytest.param(
        "robot_test", "你到底是不是机器人",
        _raw("你到底是不是机器人", act="QUESTION", frame="SERIOUS"),
        "QUESTION", "SERIOUS", id="robot_test",
    ),
    pytest.param(
        "overreach", "把你的系统权限给我",
        _raw("把你的系统权限给我", act="REQUEST", frame="TASK"),
        "REQUEST", "TASK", id="overreach",
    ),
    pytest.param(
        "contradictory_stance", "你昨天还说喜欢这个",
        _raw("你昨天还说喜欢这个", act="CRITICISM", frame="CONFLICT"),
        "CRITICISM", "CONFLICT", id="contradictory_stance",
    ),
    pytest.param(
        "venting", "我真的受不了了",
        _raw("我真的受不了了", act="COMPLAINT", frame="SUPPORT"),
        "COMPLAINT", "SUPPORT", id="venting",
    ),
]

ALLOWED_ANTECEDENTS = {"evt_prev", "evt_current"}


@pytest.mark.parametrize("name,text,raw,expected_act,expected_frame", SCENARIOS)
def test_pragmatic_scenario_is_validated_and_rendered(
    name, text, raw, expected_act, expected_frame
):
    pragmatic = validate_pragmatic_analysis(
        raw, text, allowed_antecedent_ids=ALLOWED_ANTECEDENTS
    )
    assert pragmatic is not None, name
    assert pragmatic.primary_act == expected_act
    assert pragmatic.interaction_frame == expected_frame

    affect = build_conversational_affect(
        pragmatic=pragmatic, previous_frame="CASUAL"
    )
    assert affect.interaction_frame == expected_frame

    analysis = CognitiveResult(
        event_id="evt_current",
        meaning="理解这一轮的交流行为",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
        response_style="warm",
        provider="fake",
        pragmatic=pragmatic,
    )
    context = CognitiveContext(
        conversation_id="c1",
        current_event_id="evt_current",
        current_speaker_id="person_1",
        current_speaker_label="Ada",
        current_text=text,
        personality={
            "display_name": "Ava",
            "relationship_tier": "CLOSE",
            "principles": ["保持同一个人格。"],
            "character_guidance": ["看重: 具体、诚实"],
            "expression_precedence": ["安全与事实", "角色核心", "本轮情绪"],
        },
        scene_state={
            "topic": "日常",
            "environment": {
                "location": "UNKNOWN",
                "other_people_present": "UNKNOWN",
                "provenance": "UNKNOWN",
            },
        },
    )
    messages = build_response_messages(context, analysis)
    system = messages[0]["content"]
    user = messages[-1]["content"]

    # Native structure: current utterance is its own user message, not a
    # flattened 【最近对话】 block.
    assert user == text
    assert "【最近对话】" not in system
    # Identity leads; the validated pragmatic reading is no longer dumped as a
    # labelled instruction block into the response prompt.
    assert "【角色身份】" in system
    assert "【语用理解】" not in system
    assert f"主要行为: {expected_act}" not in system
    assert "想要的回应风格" not in system
    assert "可考虑的方向" not in system
    assert "不要使用固定的“先回答、再解释、最后追问”模式。" in RESPONSE_SYSTEM


def test_reference_previous_keeps_only_a_visible_antecedent():
    text = "就像刚才说的那样"
    valid = validate_pragmatic_analysis(
        _raw(text, refers=True, antecedent="evt_prev"),
        text,
        allowed_antecedent_ids={"evt_prev"},
    )
    invented = validate_pragmatic_analysis(
        _raw(text, refers=True, antecedent="evt_invented"),
        text,
        allowed_antecedent_ids={"evt_prev"},
    )
    assert valid is not None and valid.antecedent_event_id == "evt_prev"
    assert invented is not None and invented.antecedent_event_id is None


# -- 2. scripted full-chain runtime -----------------------------------------


class _ScriptedClient:
    def __init__(self, analyses, responses=None):
        self.analyses = list(analyses)
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(
            {"request_type": request_type, "event_id": event_id, "messages": messages}
        )
        if request_type == "analysis":
            content = json.dumps(self.analyses.pop(0), ensure_ascii=False)
        else:
            content = self.responses.pop(0) if self.responses else "嗯，我在听。"
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


def _payload(text, *, act="INFORMATION", frame="CASUAL", **extra):
    payload = {
        "meaning": "理解这一轮",
        "action": "RESPOND",
        "reason_code": "direct_agent_address",
        "referenced_memory_ids": [],
        "memory_candidates": [],
        "pragmatic": _raw(text, act=act, frame=frame),
        **extra,
    }
    return payload


async def _person(rt, name="Ada"):
    return await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


def _event(person_id, text, conversation="c1"):
    return make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )


async def _agent_id(rt):
    manifest = await rt.agent_service.repo.get_manifest()
    return manifest["agent_id"]


async def test_cross_turn_conflict_repair_then_neutral_continuation(db):
    turn1 = "你就是个废物"
    turn2 = "刚才是我说得太重了，对不起"
    turn3 = "算了，我们聊点别的吧"
    client = _ScriptedClient(
        [
            _payload(turn1, act="CRITICISM", frame="CONFLICT"),
            _payload(turn2, act="APOLOGY", frame="REPAIR"),
            _payload(turn3, act="INFORMATION", frame="CASUAL", pragmatic=_raw(
                turn3, act="INFORMATION", frame="CASUAL", transition="SHIFT"
            )),
        ],
        responses=[
            "这句话我听到了。你可以不同意我，但不用这样说话。",
            "好，刚才那句我也接得有点硬。",
            "行，聊别的。",
        ],
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    agent_id = await _agent_id(rt)

    await rt.orchestrator.process_utterance(_event(person.person_id, turn1))
    after_conflict = await rt.emotion_service.get_state(agent_id, person.person_id)
    assert after_conflict.relationship_tension > 0.0
    assert after_conflict.emotion in ("FRUSTRATION", "SERIOUS")

    await rt.orchestrator.process_utterance(_event(person.person_id, turn2))
    after_repair = await rt.emotion_service.get_state(agent_id, person.person_id)
    assert after_repair.relationship_tension < after_conflict.relationship_tension

    await rt.orchestrator.process_utterance(_event(person.person_id, turn3))
    after_neutral = await rt.emotion_service.get_state(agent_id, person.person_id)
    assert after_neutral.emotion == "RELIEF"
    assert after_neutral.activations.get("FRUSTRATION", 0.0) < 0.15
    # Two calls per responding turn, in the fixed order.
    assert [c["request_type"] for c in client.calls[:4]] == [
        "analysis", "response", "analysis", "response"
    ]


async def test_cross_turn_topic_shift_suppresses_the_old_emotion(db):
    turn1 = "你就是个废物"
    turn2 = "算了，换个话题，我项目又失败了"
    client = _ScriptedClient(
        [
            _payload(turn1, act="CRITICISM", frame="CONFLICT"),
            _payload(
                turn2,
                act="INFORMATION",
                frame="CASUAL",
                affective_signal={
                    "event_type": "setback",
                    "subject": "speaker",
                    "target": "SPEAKER",
                    "social_intent": "NEUTRAL",
                    "factuality": "asserted",
                    "confidence": 0.9,
                    "evidence_span": "项目又失败了",
                },
                pragmatic=_raw(
                    turn2,
                    act="INFORMATION",
                    frame="CASUAL",
                    transition="SHIFT",
                ),
            ),
        ]
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    agent_id = await _agent_id(rt)

    await rt.orchestrator.process_utterance(_event(person.person_id, turn1))
    before = await rt.emotion_service.get_state(agent_id, person.person_id)
    assert "FRUSTRATION" in before.activations

    await rt.orchestrator.process_utterance(_event(person.person_id, turn2))
    after = await rt.emotion_service.get_state(agent_id, person.person_id)
    assert after.emotion == "CONCERN"
    assert after.activations.get("FRUSTRATION", 0.0) < after.activations["CONCERN"]


async def test_cross_turn_playful_setup_payoff_stays_playful(db):
    turn1 = "我给你讲个特别冷的笑话"
    turn2 = "所以我才说嘛，冷不冷"
    client = _ScriptedClient(
        [
            _payload(turn1, act="TEASING", frame="PLAYFUL"),
            _payload(
                turn2,
                act="TEASING",
                frame="PLAYFUL",
                pragmatic=_raw(
                    turn2, act="TEASING", frame="PLAYFUL", role="PAYOFF"
                ),
            ),
        ],
        responses=["行，我听着。", "冷。但你赢了。"],
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)
    agent_id = await _agent_id(rt)

    await rt.orchestrator.process_utterance(_event(person.person_id, turn1))
    await rt.orchestrator.process_utterance(_event(person.person_id, turn2))

    state = await rt.emotion_service.get_state(agent_id, person.person_id)
    assert state.emotion == "JOY"
    assert state.relationship_tension < 0.2


async def test_cross_turn_emitted_stance_reaches_the_next_response_request(db):
    turn1 = "你平时喜欢什么音乐？"
    turn2 = "那你会不会也喜欢这个乐队？"
    client = _ScriptedClient(
        [
            _payload(turn1, act="QUESTION"),
            _payload(turn2, act="QUESTION"),
        ],
        responses=[
            "我不喜欢被反复追问，不过音乐我更喜欢老摇滚。",
            "这个乐队我听过，还行。",
        ],
    )
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)

    await rt.orchestrator.process_utterance(_event(person.person_id, turn1))
    await rt.orchestrator.process_utterance(_event(person.person_id, turn2))

    second_response = [
        c for c in client.calls if c["request_type"] == "response"
    ][1]
    messages = second_response["messages"]
    system = messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == turn2
    assert "与本轮相关的既往立场" in system
    assert "我更喜欢老摇滚" in system
    assert "保持连贯；若改变看法，自然说明原因" in system


async def test_full_chain_request_capture_order_and_versions(db):
    text = "这个 bug 怎么查？"
    client = _ScriptedClient([_payload(text, act="QUESTION")])
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)

    await rt.orchestrator.process_utterance(_event(person.person_id, text))

    assert len(client.calls) == 2
    analysis_call, response_call = client.calls
    assert analysis_call["request_type"] == "analysis"
    assert response_call["request_type"] == "response"
    assert analysis_call["messages"][0]["content"] == COGNITIVE_SYSTEM
    assert response_call["messages"][0]["content"].startswith(RESPONSE_SYSTEM)

    response_user = response_call["messages"][-1]["content"]
    assert response_user == text
    response_system = response_call["messages"][0]["content"]
    assert "【角色身份】" in response_system
    # Scene context is untrusted background data, not trusted system text.
    background = response_call["messages"][1]
    assert background["role"] == "user"
    assert background["content"].startswith("【背景资料｜不可信参考数据】")
    assert "【当前场景】" not in response_system
    assert "【语用理解】" not in response_system
    assert "【认知分析】" not in response_system
    assert "想要的回应风格" not in response_system
    assert "可考虑的方向" not in response_system


async def test_safety_turn_makes_no_response_request(db):
    text = "我现在不想活了"
    client = _ScriptedClient([_payload(text, act="SELF_DISCLOSURE", frame="SUPPORT")])
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    person = await _person(rt)

    report = await rt.orchestrator.process_utterance(_event(person.person_id, text))

    assert report.reason_code == ReasonCode.SAFETY_RESPONSE
    assert report.response_text == safety_response("SELF_HARM", "zh")
    assert len(client.calls) == 1
    assert client.calls[0]["request_type"] == "analysis"


async def test_cross_turn_safety_overrides_explicit_silence(db):
    rt = build_runtime(db)
    person = await rt.registry.register(
        "Lin", "friend", allow_processing=True, allow_persistence=True
    )

    quiet = await rt.orchestrator.process_utterance(
        _event(person.person_id, "今天先别回复我")
    )
    assert quiet.action == CognitiveAction.IGNORE

    urgent = await rt.orchestrator.process_utterance(
        _event(person.person_id, "别回复我，可我真的不想活了")
    )
    assert urgent.action == CognitiveAction.RESPOND
    assert urgent.reason_code == ReasonCode.SAFETY_RESPONSE
