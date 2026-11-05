"""V0.4 — multi-turn continuity through the real DeepSeek provider boundary.

There is no live model or key here: ``DeepSeekCognitiveProvider`` and
``DeepSeekResponseProvider`` are the production classes, driven by a scripted
transport that returns synthetic, authorized Chinese dialogue. This exercises
context building, prompt rendering, scene persistence, policy, personality
compilation and the next turn's actual prompt — the same path a live call takes.

The scenarios cover casual chat, jokes/play, technical continuity, topic
interruption, silence, a 30-turn consistency run, multi-person isolation,
correction and revoked consent.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta, timezone

from companion.cognition.scene_repository import SceneRepository
from companion.domain.enums import CognitiveAction, EventSource
from companion.domain.models import make_utterance_event
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime

BASE = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)


class ScriptedDeepSeekClient:
    """A fake transport for the real providers.

    ``script`` is a list of ``(analysis_dict, response_text_or_None)`` pairs in
    turn order. Analysis calls consume the next pair; the following response
    call (only for RESPOND) returns its text.
    """

    def __init__(self, script):
        self.script = deque(script)
        self.calls: list[dict] = []
        self._current_response = None
        self._current_event = None

    async def complete(
        self,
        messages,
        *,
        request_type,
        event_id=None,
        json_mode=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "request_type": request_type,
                "messages": messages,
                "event_id": event_id,
                "json_mode": json_mode,
            }
        )
        if request_type == "analysis":
            analysis, response = self.script.popleft()
            self._current_response = response
            self._current_event = event_id
            content = json.dumps(analysis, ensure_ascii=False)
        else:
            assert event_id == self._current_event
            assert self._current_response is not None
            content = self._current_response
        return ChatResponse(
            content=content,
            model="deepseek-flash",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )

    def requests(self, request_type):
        return [c for c in self.calls if c["request_type"] == request_type]

    def last_prompt(self, request_type):
        items = self.requests(request_type)
        assert items, f"no {request_type} call recorded"
        return items[-1]["messages"][1]["content"]


def _analysis(
    meaning,
    *,
    action="RESPOND",
    reason="direct_agent_address",
    scene=None,
    motive=None,
    addressee="agent",
):
    return {
        "meaning": meaning,
        "intent": "question" if action == "RESPOND" else "statement",
        "addressed_to": addressee,
        "reply_necessity": "REQUIRED" if action == "RESPOND" else "OPTIONAL",
        "conversation_state": "ongoing",
        "action": action,
        "reason_code": reason,
        "referenced_memory_ids": [],
        "memory_candidates": [],
        "response_style": "warm",
        "affective_signal": None,
        "nonresponse_motive": motive,
        "scene_proposal": scene,
    }


def _scene(
    *,
    change="SHIFT",
    topic="旅行",
    evidence="我去了日本",
    mode="CONVERSATION",
    phase="EXPLORING",
    tone="WARM",
    open_loop=None,
    resolve=None,
    environment=None,
):
    return {
        "topic_change": change,
        "topic": topic,
        "mode": mode,
        "discourse_phase": phase,
        "tone": tone,
        "environment": environment,
        "open_loop": open_loop,
        "resolve_loop_topics": resolve or [],
        "confidence": 0.9,
        "evidence_span": evidence,
    }


def _motive(kind, evidence, *, reentry="NO_FOLLOWUP"):
    return {
        "motive": kind,
        "confidence": 0.8,
        "evidence_span": evidence,
        "reentry_hint": reentry,
    }


async def _person(rt, name="Alice", *, external_ai=True, persistence=True):
    return await rt.registry.register(
        name,
        "owner",
        allow_processing=True,
        allow_persistence=persistence,
        allow_memory_learning=True,
        allow_external_ai=external_ai,
    )


async def _say(rt, person_id, text, turn, *, conversation="c1", to="agent"):
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint=to,
        speaker_person_id=person_id,
        source=EventSource.TEST,
        occurred_at=BASE + timedelta(seconds=turn * 10),
    )
    return await rt.orchestrator.process_utterance(event), event


def _runtime(db, script):
    client = ScriptedDeepSeekClient(script)
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        fast_path_enabled=False,
        router_mode="legacy",
)
    return rt, client


async def _snapshot(rt, person_id, conversation="c1"):
    return await SceneRepository(rt.db).load(conversation, person_id)


# -- A. casual chat ---------------------------------------------------------


async def test_casual_topic_is_carried_into_the_next_real_prompt(db):
    rt, client = _runtime(
        db,
        [
            (
                _analysis(
                    "对方分享了日本旅行",
                    scene=_scene(topic="旅行", evidence="我去了日本"),
                ),
                "日本挺适合慢慢逛的。",
            ),
            (
                _analysis(
                    "对方继续讲旅行细节",
                    scene=_scene(change="CONTINUE", topic="旅行", evidence="大阪也去了"),
                ),
                "大阪更适合吃。",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "我去了日本", 1)
    await _say(rt, alice.person_id, "大阪也去了", 2)

    second_prompt = client.requests("analysis")[1]["messages"][1]["content"]
    assert "【当前场景】" in second_prompt
    assert "话题: 旅行" in second_prompt
    assert "日本挺适合慢慢逛的。" in second_prompt  # prior assistant turn
    assert "我去了日本" in second_prompt

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "旅行"
    assert snapshot.revision == 1


async def test_topic_shift_keeps_the_old_topic_on_the_stack(db):
    rt, client = _runtime(
        db,
        [
            (_analysis("旅行", scene=_scene(topic="旅行", evidence="我去了日本")), "日本怎么样？"),
            (
                _analysis(
                    "换到工作",
                    scene=_scene(topic="工作", evidence="今天聊聊工作"),
                ),
                "工作怎么了？",
            ),
            (
                _analysis(
                    "继续工作",
                    scene=_scene(change="CONTINUE", topic="工作", evidence="项目延期了"),
                ),
                "延期到什么时候？",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "我去了日本", 1)
    await _say(rt, alice.person_id, "今天聊聊工作", 2)
    await _say(rt, alice.person_id, "项目延期了", 3)

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "工作"
    assert snapshot.state.topic_stack == ["旅行"]
    third_prompt = client.requests("analysis")[2]["messages"][1]["content"]
    assert "话题: 工作" in third_prompt
    assert "话题栈: 旅行" in third_prompt


async def test_return_restores_an_earlier_topic_without_inventing_it(db):
    rt, client = _runtime(
        db,
        [
            (_analysis("旅行", scene=_scene(topic="旅行", evidence="我去了日本")), "日本怎么样？"),
            (_analysis("工作", scene=_scene(topic="工作", evidence="今天聊聊工作")), "工作怎么了？"),
            (
                _analysis(
                    "回到旅行",
                    scene=_scene(change="RETURN", topic="旅行", evidence="还是说回旅行"),
                ),
                "说回旅行吧。",
            ),
            (
                _analysis(
                    "继续旅行",
                    scene=_scene(change="CONTINUE", topic="旅行", evidence="下次想去北海道"),
                ),
                "北海道冬天不错。",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "我去了日本", 1)
    await _say(rt, alice.person_id, "今天聊聊工作", 2)
    await _say(rt, alice.person_id, "还是说回旅行", 3)
    await _say(rt, alice.person_id, "下次想去北海道", 4)

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "旅行"
    assert snapshot.state.topic_stack == []
    fourth_prompt = client.requests("analysis")[3]["messages"][1]["content"]
    assert "话题: 旅行" in fourth_prompt


# -- B. technical / task continuity ----------------------------------------


async def test_task_mode_and_deepening_phase_persist(db):
    rt, client = _runtime(
        db,
        [
            (
                _analysis(
                    "调试支付回调",
                    scene=_scene(
                        topic="调试支付回调",
                        evidence="支付回调一直失败",
                        mode="TASK",
                        phase="DEEPENING",
                        tone="SERIOUS",
                    ),
                ),
                "先看回调有没有到网关。",
            ),
            (
                _analysis(
                    "继续排查",
                    scene=_scene(
                        change="CONTINUE",
                        topic="调试支付回调",
                        evidence="网关日志是空的",
                        mode="TASK",
                        phase="DEEPENING",
                        tone="SERIOUS",
                    ),
                ),
                "那先确认签名校验。",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "支付回调一直失败", 1)
    await _say(rt, alice.person_id, "网关日志是空的", 2)

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.mode == "TASK"
    assert snapshot.state.discourse_phase == "DEEPENING"
    assert snapshot.state.tone == "SERIOUS"
    second_prompt = client.requests("analysis")[1]["messages"][1]["content"]
    assert "模式: TASK" in second_prompt
    assert "阶段: DEEPENING" in second_prompt
    assert "语气: SERIOUS" in second_prompt


async def test_interruption_does_not_erase_the_scene(db):
    rt, client = _runtime(
        db,
        [
            (_analysis("旅行", scene=_scene(topic="旅行", evidence="我去了日本")), "日本怎么样？"),
            (
                _analysis(
                    "没听清对方在跟谁说",
                    action="WAIT",
                    reason="ambiguous_addressee",
                    addressee="unknown",
                    motive=_motive("NOT_ADDRESSED", "嗯？"),
                    scene=_scene(
                        change="CONTINUE",
                        topic=None,
                        evidence=None,
                        tone="NEUTRAL",
                    ),
                ),
                None,
            ),
            (
                _analysis(
                    "继续说旅行",
                    scene=_scene(change="CONTINUE", topic="旅行", evidence="京都也不错"),
                ),
                "京都适合慢慢走。",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "我去了日本", 1)
    report, _ = await _say(rt, alice.person_id, "嗯？", 2, to="unknown")
    assert report.action == CognitiveAction.WAIT
    await _say(rt, alice.person_id, "京都也不错", 3)

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "旅行"
    third_prompt = client.requests("analysis")[2]["messages"][1]["content"]
    assert "话题: 旅行" in third_prompt
    assert "【你之前实际执行的行为】" in third_prompt


# -- C. silence and open loops ---------------------------------------------


async def test_pending_followup_creates_a_visible_open_loop(db):
    rt, client = _runtime(
        db,
        [
            (
                _analysis(
                    "对方还有工作的事没讲",
                    action="WAIT",
                    reason="ambiguous_addressee",
                    addressee="unknown",
                    motive=_motive(
                        "AWAITING_CONTINUATION",
                        "工作的事我等会儿说",
                        reentry="NEXT_TOPIC_CONTINUATION",
                    ),
                    scene=_scene(
                        topic="工作",
                        evidence="工作的事我等会儿说",
                        open_loop={
                            "topic": "工作",
                            "summary": "工作的事还没讲完",
                            "evidence_span": "工作的事我等会儿说",
                        },
                    ),
                ),
                None,
            ),
            (
                _analysis(
                    "对方回到工作",
                    scene=_scene(
                        change="CONTINUE", topic="工作", evidence="现在说工作吧"
                    ),
                ),
                "好，你说。",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "工作的事我等会儿说", 1)
    snapshot = await _snapshot(rt, alice.person_id)
    assert len(snapshot.state.open_loops) == 1
    assert snapshot.state.open_loops[0].summary == "工作的事还没讲完"

    await _say(rt, alice.person_id, "现在说工作吧", 2)
    second_prompt = client.requests("analysis")[1]["messages"][1]["content"]
    assert "未完成: 工作" in second_prompt


async def test_nonpending_wait_never_creates_an_open_loop(db):
    rt, _ = _runtime(
        db,
        [
            (
                _analysis(
                    "不是对我说的",
                    action="WAIT",
                    reason="ambiguous_addressee",
                    addressee="unknown",
                    motive=_motive("NOT_ADDRESSED", "今天好热"),
                    scene=_scene(
                        topic="天气",
                        evidence="今天好热",
                        open_loop={
                            "topic": "天气",
                            "summary": "天气话题未完成",
                            "evidence_span": "今天好热",
                        },
                    ),
                ),
                None,
            )
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "今天好热", 1)
    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.open_loops == []


async def test_new_topic_after_silence_is_not_stuck_in_the_old_one(db):
    rt, client = _runtime(
        db,
        [
            (_analysis("旅行", scene=_scene(topic="旅行", evidence="我去了日本")), "日本怎么样？"),
            (
                _analysis(
                    "没接上",
                    action="WAIT",
                    reason="ambiguous_addressee",
                    addressee="unknown",
                    motive=_motive("UNCERTAIN_CONTEXT", "嗯"),
                    scene=_scene(
                        change="CONTINUE", topic=None, evidence=None, tone="NEUTRAL"
                    ),
                ),
                None,
            ),
            (
                _analysis(
                    "换到新话题",
                    scene=_scene(topic="养猫", evidence="我最近养了只猫"),
                ),
                "猫多大了？",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "我去了日本", 1)
    await _say(rt, alice.person_id, "嗯", 2, to="unknown")
    await _say(rt, alice.person_id, "我最近养了只猫", 3)

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "养猫"
    assert snapshot.state.topic_stack == ["旅行"]
    third_prompt = client.requests("analysis")[2]["messages"][1]["content"]
    # The prompt is assembled *before* this turn's proposal is committed, so it
    # still shows the interrupted topic; the committed state then moves on.
    assert "话题: 旅行" in third_prompt
    assert "我最近养了只猫" in third_prompt


# -- D. correction and multi-person isolation -----------------------------


async def test_correction_updates_topic_and_keeps_evidence_local(db):
    rt, client = _runtime(
        db,
        [
            (
                _analysis(
                    "对方说住在北京",
                    scene=_scene(topic="居住地", evidence="我住在北京"),
                ),
                "北京挺干的。",
            ),
            (
                _analysis(
                    "对方纠正为上海",
                    scene=_scene(topic="居住地", evidence="其实我搬到上海了"),
                ),
                "上海潮湿一些。",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "我住在北京", 1)
    await _say(rt, alice.person_id, "其实我搬到上海了", 2)

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "居住地"
    assert snapshot.state.topic_stack == []  # same-topic correction is not a stack push
    second_prompt = client.requests("analysis")[1]["messages"][1]["content"]
    assert "其实我搬到上海了" in second_prompt
    # The old statement remains history but the scene evidence is the correction.
    assert snapshot.state.evidence[-1].quote == "其实我搬到上海了"


async def test_two_people_in_one_conversation_have_isolated_scenes(db):
    rt, client = _runtime(
        db,
        [
            (_analysis("Alice 旅行", scene=_scene(topic="旅行", evidence="我去了日本")), "日本怎么样？"),
            (_analysis("Bob 机械臂", scene=_scene(topic="机械臂", evidence="机械臂坏了")), "哪里坏了？"),
            (
                _analysis(
                    "Alice 继续旅行",
                    scene=_scene(change="CONTINUE", topic="旅行", evidence="北海道也想去"),
                ),
                "北海道不错。",
            ),
        ],
    )
    alice = await _person(rt, "Alice")
    bob = await _person(rt, "Bob")
    await _say(rt, alice.person_id, "我去了日本", 1)
    await _say(rt, bob.person_id, "机械臂坏了", 2)
    await _say(rt, alice.person_id, "北海道也想去", 3)

    a_scene = await _snapshot(rt, alice.person_id)
    b_scene = await _snapshot(rt, bob.person_id)
    assert a_scene.state.topic == "旅行"
    assert b_scene.state.topic == "机械臂"
    third_prompt = client.requests("analysis")[2]["messages"][1]["content"]
    assert "话题: 旅行" in third_prompt
    assert "话题: 机械臂" not in third_prompt


# -- E. long-run consistency ------------------------------------------------


async def test_thirty_turns_keep_revision_and_personality_stable(db):
    script = []
    for index in range(30):
        text = f"内容{index}"
        script.append(
            (
                _analysis(
                    f"第 {index} 轮",
                    scene=_scene(
                        topic=f"话题{index}",
                        evidence=text,
                        tone="WARM" if index % 2 == 0 else "NEUTRAL",
                    ),
                ),
                f"回应{index}",
            )
        )
    rt, client = _runtime(db, script)
    alice = await _person(rt)
    for index in range(30):
        report, _ = await _say(rt, alice.person_id, f"内容{index}", index)
        assert report.status == "COMPLETED"

    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.revision == 29
    assert snapshot.state.topic == "话题29"
    assert len(client.requests("analysis")) == 30
    assert len(client.requests("response")) == 30

    first_response_prompt = client.requests("response")[0]["messages"][0]["content"]
    last_response_prompt = client.requests("response")[-1]["messages"][0]["content"]
    for prompt in (first_response_prompt, last_response_prompt):
        assert "【角色身份】" in prompt
        assert "【边界】" in prompt
        assert "与当前说话者的关系：" in prompt
    # Both calls see the same trusted character view.
    assert first_response_prompt.split("【当前相关的上下文】")[0] == (
        last_response_prompt.split("【当前相关的上下文】")[0]
    )
    # The 30th analysis sees the 29th scene, not a stale early one.
    last_analysis_prompt = client.requests("analysis")[-1]["messages"][1]["content"]
    assert "话题: 话题28" in last_analysis_prompt


# -- F. consent boundaries --------------------------------------------------


async def test_revoked_external_ai_deletes_scene_and_starts_fresh(db):
    rt, client = _runtime(
        db,
        [
            (_analysis("旅行", scene=_scene(topic="旅行", evidence="我去了日本")), "日本怎么样？"),
            (
                _analysis(
                    "重新建立话题",
                    scene=_scene(topic="养猫", evidence="我养了只猫"),
                ),
                "猫叫什么？",
            ),
        ],
    )
    alice = await _person(rt)
    await _say(rt, alice.person_id, "我去了日本", 1)
    assert await _snapshot(rt, alice.person_id) is not None

    consent = await rt.registry.get_consent(alice.person_id)
    await rt.registry.set_consent(
        consent.model_copy(update={"allow_external_ai": False})
    )
    report, _ = await _say(rt, alice.person_id, "还在吗", 2)
    assert report.action == CognitiveAction.WAIT
    assert len(client.requests("analysis")) == 1  # no network after revocation
    # The old derived scene is deleted, not retained. A new local snapshot for
    # the post-revocation turn is UNKNOWN and contains none of the old topic.
    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot is None or (
        snapshot.state.topic == "UNKNOWN" and snapshot.revision == 0
    )

    await rt.registry.set_consent(
        consent.model_copy(update={"allow_external_ai": True})
    )
    await _say(rt, alice.person_id, "我养了只猫", 3)
    fresh_prompt = client.requests("analysis")[1]["messages"][1]["content"]
    assert "话题: 旅行" not in fresh_prompt  # no stale summary after regrant
    assert "话题: UNKNOWN" in fresh_prompt
    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "养猫"
    assert snapshot.state.topic_stack == []


async def test_non_persistence_turn_never_writes_scene_or_analysis(db):
    rt, client = _runtime(
        db,
        [(_analysis("旅行", scene=_scene(topic="旅行", evidence="我去了日本")), "日本怎么样？")],
    )
    alice = await _person(rt, persistence=False)
    report, event = await _say(rt, alice.person_id, "我去了日本", 1)
    assert report.action == CognitiveAction.RESPOND
    assert len(client.calls) == 2
    assert await _snapshot(rt, alice.person_id) is None
    assert await rt.repos.analysis.get_analysis(event.event_id) is None


# -- G. provider-boundary validation and prompt sections --------------------


async def test_provider_drops_scene_claim_without_verbatim_evidence(db):
    rt, client = _runtime(
        db,
        [
            (
                _analysis(
                    "模型声称用户在办公室",
                    scene=_scene(
                        topic="工作",
                        evidence="我同事在旁边",
                        environment={
                            "location": "办公室",
                            "other_people_present": "YES",
                            "evidence_span": "我同事在旁边",
                        },
                    ),
                ),
                "嗯。",
            )
        ],
    )
    alice = await _person(rt)
    report, event = await _say(rt, alice.person_id, "今天有点累", 1)
    assert report.status == "COMPLETED"

    analysis = await rt.repos.analysis.get_analysis(event.event_id)
    assert analysis.scene_proposal is not None
    assert analysis.scene_proposal.topic is None
    assert analysis.scene_proposal.environment is None
    assert analysis.scene_proposal.topic_change == "UNCERTAIN"
    snapshot = await _snapshot(rt, alice.person_id)
    assert snapshot.state.topic == "UNKNOWN"
    assert snapshot.state.environment["location"] == "UNKNOWN"


async def test_style_example_is_not_injected_per_turn(db):
    # V0.5 removed per-turn style-example retrieval from the default chain: an
    # admin reference is stored data, not something every response sees.
    rt, client = _runtime(
        db,
        [
            (
                _analysis(
                    "技术问题",
                    scene=_scene(topic="调试", evidence="支付回调失败了"),
                ),
                "先看日志。",
            )
        ],
    )
    alice = await _person(rt)
    manifest = await rt.agent_service.repo.get_manifest()
    await rt.style_repo.add(
        agent_id=manifest["agent_id"],
        scenario="技术排查",
        user_text="支付回调失败了",
        ideal_response="先确认回调有没有到网关，再看签名。",
        person_id=alice.person_id,
    )
    await _say(rt, alice.person_id, "支付回调失败了", 1)

    response_prompt = client.requests("response")[0]["messages"][0]["content"]
    analysis_prompt = client.requests("analysis")[0]["messages"][1]["content"]
    assert "风格参考" not in response_prompt
    assert "先确认回调有没有到网关，再看签名。" not in response_prompt
    assert "先确认回调有没有到网关，再看签名。" not in analysis_prompt
    turns = await rt.repos.utterance.recent_context("c1", None, 20)
    assert all("先确认回调有没有到网关" not in t.text for t in turns)


async def test_both_feature_flags_off_remove_v04_prompt_sections(db):
    rt, client = _runtime(
        db,
        [
            (
                _analysis(
                    "旅行",
                    scene=_scene(topic="旅行", evidence="我去了日本"),
                ),
                "日本怎么样？",
            )
        ],
    )
    alice = await _person(rt)
    await rt.repos.settings.set("dialogue.scene.enabled", "false")
    await rt.repos.settings.set("dialogue.personality_generation.enabled", "false")
    report, _ = await _say(rt, alice.person_id, "我去了日本", 1)

    assert report.status == "COMPLETED"
    analysis_prompt = client.requests("analysis")[0]["messages"][1]["content"]
    response_prompt = client.requests("response")[0]["messages"][1]["content"]
    assert "【当前场景】" not in analysis_prompt
    assert "【当前场景】" not in response_prompt
    assert "【人格准则】" not in response_prompt
    assert "【对话风格】" not in response_prompt
    assert "【风格参考】" not in response_prompt
    assert await _snapshot(rt, alice.person_id) is None
