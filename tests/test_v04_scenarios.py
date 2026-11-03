"""V0.4 — 40 synthetic three-turn continuity scenarios.

Each case is a complete multi-turn scene: propose, persist, reload, propose
again. The scenarios are grouped into ten categories (casual chat, jokes/play,
technical work, interruption, topic switching, returning, correction,
environment, open loops and affect). All text is synthetic; no model or network
is involved. The point is breadth: the same trusted merge rules must hold for
ordinary chat and for awkward boundaries.
"""

from __future__ import annotations

import pytest

from companion.cognition.scene_manager import SceneManager
from companion.domain.enums import CognitiveAction, EventSource
from companion.domain.models import make_utterance_event

# -- scenario construction --------------------------------------------------


def _p(
    text,
    *,
    change="CONTINUE",
    topic=None,
    mode="CONVERSATION",
    phase="EXPLORING",
    tone="WARM",
    env=None,
    loop=None,
    resolve=None,
):
    return {
        "topic_change": change,
        "topic": topic,
        "mode": mode,
        "discourse_phase": phase,
        "tone": tone,
        "environment": env,
        "open_loop": loop,
        "resolve_loop_topics": resolve or [],
        "confidence": 0.85,
        "evidence_span": text if topic is not None else None,
    }


def _t(text, proposal, *, action=CognitiveAction.RESPOND, pending=False):
    return {
        "text": text,
        "proposal": proposal,
        "action": action,
        "pending": pending,
    }


def _case(category, index, turns, **expected):
    expected.setdefault("stack", [])
    expected.setdefault("loops", 0)
    expected.setdefault("mode", "CONVERSATION")
    expected.setdefault("phase", "EXPLORING")
    return pytest.param(
        category, index, turns, expected, id=f"{category}-{index}"
    )


SCENARIOS: list = []

# 1. Casual chat: shift, continue, continue.
for index, topic in enumerate(["旅行", "美食", "电影", "音乐"], 1):
    SCENARIOS.append(
        _case(
            "casual",
            index,
            [
                _t(f"最近想聊{topic}", _p(f"最近想聊{topic}", change="SHIFT", topic=topic)),
                _t(f"{topic}挺有意思", _p(f"{topic}挺有意思", topic=topic)),
                _t(f"再说说{topic}", _p(f"再说说{topic}", topic=topic)),
            ],
            topic=topic,
            tone="WARM",
        )
    )

# 2. Jokes / play: the tone and mode stay playful without inventing a topic.
for index, topic in enumerate(["游戏", "冷笑话", "梗图", "宠物"], 1):
    SCENARIOS.append(
        _case(
            "playful",
            index,
            [
                _t(
                    f"开个玩笑，说说{topic}",
                    _p(
                        f"开个玩笑，说说{topic}",
                        change="SHIFT",
                        topic=topic,
                        mode="PLAY",
                        tone="PLAYFUL",
                    ),
                ),
                _t(
                    f"{topic}这个梗好笑",
                    _p(
                        f"{topic}这个梗好笑",
                        topic=topic,
                        mode="PLAY",
                        tone="PLAYFUL",
                    ),
                ),
                _t(
                    f"继续逗我聊{topic}",
                    _p(
                        f"继续逗我聊{topic}",
                        topic=topic,
                        mode="PLAY",
                        tone="PLAYFUL",
                    ),
                ),
            ],
            topic=topic,
            tone="PLAYFUL",
            mode="PLAY",
        )
    )

# 3. Technical continuity: task mode and a serious, deepening phase.
for index, topic in enumerate(
    ["支付回调", "数据库迁移", "向量检索", "并发锁"], 1
):
    SCENARIOS.append(
        _case(
            "technical",
            index,
            [
                _t(
                    f"{topic}一直失败，帮我看看",
                    _p(
                        f"{topic}一直失败，帮我看看",
                        change="SHIFT",
                        topic=topic,
                        mode="TASK",
                        phase="DEEPENING",
                        tone="SERIOUS",
                    ),
                ),
                _t(
                    f"继续排查{topic}",
                    _p(
                        f"继续排查{topic}",
                        topic=topic,
                        mode="TASK",
                        phase="DEEPENING",
                        tone="SERIOUS",
                    ),
                ),
                _t(
                    f"{topic}的日志还是空的",
                    _p(
                        f"{topic}的日志还是空的",
                        topic=topic,
                        mode="TASK",
                        phase="DEEPENING",
                        tone="SERIOUS",
                    ),
                ),
            ],
            topic=topic,
            tone="SERIOUS",
            mode="TASK",
            phase="DEEPENING",
        )
    )

# 4. Interruption: a WAIT with no new topic must not erase the old scene.
for index, topic in enumerate(["旅行", "摄影", "烘焙", "徒步"], 1):
    SCENARIOS.append(
        _case(
            "interruption",
            index,
            [
                _t(f"最近在聊{topic}", _p(f"最近在聊{topic}", change="SHIFT", topic=topic)),
                _t(
                    "嗯？",
                    _p("嗯？", change="CONTINUE", topic=None, tone="NEUTRAL"),
                    action=CognitiveAction.WAIT,
                ),
                _t(f"回到{topic}", _p(f"回到{topic}", topic=topic)),
            ],
            topic=topic,
            tone="WARM",
        )
    )

# 5. Topic switching: the previous topic is retained on the bounded stack.
for index, topic in enumerate(["旅行", "工作", "健康", "学习"], 1):
    first = f"{topic}入门"
    second = f"{topic}进阶"
    SCENARIOS.append(
        _case(
            "switch",
            index,
            [
                _t(f"想聊{first}", _p(f"想聊{first}", change="SHIFT", topic=first)),
                _t(f"换成{second}", _p(f"换成{second}", change="SHIFT", topic=second)),
                _t(f"继续{second}", _p(f"继续{second}", topic=second)),
            ],
            topic=second,
            stack=[first],
            tone="WARM",
        )
    )

# 6. Returning: an evidenced earlier topic is restored, not invented.
for index, topic in enumerate(["旅行", "工作", "健康", "学习"], 1):
    first = f"{topic}旧话题"
    second = f"{topic}新话题"
    SCENARIOS.append(
        _case(
            "return",
            index,
            [
                _t(f"先聊{first}", _p(f"先聊{first}", change="SHIFT", topic=first)),
                _t(f"再聊{second}", _p(f"再聊{second}", change="SHIFT", topic=second)),
                _t(
                    f"还是说回{first}",
                    _p(f"还是说回{first}", change="RETURN", topic=first),
                ),
            ],
            topic=first,
            stack=[],
            tone="WARM",
        )
    )

# 7. Correction: same-topic correction updates evidence without pushing itself.
for index, city in enumerate(["北京", "上海", "广州", "成都"], 1):
    SCENARIOS.append(
        _case(
            "correction",
            index,
            [
                _t(f"我住在{city}", _p(f"我住在{city}", change="SHIFT", topic="居住地")),
                _t(
                    f"其实我搬到{city}了",
                    _p(f"其实我搬到{city}了", change="SHIFT", topic="居住地"),
                ),
                _t("还是住在那", _p("还是住在那", topic="居住地")),
            ],
            topic="居住地",
            stack=[],
            tone="WARM",
        )
    )

# 8. Environment: explicit evidence is kept and ages without being re-inferred.
for index, (presence, place, text) in enumerate(
    [
        ("NO", "家里", "我一个人在家里"),
        ("YES", "办公室", "我在办公室，同事在旁边"),
        ("NO", "房间里", "我一个人在房间里"),
        ("YES", "客厅", "我在客厅，朋友在旁边"),
    ],
    1,
):
    SCENARIOS.append(
        _case(
            "environment",
            index,
            [
                _t(
                    text,
                    _p(
                        text,
                        change="SHIFT",
                        topic="此刻",
                        env={
                            "location": place,
                            "other_people_present": presence,
                            "evidence_span": text,
                        },
                    ),
                ),
                _t("嗯", None),
                _t("好", None),
            ],
            topic="此刻",
            tone="WARM",
            environment_location=place,
            environment_presence=presence,
        )
    )

# 9. Open loops: pending follow-up creates one, explicit resolution removes it.
for index in range(1, 5):
    text = f"第{index}件事等会儿说"
    topic = f"未完成事项{index}"
    SCENARIOS.append(
        _case(
            "open_loop",
            index,
            [
                _t(
                    text,
                    _p(
                        text,
                        change="SHIFT",
                        topic=topic,
                        loop={
                            "topic": topic,
                            "summary": f"第{index}件事还没讲完",
                            "evidence_span": text,
                        },
                    ),
                    pending=True,
                ),
                _t(
                    f"第{index}件事说完了",
                    _p(
                        f"第{index}件事说完了",
                        topic=topic,
                        resolve=[topic],
                    ),
                ),
                _t(f"接着聊{index}", _p(f"接着聊{index}", topic=topic)),
            ],
            topic=topic,
            loops=0,
            tone="WARM",
        )
    )

# 10. Affect: support and celebration tones survive while the topic stays put.
for index, tone in enumerate(["SAD", "TENSE", "CELEBRATORY", "NEUTRAL"], 1):
    topic = f"近况{index}"
    mode = "SUPPORT" if tone in ("SAD", "TENSE") else "CONVERSATION"
    SCENARIOS.append(
        _case(
            "affect",
            index,
            [
                _t(
                    f"最近有些事想说，{topic}",
                    _p(
                        f"最近有些事想说，{topic}",
                        change="SHIFT",
                        topic=topic,
                        mode=mode,
                        tone=tone,
                    ),
                ),
                _t(
                    f"{topic}还在继续",
                    _p(
                        f"{topic}还在继续",
                        topic=topic,
                        mode=mode,
                        tone=tone,
                    ),
                ),
                _t(
                    f"{topic}后来好多了",
                    _p(
                        f"{topic}后来好多了",
                        topic=topic,
                        mode=mode,
                        tone=tone,
                    ),
                ),
            ],
            topic=topic,
            tone=tone,
            mode=mode,
        )
    )


@pytest.mark.parametrize("category,index,turns,expected", SCENARIOS)
async def test_three_turn_scene_scenario(db, rt, category, index, turns, expected):
    person = await rt.registry.register(
        f"Person{category}{index}",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=False,
    )
    await rt.conversations.get_or_create("c1")
    manager = SceneManager(repos=rt.repos, registry=rt.registry)

    previous = None
    for turn_number, turn in enumerate(turns, 1):
        event = make_utterance_event(
            conversation_id="c1",
            text=turn["text"],
            addressee_hint="agent",
            speaker_person_id=person.person_id,
            source=EventSource.TEST,
        )
        await rt.repos.utterance.save(event)
        candidate = manager.preview(
            previous,
            turn["proposal"],
            event,
            pending_followup=turn["pending"],
        )
        candidate = manager.finalize(
            candidate, event=event, action=turn["action"]
        )
        expected_revision = previous.revision if previous is not None else 0
        await manager.scene_repo.apply_transition(
            source_event_id=event.event_id,
            conversation_id="c1",
            person_id=person.person_id,
            expected_revision=expected_revision,
            state=candidate,
            transition={
                "action": turn["action"].value,
                "topic": candidate.topic,
                "revision": candidate.revision,
            },
        )
        previous = await manager.load("c1", person.person_id)
        assert previous is not None

    state = previous.state
    assert previous.revision == 2
    assert state.topic == expected["topic"]
    assert state.topic_stack == expected["stack"]
    assert state.tone == expected["tone"]
    assert state.mode == expected["mode"]
    assert state.discourse_phase == expected["phase"]
    assert len(state.open_loops) == expected["loops"]
    if "environment_location" in expected:
        assert state.environment["location"] == expected["environment_location"]
        assert (
            state.environment["other_people_present"]
            == expected["environment_presence"]
        )
        assert state.prompt_dict()["environment"]["location"] == expected[
            "environment_location"
        ]
    else:
        assert state.environment["location"] == "UNKNOWN"
        assert state.environment["other_people_present"] == "UNKNOWN"


def test_scenario_catalog_has_forty_cases():
    assert len(SCENARIOS) == 40
    categories = {param.id.split("-")[0] for param in SCENARIOS}
    assert len(categories) == 10
