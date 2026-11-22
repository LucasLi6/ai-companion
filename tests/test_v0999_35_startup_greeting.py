"""V0.9.9.35 — startup greeting as a natural fresh opening.

The greeting is not a compressed memory lookup or a recap of the previous
conversation. It uses the trusted persona identity, trusted local time and
the real gap inside this conversation only. These tests are offline: a fake
DeepSeek client; no network and no paid API calls.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from companion.cognition.models import CognitiveContext
from companion.domain.enums import (
    EventSource,
    ProcessingStatus,
)
from companion.domain.models import (
    AgentAction,
    CognitiveResult,
    make_utterance_event,
    new_id,
    utcnow,
)
from companion.domain.enums import CognitiveAction, ReasonCode
from companion.prompts.proactive import (
    PROACTIVE_MODE_BLOCK,
    PROACTIVE_MODE_BLOCK_LONG_IDLE,
)
from companion.prompts.response import (
    build_proactive_response_messages,
    build_startup_greeting_messages,
)
from companion.providers.deepseek_client import ChatResponse
from companion.runtime import build_runtime
from companion.runtime_context.timezone import TimeZoneService


class _GreetingClient:
    def __init__(self, text: str = "随便说一句，窗外挺安静的。") -> None:
        self.text = text
        self.calls: list[dict] = []

    async def complete(self, messages, *, request_type, event_id=None, **kwargs):
        self.calls.append(
            {"request_type": request_type, "messages": messages}
        )
        return ChatResponse(
            content=self.text,
            model="fake",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


async def _runtime(db, client, *, timezone_name=None):
    rt = build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        with_worker=False,
        router_mode="legacy",
    )
    person = await rt.registry.register(
        "Alice",
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )
    await rt.conversations.get_or_create("c1")
    if timezone_name is not None:
        await rt.repos.settings.set("runtime.timezone", timezone_name)
    return rt, person


async def _seed(
    rt,
    person_id: str,
    *,
    text: str,
    ago: timedelta,
    reply: str | None = None,
) -> str:
    occurred = utcnow() - ago
    event = make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        speaker_label="Alice",
        source=EventSource.TEST,
        occurred_at=occurred,
    ).model_copy(update={"processing_status": ProcessingStatus.COMPLETED})
    await rt.repos.utterance.save(event)
    if reply is not None:
        await rt.repos.analysis.save_action(
            AgentAction(
                action_id=new_id("act"),
                event_id=event.event_id,
                action_type=CognitiveAction.RESPOND,
                response_text=reply,
                reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
                status="COMPLETED",
                created_at=occurred,
            )
        )
    return event.event_id


def _system(client: _GreetingClient) -> str:
    return client.calls[0]["messages"][0]["content"]


async def test_no_history_greeting_uses_only_identity_and_time(db) -> None:
    client = _GreetingClient()
    rt, person = await _runtime(db, client)

    async def _must_not_retrieve(*_args, **_kwargs):
        raise AssertionError("startup greeting must not retrieve memories")

    assert rt.orchestrator.memory is not None
    rt.orchestrator.memory.retrieve = _must_not_retrieve  # type: ignore[method-assign]

    result = await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    assert result is not None
    assert len(client.calls) == 1
    system = _system(client)
    assert "【Ava 启动开场白】" in system
    assert "【角色身份】" in system
    assert "【本次开场白的可信时间信息】" in system
    assert "上次真实交流：没有历史记录。" in system
    # No normal Speaker background, memory, scene, tools or routing.
    for absent in (
        "【本轮生成模式：主动发言】",
        "【本轮生成模式：长时间安静后的主动发言】",
        "【本地记忆检索】",
        "【对话历史】",
        "【当前相关的上下文】",
        "【本轮语义路由】",
        "【本轮工具状态】",
        "【你实际的长期记忆样本】",
    ):
        assert absent not in system
    assert len(client.calls[0]["messages"]) == 1

    # The synthetic event is never stored as a user turn.
    turns = await rt.repos.utterance.recent_context(
        "c1", exclude_event_id=None, limit=20
    )
    assert turns == []


async def test_minutes_gap_includes_only_the_last_untrusted_exchange(db) -> None:
    client = _GreetingClient()
    rt, person = await _runtime(db, client)
    await _seed(
        rt,
        person.person_id,
        text="我们刚才说到雨",
        ago=timedelta(minutes=2),
        reply="嗯，雨声听起来很舒服。",
    )

    result = await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    assert result is not None
    system = _system(client)
    assert "距离上次真实交流：2 分钟（连续交流）" in system
    history = [
        m for m in client.calls[0]["messages"] if m["role"] == "user"
    ]
    assert len(history) == 1
    body = history[0]["content"]
    assert "不可信" in body
    assert "我们刚才说到雨" in body
    assert "嗯，雨声听起来很舒服。" in body


async def test_hours_gap_does_not_resume_old_topics(db) -> None:
    client = _GreetingClient()
    rt, person = await _runtime(db, client)
    await _seed(
        rt,
        person.person_id,
        text="上次那个旧话题",
        ago=timedelta(hours=3),
        reply="嗯，我记得。",
    )

    await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    system = _system(client)
    assert "距离上次真实交流：3 小时（几小时间隔）" in system
    assert "上次那个旧话题" not in system
    assert "嗯，我记得。" not in system
    assert len(client.calls[0]["messages"]) == 1


async def test_day_gap_is_a_new_exchange_stage(db) -> None:
    client = _GreetingClient()
    rt, person = await _runtime(db, client)
    await _seed(
        rt,
        person.person_id,
        text="昨天的问题",
        ago=timedelta(hours=25),
        reply="昨天的回答。",
    )

    await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    system = _system(client)
    assert "距离上次真实交流：1 天 1 小时（新的交流阶段）" in system
    assert "昨天的问题" not in system
    assert "昨天的回答。" not in system
    assert len(client.calls[0]["messages"]) == 1


async def test_prompt_carries_trusted_local_time_and_day_phase(db) -> None:
    client = _GreetingClient()
    rt, person = await _runtime(db, client, timezone_name="Asia/Shanghai")

    await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    system = _system(client)
    assert "当前时区未确定" not in system
    assert "本地时间：" in system
    assert "Asia/Shanghai" in system
    local_now = datetime.now(timezone.utc).astimezone(
        ZoneInfo("Asia/Shanghai")
    )
    assert f"（Asia/Shanghai，{_day_phase(local_now)}）" in system


def _day_phase(moment: datetime) -> str:
    hour = moment.hour
    if hour < 5:
        return "凌晨"
    if hour < 11:
        return "早晨"
    if hour < 13:
        return "中午"
    if hour < 18:
        return "下午"
    if hour < 23:
        return "晚上"
    return "深夜"


async def test_unknown_timezone_never_presents_utc_as_local(db) -> None:
    client = _GreetingClient()
    rt, person = await _runtime(db, client)
    rt.orchestrator.runtime_snapshot_builder.timezone_service = (
        TimeZoneService(system_zone="Invalid/Zone")
    )

    await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    system = _system(client)
    assert "当前时区未确定" in system
    assert "本地时间：" not in system


async def test_exactly_one_call_and_no_proactive_or_router_path(db) -> None:
    client = _GreetingClient()
    rt, person = await _runtime(db, client)

    async def _must_not_call(*_args, **_kwargs):
        raise AssertionError("startup greeting must not use proactive generation")

    rt.orchestrator.response.generate_proactive = _must_not_call  # type: ignore[method-assign]

    result = await rt.orchestrator.generate_startup_greeting(
        conversation_id="c1", person_id=person.person_id
    )

    assert result is not None
    assert result["text"] == client.text
    assert result["display_text"] is None
    assert len(client.calls) == 1
    assert client.calls[0]["request_type"] == "response"


def test_normal_proactive_prompt_construction_is_unchanged() -> None:
    context = CognitiveContext(
        conversation_id="c1",
        current_event_id="e1",
        current_speaker_id="p1",
        current_speaker_label="Alice",
        current_text="hi",
        current_addressee_hint="agent",
    )
    analysis = CognitiveResult(
        event_id="e1",
        meaning="test",
        action=CognitiveAction.RESPOND,
        reason_code=ReasonCode.DIRECT_AGENT_ADDRESS,
    )

    immediate = build_proactive_response_messages(
        context, analysis, mode="IMMEDIATE"
    )[0]["content"]
    long_idle = build_proactive_response_messages(
        context, analysis, mode="LONG_IDLE"
    )[0]["content"]
    legacy_opening = build_proactive_response_messages(
        context, analysis, mode="OPENING"
    )[0]["content"]

    assert PROACTIVE_MODE_BLOCK in immediate
    assert PROACTIVE_MODE_BLOCK_LONG_IDLE in long_idle
    assert PROACTIVE_MODE_BLOCK in legacy_opening
    assert "【Ava 启动开场白】" not in immediate
    assert "【Ava 启动开场白】" not in long_idle


def test_startup_builder_is_standalone() -> None:
    """A direct builder call sends only identity, time and optional history."""
    from companion.cognition.models import CognitiveContext as Context

    messages = build_startup_greeting_messages(
        Context(
            conversation_id="c1",
            current_event_id="e1",
            current_speaker_id="p1",
            current_speaker_label="Alice",
            current_text="（启动开场白）",
            current_addressee_hint="agent",
        ),
        {
            "timezone_known": False,
            "gap_seconds": None,
            "gap_text": None,
            "continuity_label": "尚无历史",
            "last_exchange": [],
        },
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    assert "【Ava 启动开场白】" in system
    assert "上次真实交流：没有历史记录。" in system
    assert "【本地记忆检索】" not in system
