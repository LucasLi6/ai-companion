"""DeepSeek router provider: wire parameters, parsing and truthful failures."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from companion.providers.deepseek_client import (
    DeepSeekClient,
    DeepSeekConfig,
    DeepSeekTimeoutError,
)
from companion.providers.deepseek_router import DeepSeekRouterProvider
from companion.routing.errors import RouterInvalidOutput, RouterTimeout
from companion.routing.models import RouteAction, RoutingContext


def routing_context() -> RoutingContext:
    return RoutingContext(
        conversation_id="conv",
        current_event_id="evt",
        current_speaker_id="p1",
        current_speaker_label="Lucas",
        current_text="hello",
        trusted_addressee_hint="agent",
    )


def decision_json(**overrides) -> str:
    data = {
        "schema_version": 1,
        "action": "RESPOND",
        "reply_necessity": "REQUIRED",
        "addressee_assessment": "AGENT",
        "input_semantics": "COMMUNICATIVE",
        "reason": "addressed to Ava",
    }
    data.update(overrides)
    return json.dumps(data)


class _FakeCompletions:
    def __init__(self, content: str = "", error: Exception | None = None):
        self.params: list[dict] = []
        self._content = content
        self._error = error

    async def create(self, **params):
        self.params.append(params)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content),
                    finish_reason="stop",
                )
            ],
            model=params["model"],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )


class _FakeSdk:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def client_for(content: str = "", error=None, **config_overrides):
    completions = _FakeCompletions(content=content, error=error)
    config = DeepSeekConfig(
        api_key="test-key-not-real",
        model=config_overrides.pop("model", "speaker-model"),
        analysis_max_tokens=111,
        response_max_tokens=222,
        max_retries=0,
        **config_overrides,
    )
    client = DeepSeekClient(config, client=_FakeSdk(completions))
    return client, completions


@pytest.mark.asyncio
async def test_router_request_uses_router_model_tokens_json_and_thinking():
    client, completions = client_for(
        decision_json(),
        router_model="router-model",
        router_max_tokens=128,
        router_thinking="disabled",
    )

    provider = DeepSeekRouterProvider(client)
    decision = await provider.route(routing_context())

    assert decision.action == RouteAction.RESPOND
    params = completions.params[0]
    assert params["model"] == "router-model"
    assert params["max_tokens"] == 128
    assert params["response_format"] == {"type": "json_object"}
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "temperature" not in params


@pytest.mark.asyncio
async def test_router_defaults_to_configured_model_when_not_overridden():
    client, completions = client_for(decision_json())
    provider = DeepSeekRouterProvider(client)
    await provider.route(routing_context())
    assert completions.params[0]["model"] == "speaker-model"
    assert completions.params[0]["max_tokens"] == client.config.router_max_tokens


@pytest.mark.asyncio
async def test_speaker_and_analysis_wire_settings_are_unchanged():
    client, completions = client_for(
        "hello", router_model="router-model", router_max_tokens=128
    )

    await client.complete(
        [{"role": "user", "content": "hi"}], request_type="response"
    )
    response_params = completions.params[-1]
    assert response_params["model"] == "speaker-model"
    assert response_params["max_tokens"] == 222
    assert "response_format" not in response_params

    await client.complete(
        [{"role": "user", "content": "hi"}], request_type="analysis"
    )
    analysis_params = completions.params[-1]
    assert analysis_params["model"] == "speaker-model"
    assert analysis_params["max_tokens"] == 111
    assert analysis_params["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_non_json_and_schema_failures_never_fabricate_a_decision():
    client, _ = client_for("I think you should respond.")
    provider = DeepSeekRouterProvider(client)
    with pytest.raises(RouterInvalidOutput):
        await provider.route(routing_context())

    client2, _ = client_for(decision_json(action="MAYBE"))
    with pytest.raises(RouterInvalidOutput):
        await DeepSeekRouterProvider(client2).route(routing_context())


@pytest.mark.asyncio
async def test_timeout_is_reported_as_router_timeout_not_silence():
    client, _ = client_for(error=DeepSeekTimeoutError("timed out"))
    provider = DeepSeekRouterProvider(client)
    with pytest.raises(RouterTimeout):
        await provider.route(routing_context())


@pytest.mark.asyncio
async def test_replan_sends_previous_decision_and_tool_results_as_data():
    client, completions = client_for(decision_json(needs_followup_planning=False))
    provider = DeepSeekRouterProvider(client)
    previous = await provider.route(routing_context())
    await provider.replan(
        routing_context(),
        previous,
        [{"tool_name": "memory.search", "status": "EMPTY", "data": []}],
    )

    payload = json.loads(completions.params[-1]["messages"][-1]["content"])
    assert payload["previous_decision"]["action"] == "RESPOND"
    assert payload["tool_results"][0]["status"] == "EMPTY"
    system = completions.params[-1]["messages"][0]["content"]
    assert "不生成 Ava 的回复正文" in system
