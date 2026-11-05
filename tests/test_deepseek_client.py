"""Phase 1 — DeepSeek API client tests using a fake AsyncOpenAI-compatible SDK.

No network and no real ``openai`` package is required: the client is
constructed with an injected fake ``chat.completions.create`` handler, exactly
as the spec asks ("Fake HTTP Transport / Mock SDK Client").
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from companion.providers.deepseek_client import (
    ChatResponse,
    DeepSeekAuthError,
    DeepSeekConfig,
    DeepSeekConfigError,
    DeepSeekClient,
    DeepSeekInvalidRequestError,
    DeepSeekResponseError,
    DeepSeekServerError,
    DeepSeekTimeoutError,
    build_deepseek_client,
    load_deepseek_config,
)


# -- fake SDK ---------------------------------------------------------------


class FakeResponse:
    def __init__(self, content, *, prompt=1, completion=2, model="deepseek-flash", finish="stop"):
        self._content = content
        self._prompt = prompt
        self._completion = completion
        self._model = model
        self._finish = finish

    @property
    def choices(self):
        msg = SimpleNamespace(content=self._content, finish_reason=self._finish)
        return [SimpleNamespace(message=msg, finish_reason=self._finish)]

    @property
    def usage(self):
        return SimpleNamespace(
            prompt_tokens=self._prompt,
            completion_tokens=self._completion,
            total_tokens=self._prompt + self._completion,
        )

    @property
    def model(self):
        return self._model


class FakeError(Exception):
    def __init__(self, status_code=None, name=None):
        super().__init__("boom")
        self.status_code = status_code
        self._name = name

    def __repr__(self):
        return self._name or type(self).__name__


class _TimeoutError(Exception):
    """Named like the SDK's APITimeoutError for duck-typed classification."""


class FakeClient:
    """Injected client whose ``create`` delegates to ``handler(**kwargs)``."""

    def __init__(self, handler):
        self._handler = handler
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls = []

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._handler(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result


async def _ok(**kwargs):
    return FakeResponse("hello")


def _make_client(config, handler):
    fake = FakeClient(handler)
    client = DeepSeekClient(config, client=fake)
    return client, fake


def _config(**overrides) -> DeepSeekConfig:
    values = dict(api_key="test-key", max_retries=2, retry_backoff_seconds=0.0)
    values.update(overrides)
    return DeepSeekConfig(**values)


# -- config -----------------------------------------------------------------


def test_load_config_defaults():
    cfg = load_deepseek_config({})
    assert cfg.api_key == ""
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.model == "deepseek-flash"
    assert cfg.timeout_seconds == 60.0
    assert cfg.max_retries == 2
    assert cfg.analysis_max_tokens == 1024
    assert cfg.response_max_tokens == 2048
    assert cfg.analysis_thinking == "disabled"
    assert cfg.response_thinking == "disabled"
    assert cfg.analysis_reasoning_effort == "none"
    assert cfg.response_reasoning_effort == "none"


def test_load_config_reads_env():
    cfg = load_deepseek_config(
        {
            "DEEPSEEK_API_KEY": "sk-abc",
            "DEEPSEEK_BASE_URL": "https://example.test/v1",
            "DEEPSEEK_MODEL": "deepseek-reasoner",
            "DEEPSEEK_TIMEOUT": "30",
            "DEEPSEEK_MAX_RETRIES": "5",
            "DEEPSEEK_ANALYSIS_MAX_TOKENS": "512",
            "DEEPSEEK_RESPONSE_MAX_TOKENS": "4096",
            "DEEPSEEK_THINKING": "enabled",
            "DEEPSEEK_ANALYSIS_REASONING_EFFORT": "high",
        }
    )
    assert cfg.api_key == "sk-abc"
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.model == "deepseek-reasoner"
    assert cfg.timeout_seconds == 30.0
    assert cfg.max_retries == 5
    assert cfg.analysis_max_tokens == 512
    assert cfg.response_max_tokens == 4096
    assert cfg.analysis_thinking == "enabled"
    assert cfg.response_thinking == "enabled"
    assert cfg.analysis_reasoning_effort == "high"
    assert cfg.response_reasoning_effort == "none"


def test_build_client_requires_key():
    with pytest.raises(DeepSeekConfigError):
        build_deepseek_client(DeepSeekConfig(api_key=""))


# -- success path -----------------------------------------------------------


async def test_complete_parses_response():
    client, fake = _make_client(_config(), _ok)
    resp = await client.complete(
        [{"role": "user", "content": "hi"}], request_type="analysis"
    )
    assert isinstance(resp, ChatResponse)
    assert resp.content == "hello"
    assert resp.prompt_tokens == 1
    assert resp.completion_tokens == 2
    assert resp.total_tokens == 3
    # one request only
    assert len(fake.calls) == 1


async def test_analysis_defaults_json_mode_and_tokens():
    client, fake = _make_client(
        _config(analysis_max_tokens=512), _ok
    )
    await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    kwargs = fake.calls[0]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["max_tokens"] == 512
    assert kwargs["model"] == "deepseek-flash"


async def test_response_defaults_plain_and_tokens():
    client, fake = _make_client(
        _config(response_max_tokens=2048), _ok
    )
    await client.complete([{"role": "user", "content": "x"}], request_type="response")
    kwargs = fake.calls[0]
    assert "response_format" not in kwargs
    assert kwargs["max_tokens"] == 2048


async def test_thinking_mode_is_off_by_default_on_both_requests():
    client, fake = _make_client(_config(), _ok)
    await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    await client.complete([{"role": "user", "content": "x"}], request_type="response")
    # The V4 API defaults to thinking ON, so off is an explicit wire switch.
    assert fake.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert fake.calls[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    # "none" is not an effort value; disabled thinking is expressed by the
    # switch alone, never by a contradictory effort.
    assert "reasoning_effort" not in fake.calls[0]
    assert "reasoning_effort" not in fake.calls[1]


async def test_thinking_enabled_sends_switch_and_effort():
    client, fake = _make_client(
        _config(
            analysis_thinking="enabled",
            response_thinking="enabled",
            analysis_reasoning_effort="max",
            response_reasoning_effort="high",
        ),
        _ok,
    )
    await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    await client.complete([{"role": "user", "content": "x"}], request_type="response")
    assert fake.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert fake.calls[0]["reasoning_effort"] == "max"
    assert fake.calls[1]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert fake.calls[1]["reasoning_effort"] == "high"


async def test_thinking_enabled_with_unset_effort_uses_provider_default():
    client, fake = _make_client(
        _config(response_thinking="enabled"), _ok
    )
    await client.complete([{"role": "user", "content": "x"}], request_type="response")
    assert fake.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in fake.calls[0]


async def test_thinking_can_be_enabled_for_one_request_type_only():
    cfg = load_deepseek_config({"DEEPSEEK_RESPONSE_THINKING": "enabled"})
    assert cfg.analysis_thinking == "disabled"
    assert cfg.response_thinking == "enabled"
    cfg = load_deepseek_config({"DEEPSEEK_THINKING": "on"})
    assert cfg.analysis_thinking == "enabled"
    assert cfg.response_thinking == "enabled"


async def test_missing_usage_zeroes_tokens():
    class NoUsage(FakeResponse):
        @property
        def usage(self):
            return None

    client, _ = _make_client(_config(), lambda **kw: NoUsage("hi"))
    resp = await client.complete([{"role": "user", "content": "x"}], request_type="response")
    assert resp.prompt_tokens == 0
    assert resp.completion_tokens == 0
    assert resp.total_tokens == 0


# -- error handling ---------------------------------------------------------


async def test_fails_fast_on_auth():
    client, fake = _make_client(_config(), lambda **kw: (_ for _ in ()).throw(FakeError(status_code=401)))
    with pytest.raises(DeepSeekAuthError):
        await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    assert len(fake.calls) == 1  # no retry


async def test_fails_fast_on_insufficient_balance():
    # 402 is an account/billing problem, not transient: never retry.
    client, fake = _make_client(
        _config(max_retries=2),
        lambda **kw: (_ for _ in ()).throw(FakeError(status_code=402)),
    )
    with pytest.raises(DeepSeekAuthError):
        await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    assert len(fake.calls) == 1


async def test_fails_fast_on_400():
    client, fake = _make_client(
        _config(), lambda **kw: (_ for _ in ()).throw(FakeError(status_code=400))
    )
    with pytest.raises(DeepSeekInvalidRequestError):
        await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    assert len(fake.calls) == 1


async def test_retries_then_succeeds():
    attempts = {"n": 0}

    async def flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FakeError(status_code=429)
        return FakeResponse("finally")

    client, fake = _make_client(_config(max_retries=2), flaky)
    resp = await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    assert resp.content == "finally"
    assert attempts["n"] == 3


async def test_gives_up_after_max_retries():
    client, fake = _make_client(
        _config(max_retries=2),
        lambda **kw: (_ for _ in ()).throw(FakeError(status_code=503)),
    )
    with pytest.raises(DeepSeekServerError):
        await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    assert len(fake.calls) == 3  # initial + 2 retries


async def test_timeout_classified_retryable():
    client, fake = _make_client(
        _config(max_retries=0),
        lambda **kw: (_ for _ in ()).throw(_TimeoutError()),
    )
    with pytest.raises(DeepSeekTimeoutError):
        await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    assert len(fake.calls) == 1


async def test_empty_content_raises_response_error():
    client, fake = _make_client(
        _config(max_retries=0), lambda **kw: FakeResponse("   ")
    )
    with pytest.raises(DeepSeekResponseError):
        await client.complete([{"role": "user", "content": "x"}], request_type="analysis")
    assert len(fake.calls) == 1


async def test_no_client_raises_config_error():
    client = DeepSeekClient(_config(api_key=""))
    with pytest.raises(DeepSeekConfigError):
        await client.complete([{"role": "user", "content": "x"}], request_type="analysis")


# -- logging hygiene --------------------------------------------------------


async def test_audit_log_never_contains_content_or_key(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="companion.deepseek")
    client, _ = _make_client(_config(api_key="sk-super-secret"), _ok)
    await client.complete(
        [{"role": "user", "content": "PRIVATE-CONTENT-SENTINEL"}],
        request_type="analysis",
        event_id="evt_1",
    )
    text = caplog.text
    assert "PRIVATE-CONTENT-SENTINEL" not in text
    assert "sk-super-secret" not in text
    assert "provider=deepseek" in text
    assert "request_type=analysis" in text
    assert "success=true" in text
