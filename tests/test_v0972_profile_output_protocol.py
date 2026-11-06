"""V0.9.7.2 follow-up: the model-output -> ProfileReviewDecision protocol.

The real model is allowed to be wrong, but it is never allowed to be
ambiguous. One dedicated format-repair call is permitted; after that the
review fails through the normal job retry path. Invalid output must never
become ``NO_CHANGE``, and nothing private may enter the diagnostics.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from companion.people.profile_refinement import (
    ProfileRefinementValidationError,
    ProfileReviewDecision,
)
from companion.prompts.profile_refinement import (
    PROFILE_NO_CHANGE_EXAMPLE,
    PROFILE_PROPOSE_CHANGES_EXAMPLE,
    PROFILE_REFINEMENT_SYSTEM_PROMPT,
    build_profile_refinement_messages,
)
from companion.providers.deepseek_profile import (
    DeepSeekProfileRefinementProvider,
)

PROMPT_VERSION = "profile-refinement-v2"
PAYLOAD = {"person_id": "person_trusted", "profile_revision": 4}
LOG_NAME = "companion.profile_refinement.provider"


class FakeClient:
    """Replay (content, finish_reason) pairs and record every request."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs):
        self.calls.append(messages)
        item = self.outputs.pop(0)
        content, finish_reason = item if isinstance(item, tuple) else (item, "stop")
        return SimpleNamespace(
            content=content, model="fake-model", finish_reason=finish_reason
        )


def _valid_no_change(reason="没有稳定新信息"):
    return json.dumps(
        {"decision": "NO_CHANGE", "reason": reason, "changes": []},
        ensure_ascii=False,
    )


def _valid_proposal(content="正在开发机器狗", quote="我在做机器狗"):
    return json.dumps(
        {
            "decision": "PROPOSE_CHANGES",
            "reason": "本人自述",
            "changes": [
                {
                    "operation": "ADD",
                    "field_key": "projects.active",
                    "target_entry_ids": [],
                    "proposed_content": content,
                    "evidence_event_ids": ["evt_allowed"],
                    "evidence_quotes": [quote],
                    "reason": "明确自述",
                    "confidence": 0.9,
                    "requires_confirmation": False,
                }
            ],
        },
        ensure_ascii=False,
    )


def _provider(outputs):
    client = FakeClient(outputs)
    return DeepSeekProfileRefinementProvider(client), client


def test_prompt_examples_are_real_json_and_match_the_decision_schema():
    for example in (PROFILE_NO_CHANGE_EXAMPLE, PROFILE_PROPOSE_CHANGES_EXAMPLE):
        data = json.loads(example)  # must never be pseudo-JSON
        data["person_id"] = "person_trusted"
        data["base_revision"] = 1
        ProfileReviewDecision.model_validate(data)

    assert PROFILE_NO_CHANGE_EXAMPLE in PROFILE_REFINEMENT_SYSTEM_PROMPT
    assert PROFILE_PROPOSE_CHANGES_EXAMPLE in PROFILE_REFINEMENT_SYSTEM_PROMPT
    assert "不得使用" in PROFILE_REFINEMENT_SYSTEM_PROMPT
    for alternative in ("action", "updates", "proposals"):
        assert alternative in PROFILE_REFINEMENT_SYSTEM_PROMPT  # named as banned


async def test_a_valid_output_needs_no_repair(caplog):
    provider, client = _provider([_valid_no_change()])
    caplog.set_level(logging.INFO, logger=LOG_NAME)

    decision = await provider.review(PAYLOAD)

    assert len(client.calls) == 1
    assert decision.decision == "NO_CHANGE"
    assert decision.person_id == "person_trusted"
    assert decision.base_revision == 4
    log = "\n".join(record.getMessage() for record in caplog.records)
    assert "fields=changes,decision,reason" in log
    assert "repaired=False ok=True" in log
    assert f"prompt_version={PROMPT_VERSION}" in log


async def test_decision_change_with_valid_changes_is_normalised():
    payload = json.loads(_valid_proposal())
    payload["decision"] = "CHANGE"
    provider, client = _provider([json.dumps(payload, ensure_ascii=False)])

    decision = await provider.review(PAYLOAD)

    assert len(client.calls) == 1
    assert decision.decision == "PROPOSE_CHANGES"
    assert decision.changes[0].field_key == "projects.active"
    assert decision.changes[0].proposed_content == "正在开发机器狗"


async def test_decision_change_without_changes_is_repaired():
    provider, client = _provider(
        [
            json.dumps({"decision": "CHANGE", "reason": "x", "changes": []}),
            _valid_no_change(),
        ]
    )

    decision = await provider.review(PAYLOAD)

    assert len(client.calls) == 2
    assert decision.decision == "NO_CHANGE"
    repair_turn = client.calls[1][-1]
    assert "错误类型：change_without_changes" in repair_turn["content"]


async def test_no_change_with_changes_is_never_silently_dropped():
    contradictory = json.dumps(json.loads(_valid_proposal()) | {
        "decision": "NO_CHANGE"
    })
    provider, client = _provider(
        [contradictory, json.dumps(json.loads(contradictory) | {
            "decision": "NO_CHANGE", "changes": []
        })]
    )

    decision = await provider.review(PAYLOAD)

    assert decision.decision == "NO_CHANGE"
    assert decision.changes == []


async def test_malformed_json_gets_one_repair_call_without_echoing_output():
    provider, client = _provider(["not json at all", _valid_no_change()])

    decision = await provider.review(PAYLOAD)

    assert len(client.calls) == 2
    assert decision.decision == "NO_CHANGE"
    # The repair call carries the original task plus an error type, never the
    # previous raw output.
    sent = "\n".join(str(message.get("content", "")) for message in client.calls[1])
    assert "not json at all" not in sent
    assert "malformed_json" in sent
    assert client.calls[0] == build_profile_refinement_messages(PAYLOAD)


async def test_truncated_output_is_recognised_and_repaired(caplog):
    provider, client = _provider(
        [('{"decision": "NO_CHANGE", "reason": "x"', "length"), _valid_no_change()]
    )
    caplog.set_level(logging.INFO, logger=LOG_NAME)

    decision = await provider.review(PAYLOAD)

    assert len(client.calls) == 2
    assert decision.decision == "NO_CHANGE"
    log = "\n".join(record.getMessage() for record in caplog.records)
    assert "error=truncated" in log
    assert "repaired=True ok=True" in log


async def test_two_invalid_outputs_fail_instead_of_becoming_no_change():
    provider, client = _provider(
        [
            json.dumps({"decision": "CHANGE", "changes": []}),
            json.dumps({"decision": "MAYBE", "changes": []}),
        ]
    )

    with pytest.raises(ProfileRefinementValidationError) as excinfo:
        await provider.review(PAYLOAD)

    assert len(client.calls) == 2
    assert "invalid_decision" in str(excinfo.value)


async def test_missing_decision_and_alternative_fields_are_repaired():
    provider, client = _provider(
        [
            json.dumps({"action": "NO_CHANGE", "proposals": []}),
            _valid_no_change(),
        ]
    )

    decision = await provider.review(PAYLOAD)

    assert len(client.calls) == 2
    assert decision.decision == "NO_CHANGE"
    assert "missing_decision" in client.calls[1][-1]["content"]


async def test_broken_change_schema_is_repaired_not_invented():
    bad = json.loads(_valid_proposal())
    bad["changes"][0]["confidence"] = 5  # outside the allowed range
    provider, client = _provider([json.dumps(bad), _valid_proposal()])

    decision = await provider.review(PAYLOAD)

    assert len(client.calls) == 2
    assert decision.decision == "PROPOSE_CHANGES"
    assert decision.changes[0].confidence == 0.9


async def test_diagnostics_never_contain_private_profile_text(caplog):
    secret = "我的秘密代号是蓝鲸"
    provider, client = _provider([_valid_proposal(content=secret, quote=secret)])
    caplog.set_level(logging.INFO, logger=LOG_NAME)

    decision = await provider.review(PAYLOAD)

    assert decision.changes[0].proposed_content == secret
    log = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in log
    assert "我的秘密代号" not in log
    assert "fields=changes,decision,reason" in log
