"""V0.9.9.3: multi-turn episode memory and local auto-recall.

All tests are offline: the DeepSeek client is a captured fake, the router is
scripted, and every database is a temporary local SQLite file.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from companion.domain.enums import (
    CognitiveAction,
    EventSource,
    MemoryKind,
    MemoryStatus,
)
from companion.domain.models import make_utterance_event, utcnow
from companion.jobs import handlers as handlers_module
from companion.memory.auto_recall import (
    MAX_AUTO_RECALL,
    build_recall_query,
    router_requested_memory,
    select_auto_recalled,
    should_auto_recall,
)
from companion.memory.embeddings import EmbeddingModelSpec, content_hash
from companion.memory.episodes import (
    EpisodeProposal,
    EpisodeWriter,
    episode_source_id,
    normalize_status,
    parse_episode_content,
    render_episode_content,
)
from companion.memory.embedding_repository import EmbeddingRepository
from companion.memory.models import AccessContext
from companion.memory.service import build_memory_service
from companion.prompts.extraction import build_extraction_messages
from companion.providers.deepseek_client import ChatResponse
from companion.providers.deepseek_cognitive import DeepSeekCognitiveProvider
from companion.routing.models import (
    InputSemanticStatus,
    RawRouteDecision,
    RouteAction,
    TurnState,
)
from companion.runtime import build_runtime

EVENT_ID_RE = re.compile(r"event_id=(\S+)")


# -- helpers ---------------------------------------------------------------


def _chat_response(content: str) -> ChatResponse:
    return ChatResponse(
        content=content,
        model="fake-model",
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )


def _extraction_payload(event_ids, episodes=None) -> str:
    return json.dumps(
        {
            "results": [
                {"event_id": event_id, "memory_candidates": []}
                for event_id in event_ids
            ],
            "episodes": list(episodes or []),
        },
        ensure_ascii=False,
    )


class FakeExtractionClient:
    """Answers the extraction prompt; can also emit a scripted episode."""

    def __init__(self, *, episode_factory=None, candidate_content=None):
        self.calls: list[str] = []
        self.analysis_messages: list[list[dict]] = []
        self.episode_factory = episode_factory
        self.candidate_content = candidate_content

    async def complete(
        self, messages, *, request_type, event_id=None, json_mode=None, **kwargs
    ):
        self.calls.append(request_type)
        if request_type != "analysis":
            return _chat_response("好的，我记着。")
        self.analysis_messages.append(messages)
        text = "\n".join(str(m.get("content") or "") for m in messages)
        event_ids = EVENT_ID_RE.findall(text)
        episodes = (
            self.episode_factory(event_ids, text) if self.episode_factory else []
        )
        payload = json.loads(_extraction_payload(event_ids, episodes))
        if self.candidate_content:
            for item in payload["results"]:
                item["memory_candidates"] = [
                    {
                        "content": self.candidate_content,
                        "operation": "ADD",
                        "assertion_type": "SELF_REPORT",
                        "certainty": "DIRECT",
                        "topic": "preference",
                        "kind": "PERSONAL_FACT",
                        "evidence_span": "喜欢",
                    }
                ]
        return _chat_response(json.dumps(payload, ensure_ascii=False))


class RecordingResponse:
    def __init__(self):
        self.calls = []

    async def generate(self, event, analysis, context):
        self.calls.append((event, analysis, context))
        return "我听着呢。"


class ExplodingCognitive:
    async def analyze(self, *args, **kwargs):
        raise AssertionError("semantic mode must not call analyze")


class ExtractionOnlyCognitive:
    """Background extraction only; the speaker never calls ``analyze``."""

    def __init__(self, client):
        self.provider = DeepSeekCognitiveProvider(client)

    async def extract_events_with_episodes(self, events, **kwargs):
        return await self.provider.extract_events_with_episodes(events, **kwargs)

    async def extract_events(self, events, **kwargs):
        return await self.provider.extract_events(events, **kwargs)

    async def analyze(self, *args, **kwargs):
        raise AssertionError("semantic mode must not call analyze")


class ScriptedRouter:
    provider_name = "fake_router"
    model = "fake-router-model"

    def __init__(self, *, tools=None, turn_state="CONTINUING",
                 action="RESPOND", input_semantics="COMMUNICATIVE"):
        self.calls = 0
        self._tools = list(tools or [])
        self._turn_state = turn_state
        self._action = action
        self._input_semantics = input_semantics

    def _raw(self, tools):
        return RawRouteDecision.model_validate(
            {
                "schema_version": 1,
                "action": self._action,
                "reply_necessity": "REQUIRED"
                if self._action == "RESPOND"
                else "UNNECESSARY",
                "addressee_assessment": "AGENT",
                "input_semantics": self._input_semantics,
                "turn_state": self._turn_state,
                "tool_requests": tools,
                "reason": "scripted route",
            }
        )

    async def route(self, context):
        self.calls += 1
        return self._raw(self._tools)

    async def replan(self, context, previous, results):
        return self._raw([])


async def _active_person(rt, name="Alice"):
    return await rt.registry.register(
        name,
        "friend",
        allow_processing=True,
        allow_persistence=True,
        allow_memory_learning=True,
        allow_external_ai=True,
    )


async def _save_events(repos, person_id, texts, conversation="c1"):
    await repos.conversation.get_or_create(conversation)
    events = []
    for text in texts:
        event = make_utterance_event(
            conversation_id=conversation,
            text=text,
            addressee_hint="agent",
            speaker_person_id=person_id,
            source=EventSource.TEST,
        )
        await repos.utterance.save(event)
        events.append(event)
    return events


async def _search_episodes(service, *, person_id=None, include_retracted=True):
    return await service.search(
        access=AccessContext("admin", is_admin=True),
        kinds=[MemoryKind.EPISODE.value],
        subject_type="PERSON" if person_id else None,
        subject_id=person_id,
        include_retracted=include_retracted,
        limit=50,
    )


def _proposal(events, *, title="机器狗调试", status="ONGOING", summary="尝试调试机器狗，仍未解决。", continues=None):
    return EpisodeProposal(
        title=title,
        summary=summary,
        topic="机器狗调试",
        status=status,
        source_event_ids=[event.event_id for event in events],
        continues_title=continues,
    )


async def _writer(rt):
    service = build_memory_service(rt.repos)
    return EpisodeWriter(rt.repos, service), service


# -- episode format and prompt --------------------------------------------


def test_episode_content_is_stable_and_parseable():
    proposal = EpisodeProposal(
        title="机器狗调试",
        summary="机器人不会走直线，试过重新校准，结果仍未确认。",
        topic="机器狗",
        status="ongoing",
        source_event_ids=["e1", "e2"],
    )
    content = render_episode_content(proposal.cleaned())
    assert content.splitlines()[0] == "[EPISODE]"
    assert content.count("\n标题：") == 1
    parsed = parse_episode_content(content)
    assert parsed["title"] == "机器狗调试"
    assert parsed["topic"] == "机器狗"
    assert parsed["status"] == "ONGOING"
    assert "仍未确认" in parsed["summary"]
    assert parse_episode_content("普通事实") is None


def test_status_normalisation_never_invents_resolution():
    assert normalize_status("solved") == "RESOLVED"
    assert normalize_status("仍在进行") == "UNKNOWN"
    assert normalize_status(None) == "UNKNOWN"


def test_extraction_prompt_carries_real_turns_and_episode_rules():
    messages = build_extraction_messages(
        [{"event_id": "e1", "occurred_at": "t", "text": "机器狗又不动了"}],
        speaker_label="Alice",
        replies={"e1": "先检查电池。"},
        recent_episodes=[{"title": "机器狗调试", "topic": "机器狗", "status": "ONGOING"}],
    )
    text = "\n".join(m["content"] for m in messages)
    assert "event_id=e1" in text
    assert "USER: 机器狗又不动了" in text
    assert "AVA: 先检查电池。" in text
    assert "title=机器狗调试" in text
    # Prose only; the model is never told it may choose ids or permissions.
    assert "不输出 person_id" in text
    assert "RESOLVED | ONGOING | UNKNOWN" in text


# -- provider: one call, two outputs --------------------------------------


class _ProviderClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.messages = None

    async def complete(self, messages, *, request_type, event_id=None, json_mode=None, **kwargs):
        self.calls += 1
        self.messages = messages
        return _chat_response(self.payload)


def _event(text="机器狗又不动了", person="p1", event_id="e1"):
    return make_utterance_event(
        conversation_id="c1",
        text=text,
        addressee_hint="agent",
        speaker_person_id=person,
        source=EventSource.TEST,
        event_id=event_id,
    )


@pytest.mark.asyncio
async def test_provider_returns_candidates_and_episodes_from_one_call():
    client = _ProviderClient(
        _extraction_payload(
            ["e1"],
            [
                {
                    "title": "机器狗调试",
                    "summary": "机器狗不动，尝试换电池，仍未解决。",
                    "topic": "机器狗",
                    "status": "ONGOING",
                    "source_event_ids": ["e1", "invented"],
                }
            ],
        )
    )
    provider = DeepSeekCognitiveProvider(client)
    outcome = await provider.extract_events_with_episodes(
        [_event()], speaker_label="Alice", replies={"e1": "检查电池。"}
    )
    assert client.calls == 1
    assert outcome.candidates == {"e1": []}
    assert len(outcome.episodes) == 1
    # The invented id is dropped; a grounded citation remains.
    assert outcome.episodes[0].source_event_ids == ["e1"]
    assert "AVA: 检查电池。" in "\n".join(m["content"] for m in client.messages)


@pytest.mark.asyncio
async def test_provider_drops_an_episode_with_no_real_source():
    client = _ProviderClient(
        _extraction_payload(
            ["e1"],
            [
                {
                    "title": "凭空事件",
                    "summary": "没有依据的总结。",
                    "source_event_ids": ["invented-only"],
                }
            ],
        )
    )
    provider = DeepSeekCognitiveProvider(client)
    outcome = await provider.extract_events_with_episodes(
        [_event()], speaker_label="Alice"
    )
    assert outcome.episodes == []


@pytest.mark.asyncio
async def test_old_results_only_output_still_works():
    client = _ProviderClient(
        json.dumps(
            {
                "results": [
                    {
                        "event_id": "e1",
                        "memory_candidates": [
                            {
                                "content": "Alice 喜欢猫",
                                "operation": "ADD",
                                "assertion_type": "SELF_REPORT",
                                "certainty": "DIRECT",
                                "topic": "preference",
                                "kind": "PERSONAL_FACT",
                                "evidence_span": "猫",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )
    )
    provider = DeepSeekCognitiveProvider(client)
    extracted = await provider.extract_events([_event()], speaker_label="Alice")
    assert list(extracted) == ["e1"]
    assert len(extracted["e1"]) == 1


# -- background job persists an episode -----------------------------------


def _episode_runtime(db, client):
    return build_runtime(
        db,
        provider="deepseek",
        deepseek_client=client,
        turn_analysis_delay_seconds=0,
        router_mode="legacy",
    )


@pytest.mark.asyncio
async def test_turn_analysis_saves_multi_turn_episode_with_evidence(db):
    text_to_match = "机器狗"

    def episode_factory(event_ids, prompt):
        if text_to_match not in prompt:
            return []
        return [
            {
                "title": "机器狗调试",
                "summary": "用户准备调试机器狗；Ava 建议检查电池；仍未解决。",
                "topic": "机器狗调试",
                "status": "ONGOING",
                "source_event_ids": event_ids,
            }
        ]

    client = FakeExtractionClient(episode_factory=episode_factory)
    rt = _episode_runtime(db, client)
    person = await _active_person(rt)
    event = make_utterance_event(
        conversation_id="c1",
        text="我准备重新调试机器狗。",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    report = await rt.orchestrator.process_utterance(event)
    assert report.status == "COMPLETED"
    event2 = make_utterance_event(
        conversation_id="c1",
        text="电池也换了，还是不动。",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.orchestrator.process_utterance(event2)
    await rt.worker.run_until_idle()

    episodes = await _search_episodes(rt.memory_service, person_id=person.person_id)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["kind"] == "EPISODE"
    assert episode["status"] == MemoryStatus.ACTIVE.value
    assert episode["visibility"] == "PRIVATE"
    assert episode["model_inferred"] is True
    assert episode["human_confirmed"] is False
    parsed = parse_episode_content(episode["content"])
    assert parsed["title"] == "机器狗调试"
    assert parsed["status"] == "ONGOING"

    sources = await rt.memory_service.repo.get_sources(episode["memory_id"])
    cited = {
        source_id
        for row in sources
        for source_id in json.loads(row["source_ref"])
    }
    assert {event.event_id, event2.event_id} <= cited
    provenance = await rt.memory_service.repo.get_provenance(episode["memory_id"])
    assert provenance["topic"] == "机器狗调试"
    assert provenance["model_inferred"] == 1
    # The real Ava reply (never an invented one) was shown to the model.
    prompt = "\n".join(
        m["content"] for m in client.analysis_messages[-1]
    )
    assert "AVA: 好的，我记着。" in prompt


@pytest.mark.asyncio
async def test_repeated_writer_calls_do_not_duplicate_the_episode(db):
    rt = build_runtime(db)
    person = await _active_person(rt)
    events = await _save_events(rt.repos, person.person_id, ["机器狗不动", "换了电池"])
    writer, service = await _writer(rt)
    proposal = _proposal(events)

    first = await writer.write(
        proposal, person_id=person.person_id, conversation_id="c1"
    )
    second = await writer.write(
        proposal, person_id=person.person_id, conversation_id="c1"
    )

    assert first.outcome == "created"
    assert second.outcome == "duplicate"
    assert second.memory_id == first.memory_id
    assert len(await _search_episodes(service, person_id=person.person_id)) == 1
    assert episode_source_id(
        person.person_id, [event.event_id for event in events]
    ).startswith("episode:")


@pytest.mark.asyncio
async def test_continuation_updates_the_same_episode_and_keeps_old_sources(db):
    rt = build_runtime(db)
    person = await _active_person(rt)
    first_events = await _save_events(rt.repos, person.person_id, ["机器狗不动"])
    later_events = await _save_events(rt.repos, person.person_id, ["修好了，能走了"])
    writer, service = await _writer(rt)

    created = await writer.write(
        _proposal(first_events), person_id=person.person_id, conversation_id="c1"
    )
    updated = await writer.write(
        _proposal(
            later_events,
            status="RESOLVED",
            summary="机器狗重新校准后恢复。",
            continues="机器狗调试",
        ),
        person_id=person.person_id,
        conversation_id="c1",
    )

    assert created.outcome == "created"
    assert updated.outcome == "updated"
    assert updated.memory_id == created.memory_id
    episodes = await _search_episodes(service, person_id=person.person_id)
    assert len(episodes) == 1
    assert episodes[0]["version"] == 2
    assert parse_episode_content(episodes[0]["content"])["status"] == "RESOLVED"
    sources = await service.repo.get_sources(created.memory_id)
    assert len(sources) == 2
    revisions = await service.repo.list_revisions(created.memory_id)
    assert any(r.reason == "episode_extended" for r in revisions)
    events = await service.repo.list_provenance_events(created.memory_id)
    assert any(item["event_type"] == "episode_updated" for item in events)


@pytest.mark.asyncio
async def test_a_later_batch_cannot_unresolve_a_resolved_episode(db):
    rt = build_runtime(db)
    person = await _active_person(rt)
    first = await _save_events(rt.repos, person.person_id, ["机器狗不动"])
    second = await _save_events(rt.repos, person.person_id, ["能走了"])
    third = await _save_events(rt.repos, person.person_id, ["又提了一下机器狗"])
    writer, service = await _writer(rt)
    await writer.write(
        _proposal(first), person_id=person.person_id, conversation_id="c1"
    )
    await writer.write(
        _proposal(second, status="RESOLVED", continues="机器狗调试"),
        person_id=person.person_id,
        conversation_id="c1",
    )
    await writer.write(
        _proposal(third, status="ONGOING", continues="机器狗调试"),
        person_id=person.person_id,
        conversation_id="c1",
    )
    episodes = await _search_episodes(service, person_id=person.person_id)
    assert len(episodes) == 1
    assert parse_episode_content(episodes[0]["content"])["status"] == "RESOLVED"


@pytest.mark.asyncio
async def test_ambiguous_continuation_creates_a_new_episode(db):
    rt = build_runtime(db)
    person = await _active_person(rt)
    writer, service = await _writer(rt)
    for _ in range(2):
        await service.create(
            kind=MemoryKind.EPISODE,
            content=render_episode_content(_proposal([], title="机器狗调试")),
            access=AccessContext("admin", is_admin=True),
            created_by=person.person_id,
            subjects=[("PERSON", person.person_id)],
            model_inferred=True,
        )
    events = await _save_events(rt.repos, person.person_id, ["机器狗又出问题"])
    result = await writer.write(
        _proposal(events, continues="机器狗调试"),
        person_id=person.person_id,
        conversation_id="c1",
    )
    assert result.outcome == "created"
    assert len(await _search_episodes(service, person_id=person.person_id)) == 3


@pytest.mark.asyncio
async def test_human_confirmed_episode_is_never_overwritten(db):
    rt = build_runtime(db)
    person = await _active_person(rt)
    writer, service = await _writer(rt)
    confirmed = await service.create(
        kind=MemoryKind.EPISODE,
        content=render_episode_content(_proposal([], title="机器狗调试")),
        access=AccessContext("admin", is_admin=True),
        created_by=person.person_id,
        subjects=[("PERSON", person.person_id)],
        model_inferred=False,
        human_confirmed=True,
    )
    original = confirmed.content
    events = await _save_events(rt.repos, person.person_id, ["机器狗又出问题"])
    result = await writer.write(
        _proposal(events, continues="机器狗调试"),
        person_id=person.person_id,
        conversation_id="c1",
    )
    assert result.outcome == "created"
    episodes = await _search_episodes(service, person_id=person.person_id)
    assert len(episodes) == 2
    stored = [row for row in episodes if row["memory_id"] == confirmed.memory_id][0]
    assert stored["content"] == original
    assert stored["human_confirmed"] is True


@pytest.mark.asyncio
async def test_foreign_or_fabricated_sources_reject_the_episode(db):
    rt = build_runtime(db)
    alice = await _active_person(rt, "Alice")
    bob = await _active_person(rt, "Bob")
    bob_event = (await _save_events(rt.repos, bob.person_id, ["Bob 的机器狗"], conversation="c1"))[0]
    writer, service = await _writer(rt)

    foreign = await writer.write(
        EpisodeProposal(
            title="别人的事件",
            summary="不应保存",
            topic="机器狗",
            status="ONGOING",
            source_event_ids=[bob_event.event_id],
        ),
        person_id=alice.person_id,
        conversation_id="c1",
    )
    fabricated = await writer.write(
        EpisodeProposal(
            title="编造的事件",
            summary="不应保存",
            topic="机器狗",
            status="ONGOING",
            source_event_ids=["evt-does-not-exist"],
        ),
        person_id=alice.person_id,
        conversation_id="c1",
    )
    assert foreign.outcome == "rejected"
    assert fabricated.outcome == "rejected"
    assert await _search_episodes(service, person_id=alice.person_id) == []


@pytest.mark.asyncio
async def test_revoked_consent_blocks_the_episode_write(db):
    rt = build_runtime(db)
    person = await _active_person(rt)
    events = await _save_events(rt.repos, person.person_id, ["机器狗不动"])
    writer, service = await _writer(rt)
    from companion.domain.models import Consent

    await rt.registry.set_consent(
        Consent(
            person_id=person.person_id,
            allow_processing=True,
            allow_persistence=True,
            allow_memory_learning=False,
            allow_external_ai=True,
            updated_at=utcnow(),
        )
    )
    result = await writer.write(
        _proposal(events), person_id=person.person_id, conversation_id="c1"
    )
    assert result.outcome == "rejected"
    assert await _search_episodes(service, person_id=person.person_id) == []


@pytest.mark.asyncio
async def test_episode_failure_does_not_discard_extracted_facts(db):
    def episode_factory(event_ids, prompt):
        return [
            {
                "title": "编造事件",
                "summary": "没有依据。",
                "source_event_ids": ["only-an-invented-id"],
            }
        ]

    client = FakeExtractionClient(
        episode_factory=episode_factory,
        candidate_content="Alice 喜欢茶",
    )
    rt = _episode_runtime(db, client)
    person = await _active_person(rt)
    event = make_utterance_event(
        conversation_id="c1",
        text="我喜欢茶",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )
    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_until_idle()

    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert len(candidates) == 1
    assert await _search_episodes(rt.memory_service, person_id=person.person_id) == []
    marked = await rt.repos.extraction.get(event.event_id)
    assert marked is not None and marked["status"] == "processed"


@pytest.mark.asyncio
async def test_episode_storage_failure_is_retried_without_losing_facts(db, monkeypatch):
    def episode_factory(event_ids, prompt):
        return [
            {
                "title": "机器狗调试",
                "summary": "机器狗不动，仍在处理。",
                "topic": "机器狗调试",
                "status": "ONGOING",
                "source_event_ids": event_ids,
            }
        ]

    client = FakeExtractionClient(
        episode_factory=episode_factory,
        candidate_content="Alice 喜欢机器狗",
    )
    rt = _episode_runtime(db, client)
    person = await _active_person(rt)
    event = make_utterance_event(
        conversation_id="c1",
        text="机器狗又不动了",
        addressee_hint="agent",
        speaker_person_id=person.person_id,
        source=EventSource.TEST,
    )

    original_writer = handlers_module.EpisodeWriter
    state = {"failures": 1}

    class FlakyWriter(original_writer):
        async def write(self, proposal, *, person_id, conversation_id):
            if state["failures"]:
                state["failures"] -= 1
                raise RuntimeError("scripted episode storage failure")
            return await super().write(
                proposal, person_id=person_id, conversation_id=conversation_id
            )

    monkeypatch.setattr(handlers_module, "EpisodeWriter", FlakyWriter)
    await rt.orchestrator.process_utterance(event)
    await rt.worker.run_until_idle()

    assert state["failures"] == 0
    job = await rt.repos.job.get_by_key(f"turn_analysis:{event.event_id}")
    assert job is not None and job.status.value == "COMPLETED"
    candidates = await rt.repos.analysis.get_candidates(event.event_id)
    assert len(candidates) == 1  # INSERT OR IGNORE, not duplicated
    assert len(await _search_episodes(rt.memory_service, person_id=person.person_id)) == 1
    marked = await rt.repos.extraction.get(event.event_id)
    assert marked is not None and marked["status"] == "processed"


@pytest.mark.asyncio
async def test_episode_update_invalidates_stale_vectors(db):
    rt = build_runtime(db)
    person = await _active_person(rt)
    first = await _save_events(rt.repos, person.person_id, ["机器狗不动"])
    later = await _save_events(rt.repos, person.person_id, ["机器狗修好了"])
    writer, service = await _writer(rt)
    created = await writer.write(
        _proposal(
            first,
            status="ONGOING",
            summary="机器狗不动，旧的坏消息。",
        ),
        person_id=person.person_id,
        conversation_id="c1",
    )
    old = await service.repo.get(created.memory_id)
    repo = EmbeddingRepository(db)
    spec = EmbeddingModelSpec(provider="test", model_name="m", dimensions=4)
    model_id = await repo.upsert_model(spec)
    await repo.upsert_embedding(
        created.memory_id,
        model_id,
        b"\x00\x00\x00\x00",
        memory_version=old.version,
        content_hash=content_hash(old.kind.value, old.content),
    )
    assert [row["memory_id"] for row in await repo.list_ready(model_id)] == [
        created.memory_id
    ]

    await writer.write(
        _proposal(
            later,
            status="RESOLVED",
            summary="机器狗修好了，新的好消息。",
            continues="机器狗调试",
        ),
        person_id=person.person_id,
        conversation_id="c1",
    )
    assert await repo.list_ready(model_id) == []
    refreshed = await service.repo.get(created.memory_id)
    assert "新的好消息" in refreshed.content
    hits = await service.search(
        access=AccessContext("admin", is_admin=True),
        query="新的好消息",
        statuses=["ACTIVE"],
        limit=10,
    )
    assert [row["memory_id"] for row in hits] == [created.memory_id]


# -- auto recall: trigger and query ---------------------------------------


def _route(
    *,
    action=RouteAction.RESPOND,
    turn_state=TurnState.NEW_TOPIC,
    semantics=InputSemanticStatus.COMMUNICATIVE,
    tools=(),
):
    return SimpleNamespace(
        action=action,
        turn_state=turn_state,
        input_semantics=semantics,
        tool_requests=list(tools),
    )


def _view(*, turns=(), scene=None, quiet=False, speaker="p1"):
    return SimpleNamespace(
        recent_turns=list(turns),
        compact_scene=scene,
        quiet_mode=quiet,
        current_speaker_id=speaker,
    )


def test_trigger_a_explicit_past_reference():
    view = _view(scene={"topic": "机器狗调试"})
    assert should_auto_recall(
        _route(), "你还记得上次那个问题吗？", view
    )
    assert should_auto_recall(
        _route(), "昨天那个后来怎么解决的？", view
    )
    assert should_auto_recall(
        _route(), "我们以前聊过这个吧？", view
    )


def test_trigger_b_followup_uses_trusted_context_topic():
    view = _view(
        turns=[
            {
                "role": "human",
                "speaker_person_id": "p1",
                "text": "我准备重新调试机器狗。",
            }
        ]
    )
    assert should_auto_recall(
        _route(turn_state=TurnState.FOLLOW_UP), "这次还是不行。", view
    )
    # No concrete topic anywhere: a bare "这次还是不行" is not a memory need.
    assert not should_auto_recall(
        _route(turn_state=TurnState.FOLLOW_UP), "这次还是不行。", _view()
    )


def test_trigger_c_continuation_with_a_named_topic():
    assert should_auto_recall(
        _route(turn_state=TurnState.NEW_TOPIC), "机器狗这次还是不行。", _view()
    )
    assert not should_auto_recall(
        _route(turn_state=TurnState.NEW_TOPIC), "这次还是不行。", _view()
    )
    # "还是" as a preference ("还是喜欢…") is not unfinished work.
    assert not should_auto_recall(
        _route(turn_state=TurnState.NEW_TOPIC), "我还是喜欢机器狗。", _view()
    )


def test_greetings_knowledge_questions_and_silent_routes_do_not_trigger():
    view = _view(scene={"topic": "天气"})
    for text in ("你好呀", "今天天气怎么样？", "什么是光合作用？", "晚安"):
        assert not should_auto_recall(_route(), text, view), text
    assert not should_auto_recall(
        _route(action=RouteAction.WAIT), "你还记得上次那个问题吗？", view
    )
    assert not should_auto_recall(
        _route(action=RouteAction.IGNORE), "你还记得上次那个问题吗？", view
    )
    assert not should_auto_recall(
        _route(semantics=InputSemanticStatus.NON_COMMUNICATIVE),
        "你还记得上次那个问题吗？",
        view,
    )
    assert not should_auto_recall(
        _route(), "你还记得上次那个问题吗？", _view(quiet=True)
    )


def test_query_uses_only_this_speakers_context():
    view = _view(
        turns=[
            {
                "role": "human",
                "speaker_person_id": "other",
                "text": "别人的秘密项目代号蓝鲸",
            },
            {
                "role": "human",
                "speaker_person_id": "p1",
                "text": "我准备重新调试机器狗。",
            },
            {
                "role": "assistant",
                "speaker_person_id": None,
                "text": "先检查电池。",
            },
        ]
    )
    query = build_recall_query("这次还是不行", view)
    assert query
    assert "机器" in query
    assert "蓝鲸" not in query
    # No reliable topic at all: do not build a query from "这次还是不行".
    assert build_recall_query("这次还是不行", _view()) is None


def test_select_limits_dedupes_and_prefers_episodes_without_crowding_facts():
    memories = [
        {
            "memory_id": "ep1",
            "kind": "EPISODE",
            "content": "机器狗调试：换了电池仍未解决。",
        },
        {
            "memory_id": "ep2",
            "kind": "EPISODE",
            "content": "机器狗校准记录。",
        },
        {
            "memory_id": "ep3",
            "kind": "EPISODE",
            "content": "机器狗比赛报名。",
        },
        {
            "memory_id": "fact1",
            "kind": "PERSONAL_FACT",
            "content": "用户有一台机器狗。",
        },
        {
            "memory_id": "fact2",
            "kind": "PERSONAL_FACT",
            "content": "用户喜欢喝茶。",
        },
    ]
    selected = select_auto_recalled(
        memories, "机器狗 调试", exclude_ids={"fact2"}
    )
    ids = [row["memory_id"] for row in selected]
    assert len(ids) <= MAX_AUTO_RECALL
    assert "fact2" not in ids
    assert "fact1" in ids
    assert len([mid for mid in ids if mid.startswith("ep")]) <= 2


def test_assignment_alone_is_not_relevance():
    memories = [
        {"memory_id": "assigned", "kind": "PERSONAL_FACT", "content": "用户喜欢登山。"},
        {
            "memory_id": "vector",
            "kind": "PERSONAL_FACT",
            "content": "与查询措辞不同但向量相关。",
            "vector_rank": 2,
        },
    ]
    selected = select_auto_recalled(memories, "机器狗 调试")
    assert [row["memory_id"] for row in selected] == ["vector"]


def test_router_memory_request_is_detected_even_when_it_failed():
    route = _route(
        tools=[SimpleNamespace(tool_name="memory.search")]
    )
    assert router_requested_memory(route, [])
    result = SimpleNamespace(tool_name="memory.search", status="EMPTY")
    assert router_requested_memory(_route(), [result])
    assert not router_requested_memory(_route(), [])


# -- auto recall: the real semantic turn ----------------------------------


def _semantic_runtime(db, router, response=None):
    return build_runtime(
        db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=router,
        response=response or RecordingResponse(),
        deepseek_client=object(),
        cognitive=ExplodingCognitive(),
        allow_mixed_providers=True,
    )


async def _seed_episode(rt, person, content):
    return await rt.memory.service.create(
        kind=MemoryKind.EPISODE,
        content=content,
        access=AccessContext("admin", is_admin=True),
        visibility="PRIVATE",
        status="ACTIVE",
        subjects=[("PERSON", person.person_id)],
        model_inferred=True,
    )


async def _say(rt, person_id, text, conversation="conv"):
    event = make_utterance_event(
        conversation_id=conversation,
        text=text,
        addressee_hint="agent",
        speaker_person_id=person_id,
        source=EventSource.TEST,
    )
    return event, await rt.orchestrator.process_utterance(event)


@pytest.mark.asyncio
async def test_followup_reaches_speaker_without_a_router_memory_request(db):
    router = ScriptedRouter(turn_state="CONTINUING")
    rt = _semantic_runtime(db, router)
    person = await _active_person(rt)
    episode = await _seed_episode(
        rt,
        person,
        "[EPISODE]\n标题：机器狗调试\n主题：机器狗调试\n状态：ONGOING\n"
        "梗概：机器狗不走路，换过电池仍未解决。",
    )

    await _say(rt, person.person_id, "我准备重新调试机器狗。")
    _, report = await _say(rt, person.person_id, "这次还是不行。")

    assert report.status == "COMPLETED"
    assert any("auto_recall" in step for step in report.steps)
    context = rt.response.calls[-1][2]
    assert [m["memory_id"] for m in context.memories] == [episode.memory_id]
    # No fake tool call was invented to justify the recall.
    assert context.tool_results == []
    assert context.memory_retrieval_status == "none"


@pytest.mark.asyncio
async def test_router_memory_tool_result_is_not_duplicated_by_auto_recall(db):
    router = ScriptedRouter(
        turn_state="CONTINUING",
        tools=[{"tool_name": "memory.search", "arguments": {"query": "机器狗"}}],
    )
    rt = _semantic_runtime(db, router)
    person = await _active_person(rt)
    await _seed_episode(
        rt,
        person,
        "[EPISODE]\n标题：机器狗调试\n主题：机器狗调试\n状态：ONGOING\n"
        "梗概：机器狗不走路。",
    )
    await _say(rt, person.person_id, "我准备重新调试机器狗。")

    calls: list[str] = []
    original = rt.memory.retrieve

    async def spy(query, person_id):
        calls.append(query)
        return await original(query, person_id)

    rt.memory.retrieve = spy
    _, report = await _say(rt, person.person_id, "这次还是不行。")

    assert report.status == "COMPLETED"
    assert not any("auto_recall" in step for step in report.steps)
    assert len(calls) == 1  # exactly the router's own memory.search query
    assert rt.response.calls[-1][2].memories


@pytest.mark.asyncio
async def test_job_generated_episode_is_recalled_by_a_later_semantic_turn(db):
    """Acceptance path: real turns -> background episode -> later recall."""

    def episode_factory(event_ids, prompt):
        if "机器狗" not in prompt:
            return []
        return [
            {
                "title": "机器狗调试",
                "summary": "用户准备重新调试机器狗，问题尚未解决。",
                "topic": "机器狗调试",
                "status": "ONGOING",
                "source_event_ids": event_ids,
            }
        ]

    client = FakeExtractionClient(episode_factory=episode_factory)
    router = ScriptedRouter(turn_state="CONTINUING")
    rt = build_runtime(
        db,
        provider="deepseek",
        router_mode="semantic",
        router_provider=router,
        response=RecordingResponse(),
        deepseek_client=client,
        cognitive=ExtractionOnlyCognitive(client),
        allow_mixed_providers=True,
        turn_analysis_delay_seconds=0,
    )
    person = await _active_person(rt)

    first, _ = await _say(rt, person.person_id, "我准备重新调试机器狗。")
    await rt.worker.run_until_idle()

    episodes = await _search_episodes(rt.memory_service, person_id=person.person_id)
    assert len(episodes) == 1
    assert client.calls == ["analysis"]

    _, report = await _say(rt, person.person_id, "这次还是不行。")

    assert report.status == "COMPLETED"
    assert any("auto_recall" in step for step in report.steps)
    context = rt.response.calls[-1][2]
    assert [m["memory_id"] for m in context.memories] == [
        episodes[0]["memory_id"]
    ]
    # Recall itself added no model call; the only one was the background pass.
    assert client.calls == ["analysis"]


@pytest.mark.asyncio
async def test_recall_failure_degrades_to_empty_without_breaking_the_turn(db):
    router = ScriptedRouter(turn_state="CONTINUING")
    rt = _semantic_runtime(db, router)
    person = await _active_person(rt)
    await _seed_episode(
        rt,
        person,
        "[EPISODE]\n标题：机器狗调试\n主题：机器狗调试\n状态：ONGOING\n"
        "梗概：机器狗不走路。",
    )
    await _say(rt, person.person_id, "我准备重新调试机器狗。")

    async def broken(query, person_id):
        raise RuntimeError("scripted recall failure")

    rt.memory.retrieve = broken
    _, report = await _say(rt, person.person_id, "这次还是不行。")

    assert report.status == "COMPLETED"
    assert not any("auto_recall" in step for step in report.steps)
    context = rt.response.calls[-1][2]
    assert context.memories == []
    action = await rt.repos.analysis.get_action(report.event_id)
    assert action.action_type == CognitiveAction.RESPOND
