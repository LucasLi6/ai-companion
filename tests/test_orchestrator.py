"""Test C: IGNORE still records; Test J: 100-event integration."""

from __future__ import annotations

from companion.domain.enums import CognitiveAction


async def test_ignore_still_records_everything(register, say, rt):
    p = await register("Speaker")
    report = await say(p.person_id, "human", "我昨天完成了一个项目。")

    assert report.action == CognitiveAction.IGNORE
    assert report.response_text is None

    # Original event persisted.
    event = await rt.repos.utterance.get(report.event_id)
    assert event is not None
    assert event.text == "我昨天完成了一个项目。"

    # Analysis persisted.
    analysis = await rt.repos.analysis.get_analysis(report.event_id)
    assert analysis is not None
    assert analysis.action == CognitiveAction.IGNORE

    # Action persisted, with no response text.
    action = await rt.repos.analysis.get_action(report.event_id)
    assert action is not None
    assert action.action_type == CognitiveAction.IGNORE
    assert action.response_text is None

    # Memory candidates and a memory job were created.
    candidates = await rt.repos.analysis.get_candidates(report.event_id)
    assert len(candidates) == 1
    job = await rt.repos.job.get_by_key(f"memory:{report.event_id}")
    assert job is not None


async def test_wait_is_not_an_error(register, say, rt):
    p = await register("Speaker")
    report = await say(p.person_id, "unknown", "自言自语")
    assert report.status == "COMPLETED"
    assert report.action == CognitiveAction.WAIT
    assert report.response_text is None
    event = await rt.repos.utterance.get(report.event_id)
    assert event.processing_status.value == "COMPLETED"


async def test_hundred_events_integration(register, say, rt):
    people = [await register(f"Person{i}") for i in range(3)]
    reports = []
    for i in range(100):
        person = people[i % 3]
        if i % 3 == 0:
            to = "agent"
        elif i % 3 == 1:
            to = "human"
        else:
            to = "unknown"
        reports.append(await say(person.person_id, to, f"message {i}"))

    # Every event completed with a decision.
    assert all(r.status == "COMPLETED" for r in reports)
    assert all(r.action is not None for r in reports)

    # Exactly one analysis and one action per event (no duplicates).
    for r in reports:
        cur = await rt.db.execute(
            "SELECT COUNT(*) AS c FROM cognitive_analyses WHERE event_id = ?", (r.event_id,)
        )
        assert (await cur.fetchone())["c"] == 1
        cur = await rt.db.execute(
            "SELECT COUNT(*) AS c FROM agent_actions WHERE event_id = ?", (r.event_id,)
        )
        assert (await cur.fetchone())["c"] == 1

    # Event order is preserved by occurrence time (list_events returns newest
    # first, so the sequence is non-increasing).
    events = await rt.repos.utterance.list_events(None, 1000)
    assert len(events) == 100
    times = [e.occurred_at for e in events]
    assert all(times[i] >= times[i + 1] for i in range(len(times) - 1))

    # Database integrity: counts are consistent.
    cur = await rt.db.execute("SELECT COUNT(*) AS c FROM utterances")
    assert (await cur.fetchone())["c"] == 100
