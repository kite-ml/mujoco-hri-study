"""Measures + export derive the study's DVs from a synthetic event log."""

from mjhri.events import InMemoryEventLog
from mjhri.events.export import summary_rows
from mjhri.events.measures import (
    phase_durations,
    summarize,
    time_to_deployable_spec,
)
from mjhri.schemas.events import StudyEvent


def _ev(t, etype, **kw):
    return StudyEvent(participant_id="P1", event_type=etype, server_ts=t, **kw)


def _log_condition_B():
    """A reward-spec (B) condition: enter teach, edit, mark deployable, train, review."""
    return [
        _ev(100.0, "phase_enter", condition="B", phase="teach"),
        _ev(110.0, "spec_edit", condition="B", phase="teach"),
        _ev(125.0, "spec_edit", condition="B", phase="teach"),
        _ev(160.0, "spec_marked_deployable", condition="B", phase="teach"),
        _ev(161.0, "phase_exit", condition="B", phase="teach"),
        _ev(161.0, "phase_enter", condition="B", phase="review"),
        _ev(200.0, "eval_completed", condition="B", phase="review", payload={"success_rate": 0.65}),
        _ev(205.0, "deploy_decision", condition="B", phase="review",
            payload={"deploy": True, "confidence": 70}),
        _ev(210.0, "phase_exit", condition="B", phase="review"),
        _ev(210.0, "session_complete"),
    ]


def test_phase_durations():
    dur = phase_durations(_log_condition_B())
    assert dur[("B", "teach")] == 61.0
    assert dur[("B", "review")] == 49.0


def test_time_to_deployable_spec():
    ttds = time_to_deployable_spec(_log_condition_B(), "B")
    assert ttds == 60.0  # 160 (marked) - 100 (teach enter)


def test_summarize_pulls_all_dvs():
    summ = summarize(_log_condition_B())
    assert summ["complete"] is True
    b = summ["conditions"]["B"]
    assert b["time_to_deployable_spec"] == 60.0
    assert b["success_rate"] == 0.65
    assert b["deploy"]["confidence"] == 70
    assert b["event_counts"]["spec_edit"] == 2


def test_summary_rows_export_shape():
    rows = summary_rows({"P1": _log_condition_B()})
    assert len(rows) == 1
    r = rows[0]
    assert r["participant_id"] == "P1" and r["condition"] == "B"
    assert r["time_to_deployable_spec_s"] == 60.0
    assert r["success_rate"] == 0.65
    assert r["deploy"] is True and r["deploy_confidence"] == 70


def test_eventlog_roundtrip_and_backends(tmp_path):
    from mjhri.events import JsonlEventLog, SqliteEventLog

    for log in (InMemoryEventLog(), JsonlEventLog(tmp_path / "e.jsonl"),
                SqliteEventLog(tmp_path / "e.db")):
        for e in _log_condition_B():
            log.emit(e)
        got = log.query("P1")
        assert [e.event_type for e in got] == [e.event_type for e in _log_condition_B()]
