"""Protocol engine: gating, advancement, and event-sourced resume."""

import pytest

from mjhri.protocol import ProtocolEngine, assign
from mjhri.protocol.engine import GateError
from mjhri.schemas.events import StudyEvent
from mjhri.schemas.protocol import Condition, CounterbalanceSpec, PhaseSpec, StudyProtocol


def _protocol():
    return StudyProtocol(
        id="s", name="s",
        conditions=[Condition(id="A", label="A", modality="teleop")],
        phases=[
            PhaseSpec(name="teach", kind="teach", requires_events=["train_completed"]),
            PhaseSpec(name="review", kind="review", requires_instruments=["deploy_confidence"]),
        ],
        tasks=["place"], variants=["medium"],
        counterbalance=CounterbalanceSpec(balance_conditions=False, balance_tasks=False),
    )


def _engine():
    p = _protocol()
    return ProtocolEngine(p, assign("P1", p))


def _run(log, new):
    """Apply engine-produced events to a running list (simulating a store)."""
    log.extend(new)


def test_start_opens_first_phase():
    eng = _engine()
    log = []
    _run(log, eng.start(log))
    st = eng.state(log)
    assert st.condition_id == "A" and st.phase_name == "teach" and st.entered
    assert not st.is_complete
    # start is idempotent on resume
    assert eng.start(log) == []


def test_gate_blocks_until_required_event():
    eng = _engine()
    log = []
    _run(log, eng.start(log))
    ok, missing = eng.can_advance(log)
    assert not ok and "requires event 'train_completed'" in missing
    with pytest.raises(GateError):
        eng.advance(log)
    # satisfy the gate
    log.append(StudyEvent(participant_id="P1", event_type="train_completed",
                          condition="A", phase="teach"))
    ok, _ = eng.can_advance(log)
    assert ok


def test_full_flow_completes():
    eng = _engine()
    log = []
    _run(log, eng.start(log))
    log.append(StudyEvent(participant_id="P1", event_type="train_completed",
                          condition="A", phase="teach"))
    _run(log, eng.advance(log))  # teach → review
    st = eng.state(log)
    assert st.phase_name == "review" and st.entered
    # review gate needs a survey
    assert not eng.can_advance(log)[0]
    log.append(StudyEvent(participant_id="P1", event_type="survey_submitted",
                          condition="A", phase="review",
                          payload={"instrument": "deploy_confidence"}))
    _run(log, eng.advance(log))  # review → complete
    assert eng.state(log).is_complete


def test_multi_condition_flow_with_unstamped_events():
    """A 2-condition protocol runs to completion even when the log never stamps
    server_ts (in-memory reference store) — engine events must keep causal order."""
    proto = StudyProtocol(
        id="s", name="s",
        conditions=[Condition(id="A", label="A", modality="teleop"),
                    Condition(id="B", label="B", modality="reward_spec")],
        phases=[PhaseSpec(name="teach", requires_events=["done"]),
                PhaseSpec(name="review", requires_instruments=["survey1"])],
        tasks=["t1"], variants=["m"],
        counterbalance=CounterbalanceSpec(balance_conditions=False, balance_tasks=False),
    )
    eng = ProtocolEngine(proto, assign("P1", proto))
    log = []
    log.extend(eng.start(log))
    steps = 0
    while not eng.state(log).is_complete and steps < 20:
        steps += 1
        st = eng.state(log)
        ph = proto.phase(st.phase_name)
        for et in ph.requires_events:
            log.append(StudyEvent(participant_id="P1", event_type=et,
                                 condition=st.condition_id, phase=st.phase_name))
        for iid in ph.requires_instruments:
            log.append(StudyEvent(participant_id="P1", event_type="survey_submitted",
                                 condition=st.condition_id, phase=st.phase_name,
                                 payload={"instrument": iid}))
        assert eng.can_advance(log)[0], (st.condition_id, st.phase_name, eng.can_advance(log)[1])
        log.extend(eng.advance(log))
    assert eng.state(log).is_complete and steps == 4  # 2 conditions x 2 phases


def test_two_pass_grouping_and_run_requirements():
    """'deploy'-group phases run after all runs' 'main' phases, and a run can add its
    own gate requirements (per-method battery placement)."""
    from mjhri.protocol.counterbalance import Assignment

    proto = StudyProtocol(
        id="s", name="s",
        conditions=[Condition(id="A", label="A", modality="teleop"),
                    Condition(id="B", label="B", modality="teleop")],
        phases=[
            PhaseSpec(name="teach", group="main", requires_events=["done"]),
            PhaseSpec(name="review", group="deploy", requires_events=["played"]),
        ],
    )
    a = Assignment(
        participant_id="P1", condition_order=["A", "B"],
        task_by_condition={"A": {"extra_instruments": {"teach": ["survey1"]}}, "B": {}},
    )
    eng = ProtocolEngine(proto, a)
    # main pass for both runs, THEN deploy pass for both runs
    assert [(r, p.name) for r, p in eng.steps] == [
        ("A", "teach"), ("B", "teach"), ("A", "review"), ("B", "review")
    ]
    # run A's teach gate includes its extra instrument; run B's does not
    assert eng.requirements(0) == (["done"], ["survey1"])
    assert eng.requirements(1) == (["done"], [])


def test_resume_from_partial_log():
    """A fresh engine derives the same state from an existing log (crash/refresh)."""
    eng = _engine()
    log = []
    _run(log, eng.start(log))
    log.append(StudyEvent(participant_id="P1", event_type="train_completed",
                          condition="A", phase="teach"))
    _run(log, eng.advance(log))
    # brand-new engine instance, same log
    resumed = _engine()
    st = resumed.state(log)
    assert st.phase_name == "review" and st.entered and not st.is_complete
