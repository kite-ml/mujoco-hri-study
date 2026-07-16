"""Counterbalancing: Williams square is a valid Latin square with first-order
carryover balance, and assignment is deterministic and covers every condition."""

from itertools import product

import pytest

from mjhri.protocol import assign, williams_square
from mjhri.schemas.protocol import Condition, CounterbalanceSpec, PhaseSpec, StudyProtocol


def _is_latin_square(square, n):
    for row in square:
        if sorted(row) != list(range(n)):
            return False
    for col in range(n):
        column = [row[col] for row in square]
        # each value appears equally often down a column
        for v in range(n):
            if column.count(v) != len(square) // n:
                return False
    return True


def _carryover_balanced(square, n):
    """Each ordered adjacent pair (a immediately before b) occurs equally often."""
    counts = {pair: 0 for pair in product(range(n), repeat=2) if pair[0] != pair[1]}
    for row in square:
        for a, b in zip(row, row[1:]):
            counts[(a, b)] += 1
    return len(set(counts.values())) == 1


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_williams_is_balanced_latin_square(n):
    square = williams_square(n)
    assert _is_latin_square(square, n), f"n={n} not a Latin square: {square}"
    assert _carryover_balanced(square, n), f"n={n} not carryover-balanced: {square}"


def _protocol(n_conditions=3, tasks=("place", "sort", "stack")):
    conditions = [Condition(id=c, label=c, modality=c) for c in ["A", "B", "C"][:n_conditions]]
    return StudyProtocol(
        id="s", name="s", conditions=conditions,
        phases=[PhaseSpec(name="teach"), PhaseSpec(name="review")],
        tasks=list(tasks), variants=["medium", "hard"],
        counterbalance=CounterbalanceSpec(method="williams"),
    )


def test_assignment_is_deterministic():
    p = _protocol()
    a1 = assign("P7", p)
    a2 = assign("P7", p)
    assert a1.condition_order == a2.condition_order
    assert a1.task_by_condition == a2.task_by_condition


def test_assignment_covers_all_conditions_and_assigns_tasks():
    p = _protocol()
    a = assign("P3", p)
    assert sorted(a.condition_order) == ["A", "B", "C"]
    assert set(a.task_by_condition) == {"A", "B", "C"}
    for cond, spec in a.task_by_condition.items():
        assert spec["task"] in {"place", "sort", "stack"}
        assert spec["variant"] in {"medium", "hard"}


def test_condition_order_varies_across_participants():
    p = _protocol()
    orders = {tuple(assign(f"P{i}", p).condition_order) for i in range(6)}
    # 3 conditions → Williams gives 6 distinct orders; expect real variation
    assert len(orders) >= 3
