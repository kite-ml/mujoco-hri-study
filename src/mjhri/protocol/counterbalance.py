"""Deterministic counterbalancing from a participant id.

Condition order uses a **Williams design** — a balanced Latin square in which each
condition immediately follows every other condition equally often, controlling
first-order carryover (a participant's condition B experience colours their B→C
transition equally to everyone else's). Task assignment is rotated independently so
task and condition are not confounded.

Everything is a pure function of the participant id, so the same participant always
resumes into the same plan and no assignment state needs storing.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from ..schemas.protocol import StudyProtocol


def _stable_index(participant_id: str) -> int:
    """A deterministic non-negative int from a participant id, independent of the
    process hash seed. Numeric ids map to themselves so P1/P2/... are easy to reason
    about; otherwise fall back to a CRC of the bytes."""
    pid = participant_id.strip()
    digits = "".join(ch for ch in pid if ch.isdigit())
    if digits and digits == pid.lstrip("Pp-_ ").strip():
        return int(digits)
    return zlib.crc32(pid.encode("utf-8"))


def williams_square(n: int) -> list[list[int]]:
    """Williams Latin square over ``range(n)``.

    Returns ``n`` rows for even ``n`` (already carryover-balanced), or ``2n`` rows
    for odd ``n`` (the square plus its column-reverse, needed for balance). Each row
    is a condition order; each column is a Latin square (every position sees every
    condition equally).
    """
    if n <= 0:
        return []
    if n == 1:
        return [[0]]
    # Balanced first row: 0, 1, n-1, 2, n-2, 3, n-3, ...
    first: list[int] = []
    for j in range(n):
        first.append(j // 2 if j % 2 == 0 else n - 1 - (j - 1) // 2)
    square = [[(v + r) % n for v in first] for r in range(n)]
    if n % 2 == 1:
        square += [list(reversed(row)) for row in square]
    return square


@dataclass(frozen=True)
class Assignment:
    """A participant's fixed plan: their condition order and per-condition task."""

    participant_id: str
    condition_order: list[str]
    #: condition id -> {"task": <task id>, "variant": <variant>}
    task_by_condition: dict[str, dict[str, str]]

    def as_dict(self) -> dict:
        return {
            "participant_id": self.participant_id,
            "condition_order": list(self.condition_order),
            "task_by_condition": {k: dict(v) for k, v in self.task_by_condition.items()},
        }


def assign(participant_id: str, protocol: StudyProtocol) -> Assignment:
    """Compute a participant's condition order and task assignment deterministically."""
    idx = _stable_index(participant_id)
    cond_ids = protocol.condition_ids()
    n = len(cond_ids)

    cb = protocol.counterbalance
    if cb.balance_conditions and cb.method in ("williams", "latin_square") and n > 0:
        square = williams_square(n)
        order_row = square[idx % len(square)]
        condition_order = [cond_ids[i] for i in order_row]
    else:
        condition_order = list(cond_ids)

    task_by_condition: dict[str, dict[str, str]] = {}
    tasks = list(protocol.tasks)
    variants = list(protocol.variants) or ["medium"]
    if tasks:
        # Rotate the task list so task and condition-order are decoupled; each
        # condition in the participant's order gets the next task (wrapping).
        shift = idx % len(tasks) if cb.balance_tasks else 0
        rotated = tasks[shift:] + tasks[:shift]
        # Variant is fixed per participant (rotated too) so difficulty is balanced.
        variant = variants[idx % len(variants)] if cb.balance_tasks else variants[0]
        for pos, cond_id in enumerate(condition_order):
            task_by_condition[cond_id] = {
                "task": rotated[pos % len(rotated)],
                "variant": variant,
            }
    return Assignment(
        participant_id=participant_id,
        condition_order=condition_order,
        task_by_condition=task_by_condition,
    )
