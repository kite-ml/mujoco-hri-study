"""Memory-based behaviour cloning — a dependency-light imitation policy.

Clones a set of demonstrations without a neural net or torch: at each control tick it
matches the current state to the nearest demonstrated state (within a forward phase
window, so it keeps progressing) and replays that state's action. With demonstrations
that cover the reset variation (e.g. language-generated demos over randomized object
poses), this closed-loop nearest-state policy adapts to where the objects actually are.

For a *single* demonstration this degenerates to state-tracked replay — fine for a
one-shot imitation where a plan-execution path (``plan_from_demo`` +
``PickPlaceController``) is usually the more robust choice.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

Demo = Tuple[np.ndarray, np.ndarray]  # (states [T, d], actions [T, nu])


class NearestStatePolicy:
    def __init__(
        self,
        demos: Sequence[Demo],
        state_fn: Callable[[np.ndarray], np.ndarray],
        *,
        duration: float,
        phase_back: float = 0.12,
        phase_fwd: float = 0.30,
        weights: Sequence[float] | None = None,
    ):
        self._state_fn = state_fn
        self.duration = float(duration)
        self._pb, self._pf = float(phase_back), float(phase_fwd)
        states, actions, phases = [], [], []
        for s, a in demos:
            s = np.asarray(s, np.float64)
            a = np.asarray(a, np.float64)
            n = min(len(s), len(a))
            if n < 2:
                continue
            states.append(s[:n])
            actions.append(a[:n])
            phases.append(np.linspace(0.0, 1.0, n))
        self._states = np.vstack(states) if states else np.zeros((1, 1))
        self._actions = np.vstack(actions) if actions else np.zeros((1, 1))
        self._phase = np.concatenate(phases) if phases else np.zeros(1)
        d = self._states.shape[1]
        self._w = np.asarray(weights, np.float64) if weights is not None else np.ones(d)

    def initial_ctrl(self) -> np.ndarray:
        return self._actions[0]

    def act(self, qpos: np.ndarray, sim_time: float) -> np.ndarray:
        s = np.asarray(self._state_fn(np.asarray(qpos, np.float64)), np.float64)
        phase = min(max(sim_time / max(self.duration, 1e-3), 0.0), 1.0)
        # only consider demonstrated states within a forward phase window (keeps it moving)
        mask = (self._phase >= phase - self._pb) & (self._phase <= phase + self._pf)
        idx = np.where(mask)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(self._phase - phase)))])
        diff = (self._states[idx] - s) * self._w
        j = idx[int(np.argmin(np.einsum("ij,ij->i", diff, diff)))]
        return self._actions[j]
