"""Cross-Entropy Method (CEM) — a tiny black-box optimizer.

Optimizes a low-dimensional parameter vector against a scalar score. Used to tune a
:class:`~mjhri.learn.controller.PickPlaceController`'s parameters to a reward (the
reward-specification teaching method): the search space is ~7 interpretable knobs, so
a few hundred short rollouts converge in seconds on CPU — no gradients, no GPU.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def cem_optimize(
    score_fn: Callable[[np.ndarray], float],
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    n_iter: int = 8,
    pop: int = 24,
    elite_frac: float = 0.25,
    init: Optional[Sequence[float]] = None,
    init_std_frac: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Maximize ``score_fn(params)`` over the box ``[lo, hi]``.

    Returns ``(best_params, best_score)``. ``score_fn`` is called ``n_iter * pop`` times,
    so keep each call cheap (a handful of short rollouts).
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    dim = lo.shape[0]
    mean = np.asarray(init, np.float64) if init is not None else 0.5 * (lo + hi)
    std = init_std_frac * (hi - lo)
    n_elite = max(2, int(round(pop * elite_frac)))

    best_x, best_s = mean.copy(), -np.inf
    for _ in range(n_iter):
        samples = rng.normal(mean, std, size=(pop, dim))
        samples = np.clip(samples, lo, hi)
        scores = np.array([float(score_fn(x)) for x in samples])
        order = np.argsort(scores)[::-1]
        elite = samples[order[:n_elite]]
        if scores[order[0]] > best_s:
            best_s, best_x = float(scores[order[0]]), samples[order[0]].copy()
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6 * (hi - lo)  # keep a floor so it doesn't collapse
    return best_x, best_s
