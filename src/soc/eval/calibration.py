"""Calibration metrics for count forecasts (PIT, interval coverage) and for answer confidence.

Counts are discrete, so two notions of interval coverage are provided:

- `interval_coverage`: central intervals [lo, hi] on k from the forecast's CDF. Being discrete,
  each interval holds at least the nominal mass, so coverage of a calibrated forecast sits *above*
  nominal; the mean actual mass is returned alongside for comparison.
- `pit_coverage`: coverage from the randomised PIT, |u - 0.5| <= level / 2. Under a calibrated
  forecast this equals the nominal level exactly in expectation. Use this one for the gates.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from soc.forecast.scores import _as_batch, betabinom_cdf_grid, betabinom_pmf_grid


def randomized_pit(k, n, alpha, beta, rng: np.random.Generator):
    """u = F(k - 1) + V * P(k), V ~ U(0, 1). Uniform on [0, 1] iff the forecast is calibrated."""
    k, n, alpha, beta = _as_batch(k, n, alpha, beta)
    pmf = betabinom_pmf_grid(n, alpha, beta)
    cdf = betabinom_cdf_grid(n, alpha, beta)
    ki = k.astype(int)[..., None]
    p_k = np.take_along_axis(pmf, ki, axis=-1)[..., 0] / pmf.sum(axis=-1)
    f_below = np.where(k > 0, np.take_along_axis(cdf, np.maximum(ki - 1, 0), axis=-1)[..., 0], 0.0)
    return np.clip(f_below + rng.uniform(size=k.shape) * p_k, 0.0, 1.0)


def pit_histogram(u, bins: int = 10):
    """Bin counts of PIT values and the largest relative deviation of any bin from uniform."""
    u = np.asarray(u, dtype=float).ravel()
    counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    expected = u.size / bins
    return counts, float(np.max(np.abs(counts / expected - 1.0)))


def pit_coverage(u, level: float):
    """Fraction of PIT values inside the central `level` interval."""
    u = np.asarray(u, dtype=float)
    return float(np.mean(np.abs(u - 0.5) <= level / 2))


def central_interval(n, alpha, beta, level: float):
    """Central predictive interval [lo, hi] on k, and the probability mass it actually holds."""
    cdf = betabinom_cdf_grid(n, alpha, beta)
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    # Smallest j with F(j) >= q; small tolerance so F(j) == q exactly is not lost to rounding.
    lo = np.argmax(cdf >= lo_q - 1e-12, axis=-1)
    hi = np.argmax(cdf >= hi_q - 1e-12, axis=-1)
    f_hi = np.take_along_axis(cdf, hi[..., None], axis=-1)[..., 0]
    f_lo_below = np.where(
        lo > 0, np.take_along_axis(cdf, np.maximum(lo - 1, 0)[..., None], axis=-1)[..., 0], 0.0
    )
    return lo, hi, f_hi - f_lo_below


def interval_coverage(k, n, alpha, beta, level: float):
    """(empirical coverage, mean actual mass) of the central `level` intervals on k."""
    k, n, alpha, beta = _as_batch(k, n, alpha, beta)
    lo, hi, mass = central_interval(n, alpha, beta, level)
    return float(np.mean((k >= lo) & (k <= hi))), float(np.mean(mass))


# --- answer-level metrics (Stage 2) ---------------------------------------------------------


def brier(q, c):
    q, c = np.asarray(q, dtype=float), np.asarray(c, dtype=float)
    return float(np.mean(np.square(q - c)))


def ece(q, c, bins: int = 10):
    """Expected calibration error with equal-width bins; q = 1.0 falls in the last bin."""
    q, c = np.asarray(q, dtype=float), np.asarray(c, dtype=float)
    idx = np.minimum((q * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(q[mask].mean() - c[mask].mean())
    return float(total / q.size)


def auroc(q, c):
    """Area under the ROC curve of q for separating c = 1 from c = 0 (ties count one half)."""
    q, c = np.asarray(q, dtype=float), np.asarray(c).astype(bool)
    pos, neg = q[c], q[~c]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(q, kind="mergesort")
    ranks = np.empty(q.size)
    sorted_q = q[order]
    # average ranks over ties
    _, first, counts = np.unique(sorted_q, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[c].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


# --- uncertainty over questions ------------------------------------------------------------


def bootstrap_ci(
    stat: Callable[..., float],
    *arrays,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
):
    """Point estimate and percentile CI of `stat(*arrays)`, resampling questions (axis 0) jointly.

    For a paired comparison, pass both methods' per-question scores and a stat of their
    difference, e.g. `bootstrap_ci(lambda a, b: (a - b).mean(), nll_a, nll_b)`.
    """
    arrays = [np.asarray(a) for a in arrays]
    size = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, size, size=size)
        boots[i] = stat(*(a[idx] for a in arrays))
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(stat(*arrays)), float(lo), float(hi)
