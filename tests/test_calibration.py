import itertools

import numpy as np
import pytest
from scipy.stats import kstest

from soc.eval import (
    auroc,
    bootstrap_ci,
    brier,
    central_interval,
    ece,
    interval_coverage,
    pit_coverage,
    pit_histogram,
    randomized_pit,
)
from soc.forecast import ms_to_ab

N = 32


def simulate(num=20000, seed=0):
    """p ~ Beta(a_i, b_i), k ~ Binomial(N, p), so the forecast (a_i, b_i) is calibrated."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.1, 0.9, num)
    s = rng.uniform(0.02, 0.9, num) * np.sqrt(m * (1 - m))
    a, b = ms_to_ab(m, s)
    k = rng.binomial(N, rng.beta(a, b))
    return k, a, b, m, s, rng


def test_pit_uniform_when_calibrated():
    k, a, b, *_, rng = simulate()
    u = randomized_pit(k, N, a, b, rng)
    assert kstest(u, "uniform").pvalue > 0.01
    _, max_dev = pit_histogram(u)
    assert max_dev < 0.1


def test_pit_u_shaped_when_too_narrow():
    k, _, _, m, s, rng = simulate()
    a, b = ms_to_ab(m, 0.2 * s)
    u = randomized_pit(k, N, a, b, rng)
    counts, _ = pit_histogram(u)
    assert counts[0] > counts[4] and counts[-1] > counts[5]
    assert pit_coverage(u, 0.8) < 0.75


def test_pit_humped_when_too_wide():
    k, _, _, m, s, rng = simulate()
    a, b = ms_to_ab(m, np.minimum(2.5 * s, 0.95 * np.sqrt(m * (1 - m))))
    u = randomized_pit(k, N, a, b, rng)
    counts, _ = pit_histogram(u)
    assert counts[4] > counts[0] and counts[5] > counts[-1]
    assert pit_coverage(u, 0.8) > 0.85


@pytest.mark.parametrize("level", [0.5, 0.8, 0.95])
def test_pit_coverage_matches_nominal(level):
    k, a, b, *_, rng = simulate()
    assert abs(pit_coverage(randomized_pit(k, N, a, b, rng), level) - level) < 0.015


@pytest.mark.parametrize("level", [0.5, 0.8, 0.95])
def test_discrete_intervals_cover_at_least_nominal(level):
    k, a, b, *_ = simulate()
    cov, mass = interval_coverage(k, N, a, b, level)
    assert mass >= level - 1e-9
    assert abs(cov - mass) < 0.015


def test_central_interval_simple_case():
    # Near-Binomial(32, 0.5): the central 95% interval is roughly 11..21.
    a, b = ms_to_ab(0.5, 0.0)
    lo, hi, mass = central_interval(N, np.array([a]), np.array([b]), 0.95)
    assert 10 <= lo[0] <= 12 and 20 <= hi[0] <= 22 and mass[0] >= 0.95


def test_pit_edge_counts():
    rng = np.random.default_rng(0)
    a, b = ms_to_ab(np.array([0.5, 0.5]), np.array([0.1, 0.1]))
    u = randomized_pit(np.array([0, N]), N, a, b, rng)
    assert 0 <= u[0] < 0.01 and 0.99 < u[1] <= 1


def test_brier_and_ece():
    q = np.array([0.9, 0.9, 0.1, 0.1])
    c = np.array([1, 1, 0, 0])
    assert np.isclose(brier(q, c), 0.01)
    assert np.isclose(ece(q, c), 0.1)
    assert np.isclose(ece(np.array([1.0, 0.0]), np.array([1, 0])), 0.0)


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(3)
    q = np.round(rng.uniform(size=200), 1)  # plenty of ties
    c = rng.uniform(size=200) < q
    pairs = [
        1.0 if qp > qn else 0.5 if qp == qn else 0.0 for qp, qn in itertools.product(q[c], q[~c])
    ]
    assert np.isclose(auroc(q, c), np.mean(pairs))


def test_bootstrap_ci_contains_estimate():
    x = np.random.default_rng(0).normal(1.0, 1.0, 500)
    est, lo, hi = bootstrap_ci(np.mean, x, n_boot=500)
    assert lo < est < hi and lo < 1.0 < hi
    d_est, d_lo, _ = bootstrap_ci(lambda a, b: (a - b).mean(), x + 0.5, x, n_boot=200)
    assert np.isclose(d_est, 0.5) and d_lo > 0.49
