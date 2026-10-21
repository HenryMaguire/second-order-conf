import numpy as np
import pytest
from scipy.stats import betabinom

from soc.forecast import (
    betabinom_cdf_grid,
    betabinom_nll,
    betabinom_pmf_grid,
    leave_one_out_counts,
    ms_to_ab,
    rps,
    rps_reward,
)

RNG = np.random.default_rng(1234)
A = RNG.uniform(0.1, 80, 300)
B = RNG.uniform(0.1, 80, 300)


@pytest.mark.parametrize("n", [1, 8, 31, 32, 64])
def test_nll_matches_scipy(n):
    k = RNG.integers(0, n + 1, A.size)
    assert np.allclose(betabinom_nll(k, n, A, B), -betabinom.logpmf(k, n, A, B), rtol=1e-10)


def test_nll_extreme_params_finite():
    for a, b in [ms_to_ab(0.0, 0.0), ms_to_ab(1.0, 0.0), ms_to_ab(0.5, 0.5)]:
        for k in [0, 16, 32]:
            assert np.isfinite(betabinom_nll(k, 32, a, b))


def test_pmf_and_cdf_match_scipy():
    n = 32
    pmf, cdf = betabinom_pmf_grid(n, A, B), betabinom_cdf_grid(n, A, B)
    j = np.arange(n + 1)
    assert np.allclose(pmf, betabinom.pmf(j, n, A[:, None], B[:, None]), atol=1e-12)
    assert np.allclose(pmf.sum(-1), 1.0)
    assert np.allclose(cdf, betabinom.cdf(j, n, A[:, None], B[:, None]), atol=1e-9)


def rps_bruteforce(k, n, a, b):
    j = np.arange(n + 1)
    return np.mean((betabinom.cdf(j, n, a, b) - (j >= k)) ** 2)


def test_rps_matches_formula():
    n = 32
    k = RNG.integers(0, n + 1, A.size)
    ours = rps(k, n, A, B)
    ref = np.array([rps_bruteforce(ki, n, a, b) for ki, a, b in zip(k, A, B, strict=True)])
    assert np.allclose(ours, ref, atol=1e-10)


def test_rps_bounded_and_reward_bounded():
    for n in [1, 31, 32]:
        k = RNG.integers(0, n + 1, A.size)
        r = rps(k, n, A, B)
        assert np.all(r >= 0) and np.all(r <= 1)
        rew = rps_reward(k, n, A, B)
        assert np.all(rew >= -1) and np.all(rew <= 0)


def test_rps_extremes():
    # Confident and right -> near 0; confident and maximally wrong -> n/(n+1), the sup.
    n = 32
    a0, b0 = ms_to_ab(0.0, 0.0)
    assert rps(0, n, a0, b0) < 1e-3
    # (m is clamped to 1e-3, so a little mass leaks above 0.)
    assert np.isclose(rps(n, n, a0, b0), n / (n + 1), atol=1e-2)


def test_per_item_n_matches_scalar_n():
    n = np.array([31, 32, 8, 64] * 75)
    k = np.array([RNG.integers(0, ni + 1) for ni in n])
    batched = rps(k, n, A, B)
    single = np.array([rps(ki, ni, a, b) for ki, ni, a, b in zip(k, n, A, B, strict=True)])
    assert np.allclose(batched, single)
    assert np.allclose(betabinom_nll(k, n, A, B), -betabinom.logpmf(k, n, A, B))


@pytest.mark.parametrize("score", [rps, betabinom_nll])
def test_scores_are_proper(score):
    """Expected score under the true Beta-Binomial is minimised by forecasting it."""
    n = 32
    j = np.arange(n + 1)
    for m_true, s_true in [(0.3, 0.1), (0.7, 0.25), (0.5, 0.02)]:
        a, b = ms_to_ab(m_true, s_true)
        p = betabinom.pmf(j, n, a, b)

        def expected(m, s, p=p):
            a2, b2 = ms_to_ab(m, s)
            return np.sum(p * score(j, n, a2, b2))

        best = expected(m_true, s_true)
        for dm, ds in [(0.05, 0), (-0.05, 0), (0, 0.04), (0, -0.015), (0.1, 0.1)]:
            assert expected(m_true + dm, max(s_true + ds, 0.0)) > best


def test_spread_matters_for_rps():
    """With p truly dispersed, a too-narrow forecast with the right mean scores worse."""
    n = 32
    j = np.arange(n + 1)
    a, b = ms_to_ab(0.5, 0.3)
    p = betabinom.pmf(j, n, a, b)
    wide = np.sum(p * rps(j, n, a, b))
    narrow = np.sum(p * rps(j, n, *ms_to_ab(0.5, 0.01)))
    assert wide < narrow


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        rps(33, 32, 1.0, 1.0)
    with pytest.raises(ValueError):
        betabinom_nll(-1, 32, 1.0, 1.0)
    with pytest.raises(ValueError):
        rps(1.5, 32, 1.0, 1.0)
    with pytest.raises(ValueError):
        rps(1, 32, 0.0, 1.0)


def test_leave_one_out_counts():
    c = np.array([[1, 0, 1, 1], [0, 0, 0, 0]])
    assert leave_one_out_counts(c).tolist() == [[2, 3, 2, 2], [0, 0, 0, 0]]


def test_torch_nll_matches_numpy():
    torch = pytest.importorskip("torch")
    from soc.forecast.scores import betabinom_nll_torch

    n = 32
    k = RNG.integers(0, n + 1, A.size)
    a = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(B, dtype=torch.float64, requires_grad=True)
    out = betabinom_nll_torch(torch.tensor(k), n, a, b)
    assert np.allclose(out.detach().numpy(), betabinom_nll(k, n, A, B), rtol=1e-10)
    out.sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    # Spot-check gradient against finite differences.
    eps = 1e-6
    i = 7
    up = betabinom_nll(k[i], n, A[i] + eps, B[i])
    down = betabinom_nll(k[i], n, A[i] - eps, B[i])
    assert np.isclose(a.grad[i].item(), (up - down) / (2 * eps), rtol=1e-5)
