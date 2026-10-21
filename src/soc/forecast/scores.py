"""Beta-Binomial predictive distribution of k successes in n rollouts, and its proper scores.

Stage 0 trains on the negative log-likelihood; the RL stages use the ranked probability score

    RPS = 1/(n+1) * sum_{j=0}^{n} (F(j) - 1[j >= k])^2   in [0, 1],

with reward -RPS in [-1, 0]. Everything is vectorised over a batch of forecasts; n may be a
scalar or per-item. Per-item n is handled by padding the count grid to max(n): padded cells have
F(j) = 1 = 1[j >= k], so they add nothing to the RPS sum.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, gammaln


def _as_batch(k, n, alpha, beta):
    k, n, alpha, beta = np.broadcast_arrays(
        np.asarray(k, dtype=float),
        np.asarray(n, dtype=float),
        np.asarray(alpha, dtype=float),
        np.asarray(beta, dtype=float),
    )
    if np.any(k < 0) or np.any(k > n) or np.any(k != np.round(k)) or np.any(n != np.round(n)):
        raise ValueError("k and n must be integers with 0 <= k <= n")
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("alpha and beta must be positive")
    return k, n, alpha, beta


def betabinom_logpmf(k, n, alpha, beta):
    """log P(k | n, alpha, beta) under the Beta-Binomial. Broadcasts."""
    k, n, alpha, beta = _as_batch(k, n, alpha, beta)
    log_choose = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return log_choose + betaln(k + alpha, n - k + beta) - betaln(alpha, beta)


def betabinom_nll(k, n, alpha, beta):
    """Negative log-likelihood of observed k, per item."""
    return -betabinom_logpmf(k, n, alpha, beta)


def betabinom_pmf_grid(n, alpha, beta):
    """PMF over j = 0..max(n) for each item, shape (..., max(n) + 1); zero beyond each item's n."""
    n, alpha, beta = np.broadcast_arrays(
        np.asarray(n, dtype=float), np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    )
    j = np.arange(int(n.max()) + 1, dtype=float)
    nn, jj = n[..., None], j
    valid = jj <= nn
    jj_safe = np.where(valid, jj, 0.0)
    log_choose = gammaln(nn + 1) - gammaln(jj_safe + 1) - gammaln(nn - jj_safe + 1)
    logp = (
        log_choose
        + betaln(jj_safe + alpha[..., None], nn - jj_safe + beta[..., None])
        - betaln(alpha, beta)[..., None]
    )
    return np.where(valid, np.exp(logp), 0.0)


def betabinom_cdf_grid(n, alpha, beta):
    """CDF F(j) for j = 0..max(n); renormalised so F(n) = 1 exactly, and 1 beyond n."""
    pmf = betabinom_pmf_grid(n, alpha, beta)
    cdf = np.cumsum(pmf, axis=-1)
    return np.clip(cdf / cdf[..., -1:], 0.0, 1.0)


def rps(k, n, alpha, beta):
    """Ranked probability score of observed k, per item, in [0, 1]. Lower is better."""
    k, n, alpha, beta = _as_batch(k, n, alpha, beta)
    cdf = betabinom_cdf_grid(n, alpha, beta)
    j = np.arange(cdf.shape[-1])
    step = (j >= k[..., None]).astype(float)
    return np.sum(np.square(cdf - step), axis=-1) / (n + 1)


def rps_reward(k, n, alpha, beta):
    """RL reward: -RPS, in [-1, 0]."""
    return -rps(k, n, alpha, beta)


def leave_one_out_counts(correct):
    """For a group of rollouts' 0/1 outcomes (last axis), successes among the *other* rollouts.

    Stage 2 scores rollout i's forecast against k_{-i} out of n - 1, so its own outcome never
    enters its target.
    """
    correct = np.asarray(correct, dtype=float)
    return correct.sum(axis=-1, keepdims=True) - correct


try:  # torch is optional (Stage 0 training only)
    import torch

    def betabinom_nll_torch(k, n, alpha, beta):
        """Differentiable Beta-Binomial NLL in torch, per item; matches betabinom_nll."""

        def lbeta(a, b):
            return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

        k = torch.as_tensor(k, dtype=alpha.dtype, device=alpha.device)
        n = torch.as_tensor(n, dtype=alpha.dtype, device=alpha.device)
        log_choose = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        return -(log_choose + lbeta(k + alpha, n - k + beta) - lbeta(alpha, beta))

except ImportError:  # pragma: no cover
    pass
