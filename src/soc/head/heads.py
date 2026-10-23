"""The six Stage 0 heads (PLAN.md), each mapping prompt features to a Beta over p_x.

Every head exposes `fit(train, val)` and `predict(x) -> (alpha, beta)`, so they are scored
identically. Rows 1-3 are the baselines the main method has to beat:

1. PointHead           m only, squared error to k/n (CALIBER-style). Scored with a near-zero
                       spread, i.e. as a Binomial(n, m) forecast.
2. GlobalSpreadHead    PointHead's m plus one concentration fitted on validation.
3. DifficultySpreadHead PointHead's m; log-concentration a quadratic in logit(m), so the spread
                       can depend on predicted difficulty and nothing else.
4. LinearBetaHead      linear map to (alpha, beta), Beta-Binomial NLL. The main method.
5. MLPBetaHead         2-layer MLP version: capacity check.
6. LaplaceBetaHead     Bayesian version: one global concentration for the irreducible spread,
                       plus the head's own (epistemic) uncertainty about m from a Laplace
                       approximation to the posterior over its weights. Its per-question spread
                       grows where the features are far from the training data.

Features are standardised on the training set. Weight decay is chosen on validation NLL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from soc.forecast.beta import ms_to_ab
from soc.forecast.scores import betabinom_nll, betabinom_nll_torch

C_MAX = 1e4
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
# Features are 2,048-dim and early batches have ~500 training questions, so heads see the top
# principal components (fitted on train); the count is chosen on validation with weight decay.
N_COMPONENTS = (16, 64, 256)


@dataclass
class Data:
    x: np.ndarray  # [N, D] features
    k: np.ndarray  # [N] successes
    n: np.ndarray  # [N] rollouts
    g: np.ndarray | None = None  # [N] group labels (e.g. ambiguous / single), if any


class _Standardiser:
    """Standardise, then (optionally) project onto the top principal components of the
    training features, rescaled to unit variance."""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, x):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.proj = None
        z = (x - self.mu) / self.sd
        k = self.n_components
        if k is not None and k < min(z.shape):
            _, sv, vt = np.linalg.svd(z, full_matrices=False)
            self.proj = vt[:k].T / (sv[:k] / np.sqrt(len(z)) + 1e-6)
        return self

    def __call__(self, x):
        z = (x - self.mu) / self.sd
        if self.proj is not None:
            z = z @ self.proj
        return torch.as_tensor(z, dtype=torch.float32)


def _configs(train: Data):
    """(n_components, weight_decay) grid, skipping component counts the data cannot support."""
    limit = min(train.x.shape[1], len(train.k) - 1)
    comps = [c for c in N_COMPONENTS if c < limit] or [None]
    return [(c, wd) for c in comps for wd in WEIGHT_DECAYS]


def _train(
    module, loss_fn, x, epochs=300, lr=1e-2, weight_decay=1e-3, seed=0, val_score=None, every=5
):
    """Full-batch AdamW (a few thousand rows fit in memory). With `val_score`, keep the weights
    from the epoch with the lowest validation score (early stopping) and return that score."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    best_score, best_state = float("inf"), None
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(module(x))
        loss.backward()
        opt.step()
        if val_score is not None and (epoch % every == every - 1 or epoch == epochs - 1):
            score = val_score()
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    if best_state is not None:
        module.load_state_dict(best_state)
    return best_score


def _mean_nll(alpha, beta, d: Data) -> float:
    return float(np.mean(betabinom_nll(d.k, d.n, alpha, beta)))


class PointHead:
    """Predicts m only. As a distribution it is a near-Binomial (concentration C_MAX)."""

    def fit(self, train: Data, val: Data):
        target = torch.as_tensor(train.k / train.n, dtype=torch.float32)
        best = None
        for comps, wd in _configs(train):
            self.std = _Standardiser(comps).fit(train.x)
            xt = self.std(train.x)
            net = nn.Linear(xt.shape[1], 1)
            self.net = net
            val_err = _train(
                net,
                lambda out: ((torch.sigmoid(out[:, 0]) - target) ** 2).mean(),
                xt,
                weight_decay=wd,
                val_score=lambda: float(np.mean((self.predict_m(val.x) - val.k / val.n) ** 2)),
            )
            if best is None or val_err < best[0]:
                best = (val_err, net, self.std, (comps, wd))
        _, self.net, self.std, self.config = best
        return self

    def predict_m(self, x) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self.net(self.std(x))[:, 0]).numpy()

    def predict(self, x):
        m = self.predict_m(x)
        return m * C_MAX, (1 - m) * C_MAX


class GlobalSpreadHead:
    """PointHead's m with one concentration c shared by every question, fitted on validation."""

    def __init__(self, point: PointHead):
        self.point = point

    def fit(self, train: Data, val: Data):
        m = np.clip(self.point.predict_m(val.x), 1e-3, 1 - 1e-3)
        grid = np.exp(np.linspace(np.log(0.1), np.log(C_MAX), 400))
        nlls = [_mean_nll(m * c, (1 - m) * c, val) for c in grid]
        self.c = float(grid[int(np.argmin(nlls))])
        return self

    def predict(self, x):
        m = np.clip(self.point.predict_m(x), 1e-3, 1 - 1e-3)
        return m * self.c, (1 - m) * self.c


class GroupSpreadHead:
    """PointHead's m with one concentration per group (e.g. ambiguous vs single-answer),
    fitted on validation. The honest baseline on mixed data: a per-question spread has to beat
    knowing which kind of question it is, not just one global spread."""

    def __init__(self, point: PointHead):
        self.point = point

    def fit(self, train: Data, val: Data):
        fallback = GlobalSpreadHead(self.point).fit(train, val).c
        self.c = {}
        for group in np.unique(val.g):
            mask = val.g == group
            sub = Data(x=val.x[mask], k=val.k[mask], n=val.n[mask])
            self.c[group] = (
                GlobalSpreadHead(self.point).fit(train, sub).c if mask.sum() else fallback
            )
        self.fallback = fallback
        return self

    def predict(self, x, g):
        m = np.clip(self.point.predict_m(x), 1e-3, 1 - 1e-3)
        c = np.array([self.c.get(group, self.fallback) for group in g])
        return m * c, (1 - m) * c


class DifficultySpreadHead:
    """log c = a + b * logit(m) + d * logit(m)^2: the spread may depend on predicted difficulty
    only. Fitted on the training set by Beta-Binomial NLL with m fixed from the point head."""

    def __init__(self, point: PointHead):
        self.point = point

    def _ab(self, m, params):
        z = torch.logit(m)
        c = torch.exp(params[0] + params[1] * z + params[2] * z**2).clamp(0.1, C_MAX)
        return m * c, (1 - m) * c

    def fit(self, train: Data, val: Data):
        m = torch.as_tensor(np.clip(self.point.predict_m(train.x), 1e-3, 1 - 1e-3))
        k = torch.as_tensor(np.array(train.k, dtype=np.float64))
        n = torch.as_tensor(np.array(train.n, dtype=np.float64))
        params = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            params[0] = 1.0
        opt = torch.optim.Adam([params], lr=0.05)
        for _ in range(500):
            opt.zero_grad()
            a, b = self._ab(m.double(), params)
            betabinom_nll_torch(k, n, a, b).mean().backward()
            opt.step()
        self.params = params.detach()
        return self

    def predict(self, x):
        m = torch.as_tensor(np.clip(self.point.predict_m(x), 1e-3, 1 - 1e-3)).double()
        a, b = self._ab(m, self.params)
        return a.numpy(), b.numpy()


class _BetaNet(nn.Module):
    def __init__(self, dim, hidden: int | None):
        super().__init__()
        self.body = (
            nn.Identity() if hidden is None else nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        )
        self.out = nn.Linear(dim if hidden is None else hidden, 2)

    def forward(self, x):
        z = self.out(self.body(x))
        # Mean via sigmoid, concentration via exp: an easier parametrisation to optimise than
        # raw (alpha, beta), and it bounds both away from zero.
        m = torch.sigmoid(z[:, 0]).clamp(1e-3, 1 - 1e-3)
        c = torch.exp(z[:, 1].clamp(-2.3, 9.2))  # c in [0.1, ~1e4]
        return m * c, (1 - m) * c


class LinearBetaHead:
    hidden: int | None = None

    def fit(self, train: Data, val: Data):
        k, n = torch.as_tensor(train.k * 1.0), torch.as_tensor(train.n * 1.0)
        best = None
        for comps, wd in _configs(train):
            self.std = _Standardiser(comps).fit(train.x)
            xt = self.std(train.x)
            net = _BetaNet(xt.shape[1], self.hidden)
            with torch.no_grad():
                net.out.bias[1] = 2.0  # start at a moderate concentration
            self.net = net
            val_nll = _train(
                net,
                lambda ab: betabinom_nll_torch(k, n, ab[0], ab[1]).mean(),
                xt,
                weight_decay=wd,
                val_score=lambda: _mean_nll(*self.predict(val.x), val),
            )
            if best is None or val_nll < best[0]:
                best = (val_nll, net, self.std, (comps, wd))
        _, self.net, self.std, self.config = best
        return self

    def predict(self, x):
        with torch.no_grad():
            a, b = self.net(self.std(x))
        return a.double().numpy(), b.double().numpy()


class MLPBetaHead(LinearBetaHead):
    hidden = 256


class LaplaceBetaHead:
    """Bayesian logistic-mean head with a Laplace posterior over its weights.

    Model: p ~ Beta(m c, (1 - m) c) with m = sigmoid(w . z + b) on PCA features z and one
    global concentration c; k ~ Binomial(n, p). The MAP fit minimises the Beta-Binomial NLL plus
    a Gaussian prior (precision `prior`) on w. The Laplace approximation gives a Gaussian
    posterior over (w, b) with covariance H^-1 (H = Hessian at the MAP, prior included).

    Predictive spread (law of total variance over posterior samples of m):
        Var(p | x) = E[m (1 - m)] / (c + 1) + Var[m]
    i.e. the global irreducible spread plus the head's epistemic uncertainty about m. The
    result is moment-matched to a Beta so it can be stated as m +- s.
    """

    PRIORS = (1e-2, 1e-1, 1.0, 10.0, 100.0)
    n_samples = 512

    def _fit_map(self, z, k, n, prior):
        d = z.shape[1]
        theta = torch.zeros(d + 2, dtype=torch.float64, requires_grad=True)  # w, b, log c
        with torch.no_grad():
            theta[-1] = 1.0
        opt = torch.optim.LBFGS([theta], max_iter=500, line_search_fn="strong_wolfe")

        def loss_fn(t):
            m = torch.sigmoid(z @ t[:d] + t[d]).clamp(1e-4, 1 - 1e-4)
            c = torch.exp(t[-1].clamp(-2.3, 9.2))
            nll = betabinom_nll_torch(k, n, m * c, (1 - m) * c).sum()
            return nll + 0.5 * prior * (t[:d] ** 2).sum()

        def closure():
            opt.zero_grad()
            loss = loss_fn(theta)
            loss.backward()
            return loss

        opt.step(closure)
        theta = theta.detach()
        # Laplace: Hessian of the full objective w.r.t. (w, b), with log c held at its MAP.
        log_c = theta[-1]

        def wb_loss(wb):
            return loss_fn(torch.cat([wb, log_c.reshape(1)]))

        hess = torch.autograd.functional.hessian(wb_loss, theta[:-1])
        hess = 0.5 * (hess + hess.T) + 1e-6 * torch.eye(d + 1, dtype=torch.float64)
        return theta, torch.linalg.inv(hess)

    def _moments(self, x):
        z = self.std(x).double()
        z1 = torch.cat([z, torch.ones(len(z), 1, dtype=torch.float64)], 1)
        gen = torch.Generator().manual_seed(0)
        eps = torch.randn(self.n_samples, len(self.wb), generator=gen, dtype=torch.float64)
        samples = self.wb + eps @ self.chol.T  # [S, d + 1] posterior draws of (w, b)
        m = torch.sigmoid(z1 @ samples.T).clamp(1e-4, 1 - 1e-4)  # [N, S]
        m_bar = m.mean(1)
        epistemic = m.var(1)
        total = (m * (1 - m)).mean(1) / (self.c + 1) + epistemic
        return m_bar.numpy(), np.sqrt(total.numpy()), epistemic.numpy()

    def fit(self, train: Data, val: Data):
        k = torch.as_tensor(np.array(train.k, dtype=np.float64))
        n = torch.as_tensor(np.array(train.n, dtype=np.float64))
        limit = min(train.x.shape[1], len(train.k) - 1)
        comps = [c for c in N_COMPONENTS if c < limit] or [None]
        best = None
        for n_comp in comps:
            std = _Standardiser(n_comp).fit(train.x)
            z = std(train.x).double()
            for prior in self.PRIORS:
                theta, cov = self._fit_map(z, k, n, prior)
                self.std, self.wb, self.c = std, theta[:-1], float(torch.exp(theta[-1]))
                self.chol = torch.linalg.cholesky(cov)
                val_nll = _mean_nll(*self.predict(val.x), val)
                if best is None or val_nll < best[0]:
                    best = (val_nll, std, self.wb, self.c, self.chol, (n_comp, prior))
        _, self.std, self.wb, self.c, self.chol, self.config = best
        return self

    def predict(self, x):
        m, s, _ = self._moments(x)
        return ms_to_ab(m, s)

    def epistemic_sd(self, x) -> np.ndarray:
        """Standard deviation of m across the weight posterior (the per-question part)."""
        return np.sqrt(self._moments(x)[2])


def fit_all(train: Data, val: Data) -> dict:
    point = PointHead().fit(train, val)
    return {
        "point": point,
        "global_spread": GlobalSpreadHead(point).fit(train, val),
        **(
            {"group_spread": GroupSpreadHead(point).fit(train, val)}
            if val.g is not None and len(np.unique(val.g)) > 1
            else {}
        ),
        "difficulty_spread": DifficultySpreadHead(point).fit(train, val),
        "linear_beta": LinearBetaHead().fit(train, val),
        "mlp_beta": MLPBetaHead().fit(train, val),
        "laplace_beta": LaplaceBetaHead().fit(train, val),
    }


def constant_head(train: Data):
    """Reference with no features at all: one Beta for every question (method of moments)."""
    p = train.k / train.n
    m, v = p.mean(), p.var()
    s = np.sqrt(max(v - m * (1 - m) / train.n.mean(), 1e-6))
    a, b = ms_to_ab(m, s)
    return lambda x: (np.full(len(x), float(a)), np.full(len(x), float(b)))
