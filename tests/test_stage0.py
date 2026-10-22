import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from soc.head import stage0  # noqa: E402
from soc.head.features import final_token_states  # noqa: E402

N_ROLLOUTS = 32


def make_run(tmp_path, informative_spread: bool, seed=0, sizes=(1500, 400, 600), dim=16):
    """Synthetic run dir: features drive the mean and (optionally) the concentration of p."""
    rng = np.random.default_rng(seed)
    w_m, w_c = rng.normal(size=dim), rng.normal(size=dim)
    mix = rng.normal(size=(8, dim))
    for split, size in zip(stage0.SPLITS, sizes, strict=True):
        # Hidden states are anisotropic: a few strong latent directions (which carry the
        # signal) mixed into `dim` features, plus isotropic noise.
        z = rng.normal(size=(size, 8))
        x = (z @ mix + 0.5 * rng.normal(size=(size, dim))).astype(np.float32)
        m = 1 / (1 + np.exp(-(z[:, :4] @ w_m[:4]) / 2))
        log_c = (z[:, 4:8] @ w_c[4:8]) / 2 if informative_spread else np.zeros(size)
        c = np.exp(1.0 + np.clip(log_c, -2, 3))
        p = rng.beta(m * c, (1 - m) * c)
        outcomes = rng.random((size, N_ROLLOUTS)) < p[:, None]
        qids = [f"{split}-{i}" for i in range(size)]
        (tmp_path / "rollouts" / split).mkdir(parents=True)
        (tmp_path / "counts").mkdir(exist_ok=True)
        (tmp_path / "features").mkdir(exist_ok=True)
        pd.DataFrame(
            {
                "qid": np.repeat(qids, N_ROLLOUTS),
                "rollout": np.tile(np.arange(N_ROLLOUTS), size),
                "correct": outcomes.ravel(),
            }
        ).to_parquet(tmp_path / "rollouts" / split / "shard-00000.parquet")
        pd.DataFrame({"qid": qids, "k": outcomes.sum(1), "n": N_ROLLOUTS}).to_parquet(
            tmp_path / "counts" / f"{split}.parquet"
        )
        # Features cover more questions than the counts (a later batch's questions): ignored.
        extra = rng.normal(size=(50, dim)).astype(np.float32)
        np.savez(
            tmp_path / "features" / f"{split}.npz",
            qids=np.array(qids + [f"later-{i}" for i in range(50)]),
            last=np.concatenate([x, extra]),
            middle=np.concatenate([x, extra]),
        )
    return tmp_path


def test_informative_spread_passes_gate(tmp_path):
    report = stage0.run(make_run(tmp_path, informative_spread=True))
    assert report["nll_diff_linear_minus_global"][2] < 0
    assert report["nll_diff_linear_minus_difficulty"][2] < 0
    table = {r["head"]: r for r in report["table"]}
    assert table["linear_beta"]["nll"] < table["global_spread"]["nll"] < table["constant"]["nll"]
    assert 0.75 <= table["linear_beta"]["cov80"] <= 0.85
    assert report["gate_passed"], report["gate"]


def test_uninformative_spread_gives_no_real_gain(tmp_path):
    report = stage0.run(make_run(tmp_path, informative_spread=False, seed=3))
    est, lo, hi = report["nll_diff_linear_minus_global"]
    assert est > -0.02  # no meaningful advantage when the true spread is constant


def test_small_high_dimensional_run_like_batch_1(tmp_path):
    report = stage0.run(
        make_run(tmp_path, informative_spread=True, seed=2, sizes=(500, 250, 250), dim=512)
    )
    table = {r["head"]: r for r in report["table"]}
    assert table["linear_beta"]["nll"] < table["constant"]["nll"]
    assert report["nll_diff_linear_minus_global"][0] < 0


def test_point_head_scored_as_binomial_is_overconfident(tmp_path):
    report = stage0.run(make_run(tmp_path, informative_spread=True, seed=1))
    point = next(r for r in report["table"] if r["head"] == "point")
    assert point["cov80"] < 0.7  # spread-free forecasts under-cover on overdispersed counts


def test_final_token_states_takes_last_position():
    import torch

    hs = tuple(
        torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) + 100 * i for i in range(3)
    )
    out = final_token_states(hs, [0, 2])
    assert out[2].shape == (2, 4)
    assert np.allclose(out[2][0], hs[2][0, -1].numpy())


def test_empirical_bayes_comparison_is_paired():
    rng = np.random.default_rng(0)
    outcomes = pd.Series([rng.random(32) < 0.4 for _ in range(30)])
    a, b = np.full(30, 2.0), np.full(30, 3.0)
    e1 = stage0.empirical_bayes_error(a, b, outcomes, 4)
    e2 = stage0.empirical_bayes_error(a, b, outcomes, 4)
    assert np.array_equal(e1, e2)  # same splits for every head
