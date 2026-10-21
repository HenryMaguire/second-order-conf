"""Stage 0 report: fit every head, score on test, and check the Stage 0 -> 1 gate (PLAN.md).

    uv run --extra train python -m soc.head.stage0 RUN_DIR [--layer last|middle]

RUN_DIR holds counts/, rollouts/ and features/ for bigmath_train/val/test (whatever prefix of
each split has been generated so far, so it can run on early batches). Prints a table and writes
stage0_report.json.

The empirical-Bayes test needs fresh rollouts beyond the ones a forecast is scored on; here the
32 rollouts per test question are split at random into r "observed" and 32 - r "held out", and
the posterior mean after the r is compared with the held-out solve rate. Binomial noise in the
held-out rate adds the same constant to every method, so differences between methods are fair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from soc.eval import bootstrap_ci, pit_coverage, pit_histogram, randomized_pit
from soc.forecast.beta import ab_to_ms
from soc.forecast.scores import betabinom_nll, rps
from soc.head.heads import Data, constant_head, fit_all

SPLITS = ("bigmath_train", "bigmath_val", "bigmath_test")
EB_R = (1, 2, 4, 8)


def load(run_dir: Path, split: str, layer: str) -> tuple[Data, pd.DataFrame]:
    counts = pd.read_parquet(run_dir / "counts" / f"{split}.parquet")
    f = np.load(run_dir / "features" / f"{split}.npz")
    idx = {q: i for i, q in enumerate(f["qids"])}
    rows = [idx[q] for q in counts.qid]  # features cover the whole split; counts a prefix
    x = f[layer][rows].astype(np.float32)
    rollouts = pd.concat(
        pd.read_parquet(p) for p in sorted((run_dir / "rollouts" / split).glob("shard-*.parquet"))
    )
    outcomes = (
        rollouts.sort_values(["qid", "rollout"])
        .groupby("qid")
        .correct.apply(lambda s: s.to_numpy(bool))
        .reindex(counts.qid)
    )
    return Data(x=x, k=counts.k.to_numpy(), n=counts.n.to_numpy()), outcomes


def empirical_bayes_error(alpha, beta, outcomes, r: int, rng, repeats: int = 20) -> np.ndarray:
    """Per-question squared error of the posterior mean after r observed rollouts, against the
    solve rate of the remaining rollouts; averaged over random splits of the rollouts."""
    errs = np.zeros(len(outcomes))
    for _ in range(repeats):
        for i, o in enumerate(outcomes):
            perm = rng.permutation(len(o))
            seen, rest = o[perm[:r]], o[perm[r:]]
            post = (alpha[i] + seen.sum()) / (alpha[i] + beta[i] + r)
            errs[i] += (post - rest.mean()) ** 2
    return errs / repeats


def score(name, alpha, beta, test: Data, outcomes, seed=0) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    nll = betabinom_nll(test.k, test.n, alpha, beta)
    u = randomized_pit(test.k, test.n, alpha, beta, rng)
    _, pit_dev = pit_histogram(u)
    _, s = ab_to_ms(alpha, beta)
    row = {
        "head": name,
        "nll": float(nll.mean()),
        "rps": float(rps(test.k, test.n, alpha, beta).mean()),
        "cov50": pit_coverage(u, 0.5),
        "cov80": pit_coverage(u, 0.8),
        "cov95": pit_coverage(u, 0.95),
        "pit_max_dev": pit_dev,
        "mean_s": float(s.mean()),
    }
    for r in EB_R:
        row[f"eb_mse_r{r}"] = float(empirical_bayes_error(alpha, beta, outcomes, r, rng).mean())
    return row, nll


def run(run_dir: Path, layer: str = "last") -> dict:
    (train, _), (val, _), (test, outcomes) = (load(run_dir, s, layer) for s in SPLITS)
    heads = fit_all(train, val)
    preds = {name: h.predict(test.x) for name, h in heads.items()}
    preds["constant"] = constant_head(train)(test.x)
    rows, nlls = [], {}
    for name, (a, b) in preds.items():
        row, nll = score(name, a, b, test, outcomes)
        rows.append(row)
        nlls[name] = nll

    rng = np.random.default_rng(1)
    eb = {
        name: empirical_bayes_error(*preds[name], outcomes, 4, rng)
        for name in ("linear_beta", "global_spread")
    }

    def diff(a, b):
        return bootstrap_ci(lambda x, y: float((x - y).mean()), nlls[a], nlls[b])

    vs_global, vs_difficulty = (
        diff("linear_beta", "global_spread"),
        diff("linear_beta", "difficulty_spread"),
    )
    eb_ci = bootstrap_ci(lambda x, y: float((x - y).mean()), eb["linear_beta"], eb["global_spread"])
    main = next(r for r in rows if r["head"] == "linear_beta")
    gate = {
        "nll_beats_global (CI < 0)": vs_global[2] < 0,
        "nll_beats_difficulty (CI < 0)": vs_difficulty[2] < 0,
        "cov80_in_75_85": 0.75 <= main["cov80"] <= 0.85,
        "pit_max_dev_le_0.30": main["pit_max_dev"] <= 0.30,
        "eb_r4_beats_global (CI < 0)": eb_ci[2] < 0,
    }
    return {
        "n_train": len(train.k),
        "n_val": len(val.k),
        "n_test": len(test.k),
        "layer": layer,
        "chosen_config (components, weight_decay)": {
            name: heads[name].config for name in ("point", "linear_beta", "mlp_beta")
        },
        "global_concentration": heads["global_spread"].c,
        "table": rows,
        "nll_diff_linear_minus_global": vs_global,
        "nll_diff_linear_minus_difficulty": vs_difficulty,
        "eb_r4_diff_linear_minus_global": eb_ci,
        "gate": gate,
        "gate_passed": all(gate.values()),
    }


def main(argv):
    run_dir = Path(argv[1])
    layer = argv[argv.index("--layer") + 1] if "--layer" in argv else "last"
    report = run(run_dir, layer)
    print(
        f"train/val/test = {report['n_train']}/{report['n_val']}/{report['n_test']}, "
        f"layer = {layer}\n"
    )
    print(pd.DataFrame(report["table"]).round(4).to_string(index=False))
    for key in (
        "nll_diff_linear_minus_global",
        "nll_diff_linear_minus_difficulty",
        "eb_r4_diff_linear_minus_global",
    ):
        est, lo, hi = report[key]
        print(f"{key}: {est:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("\ngate:", json.dumps(report["gate"], indent=1), "\npassed:", report["gate_passed"])
    (run_dir / f"stage0_report_{layer}.json").write_text(
        json.dumps(report, indent=1, default=float)
    )


if __name__ == "__main__":
    main(sys.argv)
