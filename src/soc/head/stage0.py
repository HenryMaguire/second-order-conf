"""Stage 0 report: fit every head, score on test, and check the Stage 0 -> 1 gate (PLAN.md).

    uv run --extra train python -m soc.head.stage0 RUN_DIR [--layer last|middle]
        [--splits bigmath|ambigqa] [--scoring nq|any|nq_contains|any_contains]

RUN_DIR holds counts/, rollouts/ and features/ for <prefix>_train/val/test (whatever prefix of
each split has been generated so far, so it can run on early batches). Prints a table and writes
stage0_report_<prefix>_<scoring>_<layer>.json.

Groups come from the qid (aqa-amb-* / aqa-sgl-*), so on AmbigQA there is a per-group spread
baseline and per-group results. `--scoring any` (AmbigQA) re-scores every rollout as correct if
it matches the answer to any valid reading, not just the original NQ answer.

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


def splits_for(prefix: str) -> tuple[str, str, str]:
    return (f"{prefix}_train", f"{prefix}_val", f"{prefix}_test")


def group_of(qid: str) -> str:
    """aqa-amb-123 -> 'amb'; anything without a group tag -> 'all'."""
    parts = qid.split("-")
    return parts[1] if parts[0] == "aqa" and len(parts) > 2 else "all"


def rescore(rollouts: pd.DataFrame, split: str, scoring: str) -> pd.Series:
    """Re-score stored rollouts. scoring = <golds>[_contains]:
    golds "nq" = the original answer and its aliases, "any" = any valid reading's answers;
    "_contains" = lenient whole-word containment instead of exact match."""
    from soc.data import load_split
    from soc.verify.checks import contains_match, exact_match

    which, _, mode = scoring.partition("_")
    match = contains_match if mode == "contains" else exact_match
    golds = {
        q.qid: [
            *q.aliases,
            *((a for rd in q.meta.get("readings", []) for a in rd) if which == "any" else ()),
        ]
        for q in load_split(split)
    }
    return pd.Series(
        [
            isinstance(a, str) and match(a, golds[q])
            for q, a in zip(rollouts.qid, rollouts.answer, strict=True)
        ],
        index=rollouts.index,
    )


def load(run_dir: Path, split: str, layer: str, scoring: str = "nq") -> tuple[Data, pd.DataFrame]:
    counts = pd.read_parquet(run_dir / "counts" / f"{split}.parquet")
    f = np.load(run_dir / "features" / f"{split}.npz")
    idx = {q: i for i, q in enumerate(f["qids"])}
    rows = [idx[q] for q in counts.qid]  # features cover the whole split; counts a prefix
    x = f[layer][rows].astype(np.float32)
    rollouts = pd.concat(
        pd.read_parquet(p) for p in sorted((run_dir / "rollouts" / split).glob("shard-*.parquet"))
    )
    if scoring != "nq":
        rollouts = rollouts.assign(correct=rescore(rollouts, split, scoring))
        k = rollouts.groupby("qid").correct.sum().reindex(counts.qid)
        counts = counts.assign(k=k.to_numpy().astype(int))
    outcomes = (
        rollouts.sort_values(["qid", "rollout"])
        .groupby("qid")
        .correct.apply(lambda s: s.to_numpy(bool))
        .reindex(counts.qid)
    )
    g = np.array([group_of(q) for q in counts.qid])
    return Data(x=x, k=counts.k.to_numpy(), n=counts.n.to_numpy(), g=g), outcomes


def empirical_bayes_error(alpha, beta, outcomes, r: int, seed: int = 0, repeats: int = 20):
    """Per-question squared error of the posterior mean after r observed rollouts, against the
    solve rate of the remaining rollouts; averaged over random splits of the rollouts. The splits
    depend only on `seed`, so every head is scored on the same splits (a paired comparison)."""
    rng = np.random.default_rng(seed)
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
        row[f"eb_mse_r{r}"] = float(empirical_bayes_error(alpha, beta, outcomes, r).mean())
    return row, nll


def dispersion_by_group(d: Data) -> dict:
    """Model-free look at each group's solve rates: how many are intermediate, and the
    Beta-Binomial concentration that best fits the group with no features (lower = more
    spread out between questions)."""
    out = {}
    for group in np.unique(d.g):
        mask = d.g == group
        k, n = d.k[mask], d.n[mask]
        p = k / n
        m = p.mean()
        grid = np.exp(np.linspace(np.log(0.05), np.log(1e4), 400))
        nll = [betabinom_nll(k, n, m * c, (1 - m) * c).mean() for c in grid]
        out[str(group)] = {
            "n_questions": int(mask.sum()),
            "mean_p": float(m),
            "share_k0": float((k == 0).mean()),
            "share_kn": float((k == n).mean()),
            "share_intermediate (0.2<p<0.8)": float(((p > 0.2) & (p < 0.8)).mean()),
            "best_constant_concentration": float(grid[int(np.argmin(nll))]),
        }
    return out


def run(
    run_dir: Path, layer: str = "last", splits: tuple[str, str, str] = SPLITS, scoring: str = "nq"
) -> dict:
    (train, _), (val, _), (test, outcomes) = (load(run_dir, s, layer, scoring) for s in splits)
    heads = fit_all(train, val)
    preds = {
        name: h.predict(test.x, test.g) if name == "group_spread" else h.predict(test.x)
        for name, h in heads.items()
    }
    preds["constant"] = constant_head(train)(test.x)
    rows, nlls = [], {}
    for name, (a, b) in preds.items():
        row, nll = score(name, a, b, test, outcomes)
        rows.append(row)
        nlls[name] = nll

    eb = {
        name: empirical_bayes_error(*preds[name], outcomes, 4)
        for name in ("linear_beta", "global_spread")
    }

    def diff(a, b):
        return bootstrap_ci(lambda x, y: float((x - y).mean()), nlls[a], nlls[b])

    laplace_vs_global = diff("laplace_beta", "global_spread")
    vs_global, vs_difficulty = (
        diff("linear_beta", "global_spread"),
        diff("linear_beta", "difficulty_spread"),
    )
    eb_ci = bootstrap_ci(lambda x, y: float((x - y).mean()), eb["linear_beta"], eb["global_spread"])
    # Per-group test NLL, and the comparisons against the per-group baseline when there is one.
    per_group = {
        str(group): {name: float(nll[test.g == group].mean()) for name, nll in nlls.items()}
        for group in np.unique(test.g)
    }
    vs_group = {}
    if "group_spread" in nlls:
        vs_group = {
            f"nll_diff_{name}_minus_group": diff(name, "group_spread")
            for name in ("linear_beta", "mlp_beta", "laplace_beta", "global_spread")
        }
    all_data = Data(
        x=np.zeros((0, 1)),
        k=np.concatenate([train.k, val.k, test.k]),
        n=np.concatenate([train.n, val.n, test.n]),
        g=np.concatenate([train.g, val.g, test.g]),
    )
    main = next(r for r in rows if r["head"] == "linear_beta")
    gate = {
        "nll_beats_global (CI < 0)": vs_global[2] < 0,
        "nll_beats_difficulty (CI < 0)": vs_difficulty[2] < 0,
        "cov80_in_75_85": 0.75 <= main["cov80"] <= 0.85,
        "pit_max_dev_le_0.30": main["pit_max_dev"] <= 0.30,
        "eb_r4_beats_global (CI < 0)": eb_ci[2] < 0,
    }
    if vs_group:
        gate["nll_beats_group (CI < 0)"] = vs_group["nll_diff_linear_beta_minus_group"][2] < 0
    return {
        "n_train": len(train.k),
        "n_val": len(val.k),
        "n_test": len(test.k),
        "layer": layer,
        "chosen_config (components, weight_decay)": {
            name: heads[name].config
            for name in ("point", "linear_beta", "mlp_beta", "laplace_beta")
        },
        "global_concentration": heads["global_spread"].c,
        "table": rows,
        "nll_diff_linear_minus_global": vs_global,
        "nll_diff_linear_minus_difficulty": vs_difficulty,
        "eb_r4_diff_linear_minus_global": eb_ci,
        "nll_diff_laplace_minus_global": laplace_vs_global,
        **vs_group,
        "per_group_test_nll": per_group,
        "dispersion_by_group (all splits)": dispersion_by_group(all_data),
        "group_concentrations": getattr(heads.get("group_spread"), "c", None),
        "gate": gate,
        "gate_passed": all(gate.values()),
    }


def main(argv):
    run_dir = Path(argv[1])

    def opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    layer, prefix, scoring = (
        opt("--layer", "last"),
        opt("--splits", "bigmath"),
        opt("--scoring", "nq"),
    )
    report = run(run_dir, layer, splits_for(prefix), scoring)
    print(f"splits = {prefix}, scoring = {scoring}")
    print(
        f"train/val/test = {report['n_train']}/{report['n_val']}/{report['n_test']}, "
        f"layer = {layer}\n"
    )
    print(pd.DataFrame(report["table"]).round(4).to_string(index=False))
    for key in (
        "nll_diff_linear_minus_global",
        "nll_diff_linear_minus_difficulty",
        "eb_r4_diff_linear_minus_global",
        "nll_diff_laplace_minus_global",
        *(k for k in report if k.endswith("_minus_group")),
    ):
        est, lo, hi = report[key]
        print(f"{key}: {est:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("\nper-group test NLL:")
    print(pd.DataFrame(report["per_group_test_nll"]).round(4).to_string())
    print("\nsolve-rate dispersion by group (no features):")
    print(pd.DataFrame(report["dispersion_by_group (all splits)"]).round(3).to_string())
    if report["group_concentrations"]:
        print("group concentrations (fitted on val):", report["group_concentrations"])
    print("\ngate:", json.dumps(report["gate"], indent=1), "\npassed:", report["gate_passed"])
    (run_dir / f"stage0_report_{prefix}_{scoring}_{layer}.json").write_text(
        json.dumps(report, indent=1, default=float)
    )


if __name__ == "__main__":
    main(sys.argv)
