"""Evaluation metrics: PIT, interval coverage, ECE, Brier, AUROC, bootstrap CIs."""

from soc.eval.calibration import (
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

__all__ = [
    "auroc",
    "bootstrap_ci",
    "brier",
    "central_interval",
    "ece",
    "interval_coverage",
    "pit_coverage",
    "pit_histogram",
    "randomized_pit",
]
