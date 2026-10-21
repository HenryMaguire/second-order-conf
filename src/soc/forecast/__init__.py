"""Forecast scoring: (m, s) <-> Beta, Beta-Binomial NLL and RPS, and the tag parsers."""

from soc.forecast.beta import DEFAULT_CLAMPS, Clamps, ab_to_ms, is_clamped, ms_to_ab
from soc.forecast.parse import (
    Forecast,
    format_forecast,
    parse_confidence,
    parse_forecast,
    parse_forecast_body,
)
from soc.forecast.scores import (
    betabinom_cdf_grid,
    betabinom_logpmf,
    betabinom_nll,
    betabinom_pmf_grid,
    leave_one_out_counts,
    rps,
    rps_reward,
)

__all__ = [
    "DEFAULT_CLAMPS",
    "Clamps",
    "Forecast",
    "ab_to_ms",
    "betabinom_cdf_grid",
    "betabinom_logpmf",
    "betabinom_nll",
    "betabinom_pmf_grid",
    "format_forecast",
    "is_clamped",
    "leave_one_out_counts",
    "ms_to_ab",
    "parse_confidence",
    "parse_forecast",
    "parse_forecast_body",
    "rps",
    "rps_reward",
]
