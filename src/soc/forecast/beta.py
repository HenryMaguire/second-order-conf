"""Conversion between a stated forecast (m, s) and Beta parameters (alpha, beta).

The forecast is a belief about p, the model's probability of solving a question: m is its mean
and s its standard deviation. For Beta(alpha, beta) with concentration c = alpha + beta,

    mean = alpha / c,    var = m (1 - m) / (c + 1)   =>   c = m (1 - m) / s^2 - 1.

A Beta exists only when 0 < m < 1 and 0 < s^2 < m (1 - m), so stated values are clamped:
m into [m_eps, 1 - m_eps], then c into [c_min, c_max]. s = 0 maps to c_max (close to a plain
Binomial with p = m); s at or above sqrt(m (1 - m)) maps to c_min (strongly U-shaped).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

M_EPS = 1e-3
C_MIN = 0.1
C_MAX = 1e4


@dataclass(frozen=True)
class Clamps:
    m_eps: float = M_EPS
    c_min: float = C_MIN
    c_max: float = C_MAX


DEFAULT_CLAMPS = Clamps()


def ms_to_ab(m, s, clamps: Clamps = DEFAULT_CLAMPS):
    """Map (m, s) to (alpha, beta), clamping into the valid Beta region. Broadcasts."""
    m = np.clip(np.asarray(m, dtype=float), clamps.m_eps, 1.0 - clamps.m_eps)
    s = np.asarray(s, dtype=float)
    if np.any(np.isnan(m)) or np.any(np.isnan(s)) or np.any(s < 0):
        raise ValueError("m and s must be non-NaN and s must be non-negative")
    var = m * (1.0 - m)
    with np.errstate(divide="ignore"):
        c = np.where(s > 0, var / np.square(s) - 1.0, np.inf)
    c = np.clip(c, clamps.c_min, clamps.c_max)
    return m * c, (1.0 - m) * c


def ab_to_ms(alpha, beta):
    """Map (alpha, beta) to the Beta's mean and standard deviation. Broadcasts."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("alpha and beta must be positive")
    c = alpha + beta
    m = alpha / c
    return m, np.sqrt(m * (1.0 - m) / (c + 1.0))


def is_clamped(m, s, clamps: Clamps = DEFAULT_CLAMPS):
    """True where (m, s) falls outside the region ms_to_ab represents exactly (for logging)."""
    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    alpha, beta = ms_to_ab(m, s, clamps)
    m2, s2 = ab_to_ms(alpha, beta)
    return ~(np.isclose(m, m2, atol=1e-9) & np.isclose(s, s2, atol=1e-9))
