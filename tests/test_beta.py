import numpy as np
import pytest

from soc.forecast import DEFAULT_CLAMPS, ab_to_ms, is_clamped, ms_to_ab


@pytest.mark.parametrize("m", [0.05, 0.3, 0.5, 0.62, 0.97])
@pytest.mark.parametrize("frac", [0.05, 0.3, 0.8])
def test_round_trip_inside_valid_region(m, frac):
    s = frac * np.sqrt(m * (1 - m))
    a, b = ms_to_ab(m, s)
    m2, s2 = ab_to_ms(a, b)
    assert np.isclose(m2, m) and np.isclose(s2, s)
    assert not is_clamped(m, s)


def test_matches_closed_form():
    a, b = ms_to_ab(0.62, 0.14)
    c = 0.62 * 0.38 / 0.14**2 - 1
    assert np.isclose(a, 0.62 * c) and np.isclose(b, 0.38 * c)


def test_ab_round_trip():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0.2, 50, 200), rng.uniform(0.2, 50, 200)
    a2, b2 = ms_to_ab(*ab_to_ms(a, b))
    assert np.allclose(a, a2) and np.allclose(b, b2)


def test_zero_spread_clamps_to_max_concentration():
    a, b = ms_to_ab(0.4, 0.0)
    assert np.isclose(a + b, DEFAULT_CLAMPS.c_max)
    assert is_clamped(0.4, 0.0)


def test_too_wide_spread_clamps_to_min_concentration():
    for s in [np.sqrt(0.25), 0.5]:
        a, b = ms_to_ab(0.5, s)
        assert np.isclose(a + b, DEFAULT_CLAMPS.c_min)
        assert a > 0 and b > 0


@pytest.mark.parametrize("m", [0.0, 1.0])
def test_extreme_means_are_clamped_and_finite(m):
    a, b = ms_to_ab(m, 0.0)
    assert np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0
    assert DEFAULT_CLAMPS.m_eps <= a / (a + b) <= 1 - DEFAULT_CLAMPS.m_eps


def test_vectorised():
    a, b = ms_to_ab(np.array([0.1, 0.5, 0.9]), np.array([0.05, 0.1, 0.0]))
    assert a.shape == (3,) and np.all(a > 0) and np.all(b > 0)


@pytest.mark.parametrize("m,s", [(np.nan, 0.1), (0.5, np.nan), (0.5, -0.1)])
def test_invalid_inputs_raise(m, s):
    with pytest.raises(ValueError):
        ms_to_ab(m, s)


def test_ab_to_ms_rejects_nonpositive():
    with pytest.raises(ValueError):
        ab_to_ms(0.0, 1.0)
