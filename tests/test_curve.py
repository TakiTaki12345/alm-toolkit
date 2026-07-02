import numpy as np
import pytest

from alm.core.curve import YieldCurve


def test_zero_rate_at_sample_points():
    curve = YieldCurve(tenors=[1, 2, 5], rates=[0.02, 0.025, 0.03])
    assert curve.zero_rate(1) == pytest.approx(0.02, abs=1e-9)
    assert curve.zero_rate(2) == pytest.approx(0.025, abs=1e-9)
    assert curve.zero_rate(5) == pytest.approx(0.03, abs=1e-9)


def test_zero_rate_linear_interpolation():
    curve = YieldCurve(tenors=[1, 3], rates=[0.02, 0.04])
    # Midpoint t=2 should give the average rate 0.03.
    assert curve.zero_rate(2) == pytest.approx(0.03, abs=1e-9)


def test_zero_rate_flat_extrapolation():
    curve = YieldCurve(tenors=[1, 5], rates=[0.02, 0.03])
    assert curve.zero_rate(0.5) == pytest.approx(0.02, abs=1e-9)  # below range
    assert curve.zero_rate(10) == pytest.approx(0.03, abs=1e-9)   # above range


def test_discount_factor_at_zero_is_one():
    #DF(0) = 1.
    curve = YieldCurve(tenors=[1, 5], rates=[0.02, 0.03])
    assert curve.discount_factor(0) == pytest.approx(1.0, abs=1e-9)


def test_discount_factor_formula():
    #DF(t) = exp(-r * t) under continuous compounding.
    curve = YieldCurve(tenors=[1, 5], rates=[0.03, 0.03])
    # DF(2) = exp(-0.03 * 2).
    assert curve.discount_factor(2) == pytest.approx(np.exp(-0.03 * 2), abs=1e-9)


def test_discount_factor_decreases_with_time():
    #With positive rates, longer horizons have smaller discount factors.
    curve = YieldCurve(tenors=[1, 10], rates=[0.03, 0.03])
    assert curve.discount_factor(1) > curve.discount_factor(5)


def test_vectorized_input():
    #The curve should accept array input and return array output.
    curve = YieldCurve(tenors=[1, 5], rates=[0.02, 0.03])
    dfs = curve.discount_factor([1, 2, 3])
    assert dfs.shape == (3,)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        YieldCurve(tenors=[1, 2, 3], rates=[0.02, 0.03])


def test_rejects_non_increasing_tenors():
    with pytest.raises(ValueError):
        YieldCurve(tenors=[1, 1, 2], rates=[0.02, 0.03, 0.04])


def test_rejects_too_few_points():
    with pytest.raises(ValueError):
        YieldCurve(tenors=[1], rates=[0.02])

def test_forward_rate_equals_spot_on_flat_curve():
    """On a flat curve, every forward rate equals the (constant) spot rate.

    If all spot rates are identical, no future period can imply a different
    rate without creating arbitrage, so f(t1, t2) must equal that flat rate.
    """
    curve = YieldCurve(tenors=[1, 10], rates=[0.03, 0.03])
    assert curve.forward_rate(2, 5) == pytest.approx(0.03, rel=1e-12)
    assert curve.forward_rate(1, 10) == pytest.approx(0.03, rel=1e-12)


def test_forward_rate_discount_consistency():
    """Discounting over [t1, t2] at the forward rate must match the ratio of
    spot discount factors:  exp(-f * (t2 - t1)) == DF(t2) / DF(t1).

    This is the no-arbitrage identity the forward rate is derived from, so it
    should hold for any (t1, t2) on any curve.
    """
    curve = YieldCurve(tenors=[1, 2, 5, 10], rates=[0.02, 0.025, 0.03, 0.035])
    t1, t2 = 2, 7
    f = curve.forward_rate(t1, t2)
    forward_discount = np.exp(-f * (t2 - t1))
    ratio = curve.discount_factor(t2) / curve.discount_factor(t1)
    assert forward_discount == pytest.approx(ratio, rel=1e-12)


def test_forward_rate_above_spot_when_curve_rises():
    """On an upward-sloping curve, the forward rate for a period beyond t1
    exceeds the spot rate to t1 (marginal rates lie above the average).
    """
    curve = YieldCurve(tenors=[1, 5], rates=[0.02, 0.04])
    # Spot to 5y is between 0.02 and 0.04; the forward from 4y to 5y should
    # be higher than the average spot rate up to 4y.
    f = curve.forward_rate(4, 5)
    assert f > curve.zero_rate(4)


def test_forward_rate_rejects_non_increasing_interval():
    """t2 must be strictly greater than t1."""
    curve = YieldCurve(tenors=[1, 5], rates=[0.02, 0.04])
    with pytest.raises(ValueError):
        curve.forward_rate(5, 5)
    with pytest.raises(ValueError):
        curve.forward_rate(5, 3)