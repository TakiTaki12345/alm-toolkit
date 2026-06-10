import numpy as np
import pytest

from alm.core.cashflow import CashFlow
from alm.core.curve import YieldCurve


def test_present_value_single_flow():
    # A single future payment discounts to amount * DF(t).
    curve = YieldCurve(tenors=[1, 5], rates=[0.03, 0.03])  # flat 3%
    cf = CashFlow(times=[2], amounts=[100])
    expected = 100 * np.exp(-0.03 * 2)
    assert cf.present_value(curve) == pytest.approx(expected, abs=1e-9)


def test_present_value_at_time_zero():
    # A payment at t=0 is worth its face value (DF(0) = 1).
    curve = YieldCurve(tenors=[1, 5], rates=[0.03, 0.03])
    cf = CashFlow(times=[0], amounts=[100])
    assert cf.present_value(curve) == pytest.approx(100.0, abs=1e-9)


def test_present_value_sums_multiple_flows():
    # PV of a stream is the sum of the discounted individual flows.
    curve = YieldCurve(tenors=[1, 5], rates=[0.03, 0.03])
    cf = CashFlow(times=[1, 2], amounts=[100, 100])
    expected = 100 * np.exp(-0.03 * 1) + 100 * np.exp(-0.03 * 2)
    assert cf.present_value(curve) == pytest.approx(expected, abs=1e-9)


def test_present_value_zero_rate_curve():
    # With a 0% curve, PV equals the undiscounted sum of amounts.
    curve = YieldCurve(tenors=[1, 5], rates=[0.0, 0.0])
    cf = CashFlow(times=[1, 2, 3], amounts=[50, 50, 1050])
    assert cf.present_value(curve) == pytest.approx(1150.0, abs=1e-9)


def test_flows_are_sorted_by_time():
    # Construction sorts cash flows chronologically.
    cf = CashFlow(times=[3, 1, 2], amounts=[1050, 50, 50])
    assert list(cf.times) == [1.0, 2.0, 3.0]
    assert list(cf.amounts) == [50.0, 50.0, 1050.0]


def test_present_value_invariant_to_input_order():
    # PV must not depend on the order flows were provided in.
    curve = YieldCurve(tenors=[1, 5], rates=[0.03, 0.03])
    cf1 = CashFlow(times=[1, 2, 3], amounts=[50, 50, 1050])
    cf2 = CashFlow(times=[3, 2, 1], amounts=[1050, 50, 50])
    assert cf1.present_value(curve) == pytest.approx(cf2.present_value(curve), abs=1e-9)


def test_negative_amounts_allowed():
    # Outflows (negative amounts) are valid, e.g. liabilities.
    curve = YieldCurve(tenors=[1, 5], rates=[0.0, 0.0])
    cf = CashFlow(times=[1, 2], amounts=[-100, -100])
    assert cf.present_value(curve) == pytest.approx(-200.0, abs=1e-9)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        CashFlow(times=[1, 2, 3], amounts=[100, 100])


def test_rejects_negative_times():
    with pytest.raises(ValueError):
        CashFlow(times=[-1, 2], amounts=[100, 100])


def test_rejects_empty():
    with pytest.raises(ValueError):
        CashFlow(times=[], amounts=[])