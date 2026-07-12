"""Tests for life insurance and annuity pricing (actuarial present value)."""

import numpy as np
import pytest

from alm.core.curve import YieldCurve
from alm.core.mortality import LifeTable
from alm.applications.life_insurance import (
    term_insurance,
    whole_life_insurance,
    life_annuity_due,
)


def make_table(start_age=90):
    """A short table near the end of life, so whole-life sums are small and
    hand-checkable. Ages 90..94, with the last year certain (q=1)."""
    q = [0.10, 0.15, 0.25, 0.40, 1.00]
    return LifeTable(q, start_age=start_age)


def flat_curve(rate):
    """A flat continuously-compounded curve at the given rate."""
    return YieldCurve(tenors=[1, 100], rates=[rate, rate])


def test_term_insurance_hand_computed():
    """A 2-year term insurance APV, computed term by term by hand.

    Table from age 90: q_90 = 0.10, q_91 = 0.15. Flat 5% (continuous).
        v^1 = e^-0.05,  v^2 = e^-0.10
        year 1 death prob = 0p_90 * q_90 = 1 * 0.10
        year 2 death prob = 1p_90 * q_91 = 0.90 * 0.15 = 0.135
        APV = v^1 * 0.10 + v^2 * 0.135
    """
    table = make_table()
    curve = flat_curve(0.05)
    v1 = np.exp(-0.05)
    v2 = np.exp(-0.10)
    expected = v1 * 0.10 + v2 * 0.135
    assert term_insurance(table, curve, 90, 2) == pytest.approx(expected, rel=1e-12)


def test_annuity_due_hand_computed():
    """A 2-year temporary annuity-due APV, computed by hand.

        k=0: v^0 * 0p_90 = 1 * 1 = 1
        k=1: v^1 * 1p_90 = e^-0.05 * 0.90
        APV = 1 + e^-0.05 * 0.90
    """
    table = make_table()
    curve = flat_curve(0.05)
    expected = 1.0 + np.exp(-0.05) * 0.90
    assert life_annuity_due(table, curve, 90, n=2) == pytest.approx(expected, rel=1e-12)


def test_term_insurance_increases_with_term():
    """A longer term covers more years of death risk, so APV rises with n."""
    table = make_table()
    curve = flat_curve(0.05)
    apv1 = term_insurance(table, curve, 90, 1)
    apv2 = term_insurance(table, curve, 90, 2)
    apv3 = term_insurance(table, curve, 90, 3)
    assert apv1 < apv2 < apv3


def test_whole_life_equals_term_to_end_of_table():
    """Whole life is term insurance run to the end of the table."""
    table = make_table()
    curve = flat_curve(0.04)
    max_years = table.start_age + table.q.size - 90  # 5 years
    assert whole_life_insurance(table, curve, 90) == pytest.approx(
        term_insurance(table, curve, 90, max_years), rel=1e-12
    )


def test_whole_life_certain_when_rate_zero():
    """With a 0% curve and a table that ends in certain death (q=1 at the last
    age), whole life insurance of 1 is worth exactly 1: death is certain and
    there is no discounting, so the benefit is paid for sure at value 1."""
    table = make_table()
    curve = flat_curve(0.0)
    assert whole_life_insurance(table, curve, 90) == pytest.approx(1.0, rel=1e-12)


def test_insurance_annuity_identity():
    """The identity A_x = 1 - d * a-due_x links insurance and annuity.

    It holds under a flat curve, where a single discount rate d exists. With
    continuous compounding, the one-year discount factor is v = e^-r and
    d = 1 - v. Whole-life A_x and whole-life annuity-due a-due_x must satisfy
    the identity exactly — a cross-check that the two pricers agree.
    """
    table = make_table()
    r = 0.05
    curve = flat_curve(r)
    v = np.exp(-r)
    d = 1.0 - v

    A_x = whole_life_insurance(table, curve, 90)
    a_due_x = life_annuity_due(table, curve, 90)  # whole-life annuity-due

    assert A_x == pytest.approx(1.0 - d * a_due_x, rel=1e-9)


def test_higher_rate_lowers_insurance_apv():
    """Higher interest rates discount future benefits more, lowering APV."""
    table = make_table()
    low = whole_life_insurance(table, flat_curve(0.02), 90)
    high = whole_life_insurance(table, flat_curve(0.08), 90)
    assert high < low


def test_term_rejects_zero_years():
    table = make_table()
    curve = flat_curve(0.05)
    with pytest.raises(ValueError):
        term_insurance(table, curve, 90, 0)