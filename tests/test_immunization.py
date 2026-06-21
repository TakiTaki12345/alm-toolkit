"""Tests for N-bond liability-driven immunization."""

import numpy as np
import pytest

from alm.core.cashflow import CashFlow
from alm.core.risk import present_value_at_yield
from alm.applications.immunization import Bond, immunize


def make_zero(t, face, y):
    """Helper: a zero-coupon bond paying `face` at time t, wrapped as Bond."""
    return Bond(cashflow=CashFlow(times=[t], amounts=[face]), y=y)


def test_two_bond_case_matches_hand_computed_weights():
    """Cross-validate the LP solver against the closed-form two-bond solution.

    Liability: a 5-year zero (duration 5). Bonds: 3-year and 8-year zeros
    (durations 3 and 8). The analytic immunizing weights are:
        w_a = PV_L * (D_L - D_b) / (D_a - D_b) = PV_L * (5-8)/(3-8) = PV_L*3/5
        w_b = PV_L * (D_a - D_L) / (D_a - D_b) = PV_L * (3-5)/(3-8) = PV_L*2/5
    These are computed by hand and hard-coded as the expected values.
    """
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    bond_a = make_zero(3, 100, y)
    bond_b = make_zero(8, 100, y)

    result = immunize(liability, [bond_a, bond_b], y)
    pv_l = present_value_at_yield(liability, y)

    assert result.success
    assert result.weights[0] == pytest.approx(pv_l * 3 / 5, rel=1e-9)
    assert result.weights[1] == pytest.approx(pv_l * 2 / 5, rel=1e-9)


def test_weights_sum_to_liability_pv():
    """PV match: weights must sum to the liability PV."""
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    bonds = [make_zero(2, 100, y), make_zero(5, 100, y), make_zero(9, 100, y)]

    result = immunize(liability, bonds, y)
    pv_l = present_value_at_yield(liability, y)

    assert result.success
    assert np.sum(result.weights) == pytest.approx(pv_l, rel=1e-9)


def test_portfolio_duration_matches_liability():
    """Duration match: PV-weighted asset duration equals liability duration."""
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    bonds = [make_zero(2, 100, y), make_zero(5, 100, y), make_zero(9, 100, y)]
    durations = np.array([b.duration for b in bonds])

    result = immunize(liability, bonds, y)
    pv_l = present_value_at_yield(liability, y)
    asset_duration = np.dot(result.weights, durations) / pv_l

    assert result.success
    assert asset_duration == pytest.approx(5.0, rel=1e-9)


def test_maximizes_convexity():
    """The LP should select the portfolio with the highest feasible convexity.

    With three bonds spanning the liability, the optimizer should achieve
    convexity at least as high as any hand-built feasible mix. Here we just
    confirm the Redington convexity condition is satisfied.
    """
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    bonds = [make_zero(2, 100, y), make_zero(5, 100, y), make_zero(9, 100, y)]

    result = immunize(liability, bonds, y)

    assert result.success
    assert result.convexity_satisfied


def test_immunization_neutralizes_small_rate_shock():
    """End-to-end: under a small parallel shock, the surplus (assets minus
    liability) stays near zero — the defining property of immunization."""
    y = 0.04
    shock = 0.0001
    liability = CashFlow(times=[5], amounts=[1000])
    bonds = [make_zero(2, 100, y), make_zero(5, 100, y), make_zero(9, 100, y)]

    result = immunize(liability, bonds, y)
    assert result.success

    units = [w / b.pv for w, b in zip(result.weights, bonds)]

    def surplus(rate):
        assets = sum(
            u * present_value_at_yield(b.cashflow, rate)
            for u, b in zip(units, bonds)
        )
        return assets - present_value_at_yield(liability, rate)

    assert surplus(y) == pytest.approx(0.0, abs=1e-9)
    assert abs(surplus(y + shock)) < 1e-4
    assert abs(surplus(y - shock)) < 1e-4


def test_full_immunization_survives_large_shock():
    """Full immunization: a single liability bracketed by one earlier and one
    later zero-coupon bond is protected against *large* parallel shifts, not
    just infinitesimal ones. Surplus should stay >= 0 even for a 200bp move.
    """
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    # One bond maturing before (t=3), one after (t=8) the liability.
    bonds = [make_zero(3, 100, y), make_zero(8, 100, y)]

    result = immunize(liability, bonds, y)
    assert result.success

    units = [w / b.pv for w, b in zip(result.weights, bonds)]

    def surplus(rate):
        assets = sum(
            u * present_value_at_yield(b.cashflow, rate)
            for u, b in zip(units, bonds)
        )
        return assets - present_value_at_yield(liability, rate)

    # Large shocks in both directions: surplus stays non-negative.
    for shock in [-0.02, -0.01, 0.01, 0.02]:
        assert surplus(y + shock) >= -1e-9


def test_infeasible_without_short_selling():
    """When the liability duration lies outside all bond durations and short
    selling is disallowed, no feasible non-negative portfolio exists."""
    y = 0.04
    # Liability duration 10, longer than both bonds (3 and 8).
    liability = CashFlow(times=[10], amounts=[1000])
    bonds = [make_zero(3, 100, y), make_zero(8, 100, y)]

    result = immunize(liability, bonds, y, allow_short=False)
    assert not result.success


def test_feasible_with_short_selling():
    """Allowing short positions can make an otherwise-infeasible problem
    solvable: the duration constraint can be met with a negative weight."""
    y = 0.04
    liability = CashFlow(times=[10], amounts=[1000])
    bonds = [make_zero(3, 100, y), make_zero(8, 100, y)]

    result = immunize(liability, bonds, y, allow_short=True)
    assert result.success
    pv_l = present_value_at_yield(liability, y)
    assert np.sum(result.weights) == pytest.approx(pv_l, rel=1e-9)


def test_requires_at_least_two_bonds():
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    with pytest.raises(ValueError):
        immunize(liability, [make_zero(5, 100, y)], y)