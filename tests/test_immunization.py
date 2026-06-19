"""Tests for two-bond liability-driven immunization."""

import numpy as np
import pytest

from alm.core.cashflow import CashFlow
from alm.core.risk import present_value_at_yield, macaulay_duration
from alm.applications.immunization import Bond, immunize_two_bonds


def make_zero(t, face, y):
    """Helper: a zero-coupon bond paying `face` at time t, wrapped as Bond."""
    return Bond(cashflow=CashFlow(times=[t], amounts=[face]), y=y)


def test_weights_sum_to_liability_pv():
    """PV match: the two weights must sum to the liability's PV."""
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    bond_a = make_zero(3, 100, y)   # shorter duration
    bond_b = make_zero(8, 100, y)   # longer duration
    result = immunize_two_bonds(liability, bond_a, bond_b, y)

    pv_l = present_value_at_yield(liability, y)
    assert result.weight_a + result.weight_b == pytest.approx(pv_l, rel=1e-12)


def test_portfolio_duration_matches_liability():
    """Duration match: PV-weighted asset duration equals liability duration."""
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    bond_a = make_zero(3, 100, y)
    bond_b = make_zero(8, 100, y)
    result = immunize_two_bonds(liability, bond_a, bond_b, y)

    pv_l = present_value_at_yield(liability, y)
    asset_duration = (
        result.weight_a * bond_a.duration + result.weight_b * bond_b.duration
    ) / pv_l
    assert asset_duration == pytest.approx(macaulay_duration(liability, y), rel=1e-12)


def test_closed_form_weights_by_hand():
    """With zero-coupon bonds, durations are exactly 3 and 8 years.

    Liability duration is 5 (a 5-year zero). The closed form gives:
        w_a = PV_L * (5 - 8) / (3 - 8) = PV_L * 3/5
        w_b = PV_L * (3 - 5) / (3 - 8) = PV_L * 2/5
    """
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    bond_a = make_zero(3, 100, y)
    bond_b = make_zero(8, 100, y)
    result = immunize_two_bonds(liability, bond_a, bond_b, y)

    pv_l = present_value_at_yield(liability, y)
    assert result.weight_a == pytest.approx(pv_l * 3 / 5, rel=1e-12)
    assert result.weight_b == pytest.approx(pv_l * 2 / 5, rel=1e-12)


def test_feasible_when_liability_duration_between_bonds():
    """Both weights non-negative when D_L lies between the two bond durations."""
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])  # D_L = 5, between 3 and 8
    result = immunize_two_bonds(liability, make_zero(3, 100, y), make_zero(8, 100, y), y)
    assert result.feasible
    assert result.weight_a >= 0 and result.weight_b >= 0


def test_infeasible_when_liability_duration_outside_bonds():
    """A weight goes negative when D_L is outside the bonds' duration range."""
    y = 0.04
    # Liability duration 10 is longer than both bonds (3 and 8): needs shorting.
    liability = CashFlow(times=[10], amounts=[1000])
    result = immunize_two_bonds(liability, make_zero(3, 100, y), make_zero(8, 100, y), y)
    assert not result.feasible


def test_rejects_equal_duration_bonds():
    """Two bonds with identical duration cannot span a liability."""
    y = 0.04
    liability = CashFlow(times=[5], amounts=[1000])
    bond_a = make_zero(4, 100, y)
    bond_b = make_zero(4, 200, y)  # same duration (4), different size
    with pytest.raises(ValueError):
        immunize_two_bonds(liability, bond_a, bond_b, y)


def test_immunization_neutralizes_small_rate_shock():
    """The heart of immunization: under a small parallel rate shock, the
    change in asset value should approximately offset the change in
    liability value (first-order neutrality).

    We build the immunizing portfolio at yield y, then revalue both the
    assets and the liability at y +/- shock and confirm the net change in
    (assets - liability) is second-order small.

    它不驗證任何公式，而是從第一性原理驗證免疫的目的本身：
    建好免疫組合後，真的去動利率（上下各 1 個基點），確認「資產減負債」這個盈餘幾乎不動。
    這是對整個免疫邏輯的端到端驗證，如果 duration 匹配的數學錯了，這個測試會立刻抓到。
    最後兩行還順帶驗證了 Redington 理論的一個推論：免疫後的盈餘在利率小幅變動下不會變負（凸度保護）。
    """

    y = 0.04
    shock = 0.0001  # 1 basis point

    liability = CashFlow(times=[5], amounts=[1000])
    bond_a = make_zero(3, 100, y)
    bond_b = make_zero(8, 100, y)
    result = immunize_two_bonds(liability, bond_a, bond_b, y)

    # Number of units of each bond, from PV-weight / per-unit PV.
    units_a = result.weight_a / bond_a.pv
    units_b = result.weight_b / bond_b.pv

    def surplus(rate):
        assets = (
            units_a * present_value_at_yield(bond_a.cashflow, rate)
            + units_b * present_value_at_yield(bond_b.cashflow, rate)
        )
        liab = present_value_at_yield(liability, rate)
        return assets - liab

    s0 = surplus(y)
    s_up = surplus(y + shock)
    s_down = surplus(y - shock)

    # Surplus starts near zero (PV matched) and stays near zero under shocks.
    assert s0 == pytest.approx(0.0, abs=1e-9)
    # First-order change is neutralized: surplus barely moves either way.
    assert abs(s_up) < 1e-4
    assert abs(s_down) < 1e-4
    # And by Redington, the immunized surplus should be >= 0 around y
    # (convexity of assets vs liability). Check it doesn't go negative.
    assert s_up >= -1e-9
    assert s_down >= -1e-9