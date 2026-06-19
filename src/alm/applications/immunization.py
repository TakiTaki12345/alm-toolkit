# Liability-driven immunization.

"""
Classical Redington immunization requires three conditions:
  1. PV match:        asset PV == liability PV
  2. Duration match:  asset duration == liability duration
  3. Convexity:       asset convexity >= liability convexity

This module starts with the exact two-bond case (a 2x2 linear system,
verifiable by hand) and reports whether the convexity condition holds.
"""

from __future__ import annotations

from dataclasses import dataclass

from alm.core.cashflow import CashFlow
from alm.core.risk import (
    present_value_at_yield,
    macaulay_duration,
    convexity,
)


"""
我用了 @dataclass——這是 Python 表達「純資料容器」的慣用法，自動生成 __init__、__repr__ 等，少寫樣板碼、意圖清晰。
Bond 用 @property 讓 pv、duration、convexity 像屬性一樣存取（bond.duration 而非 bond.duration()），語意上它們是債券的性質，property 最貼切。
"""
@dataclass
class Bond:
    """A bond viewed through its cash flows at a single flat yield.

    Wraps a CashFlow and a yield, exposing the three risk quantities the
    immunization logic needs, computed via the tested core risk functions.
    """

    cashflow: CashFlow
    y: float

    @property
    def pv(self) -> float:
        return present_value_at_yield(self.cashflow, self.y)

    @property
    def duration(self) -> float:
        return macaulay_duration(self.cashflow, self.y)

    @property
    def convexity(self) -> float:
        return convexity(self.cashflow, self.y)


@dataclass
class ImmunizationResult:
    """The outcome of a two-bond immunization."""

    weight_a: float          # amount (in PV terms) allocated to bond A
    weight_b: float          # amount (in PV terms) allocated to bond B
    asset_convexity: float   # PV-weighted convexity of the asset portfolio
    liability_convexity: float
    convexity_satisfied: bool  # asset convexity >= liability convexity?
    feasible: bool             # are both weights non-negative?


def immunize_two_bonds(
    liability: CashFlow,
    bond_a: Bond,
    bond_b: Bond,
    y: float,
) -> ImmunizationResult:
    """Solve the two-bond immunization problem.

    Finds w_a, w_b (PV amounts in each bond) such that:
        w_a + w_b                 = PV_L          (PV match)
        w_a*D_a + w_b*D_b         = PV_L * D_L    (duration match)

    which has the closed-form solution
        w_a = PV_L * (D_L - D_b) / (D_a - D_b)
        w_b = PV_L * (D_a - D_L) / (D_a - D_b)

    The convexity condition is then *checked* (not solved), since two
    equations in two unknowns leave no remaining degrees of freedom.
    """
    pv_l = present_value_at_yield(liability, y)
    d_l = macaulay_duration(liability, y)
    c_l = convexity(liability, y)

    d_a = bond_a.duration
    d_b = bond_b.duration

    if d_a == d_b:
        raise ValueError(
            "bonds must have distinct durations to span the liability"
        )

    w_a = pv_l * (d_l - d_b) / (d_a - d_b)
    w_b = pv_l * (d_a - d_l) / (d_a - d_b)

    # Asset portfolio convexity is the PV-weighted average of the two
    # bonds' convexities (weights are PV shares).
    asset_convexity = (w_a * bond_a.convexity + w_b * bond_b.convexity) / pv_l

    return ImmunizationResult(
        weight_a=w_a,
        weight_b=w_b,
        asset_convexity=asset_convexity,
        liability_convexity=c_l,
        convexity_satisfied=asset_convexity >= c_l,
        feasible=(w_a >= 0 and w_b >= 0),
    )