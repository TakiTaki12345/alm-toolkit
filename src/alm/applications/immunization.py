"""Liability-driven immunization.

Given a liability and a set of candidate bonds, find an allocation whose
value moves in step with the liability as interest rates change, so the
obligation stays funded under rate movements.

Classical Redington immunization requires three conditions:
  1. PV match:        asset PV == liability PV
  2. Duration match:  asset duration == liability duration
  3. Convexity:       asset convexity >= liability convexity

Conditions 1 and 2 are linear equality constraints. With N bonds that
leaves N-2 degrees of freedom, which we use to *maximize* portfolio
convexity (strengthening Redington's second-order protection). Since the
objective and constraints are all linear in the PV weights, this is a
linear program, solved with scipy.optimize.linprog.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from alm.core.cashflow import CashFlow
from alm.core.risk import (
    present_value_at_yield,
    macaulay_duration,
    convexity,
)


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
class PortfolioResult:
    """The outcome of an N-bond immunization."""

    weights: np.ndarray        # PV amount allocated to each bond
    asset_convexity: float     # PV-weighted convexity of the portfolio
    liability_convexity: float
    convexity_satisfied: bool  # asset convexity >= liability convexity?
    success: bool              # did the optimizer find a feasible solution?
    message: str               # solver status message


def immunize(
    liability: CashFlow,
    bonds: list[Bond],
    y: float,
    allow_short: bool = False,
) -> PortfolioResult:
    """Immunize a liability with N bonds, maximizing portfolio convexity.

    Solves the linear program:

        maximize    sum_i w_i * C_i               (portfolio convexity * PV_L)
        subject to  sum_i w_i        = PV_L       (PV match)
                    sum_i w_i * D_i  = PV_L * D_L  (duration match)
                    w_i >= 0                       (if allow_short is False)

    where w_i is the PV amount allocated to bond i, and C_i, D_i are that
    bond's convexity and duration.

    Parameters
    ----------
    liability : CashFlow
        The obligation to immunize.
    bonds : list[Bond]
        Candidate bonds (at least two, with distinct durations).
    y : float
        The flat yield at which all measures are computed.
    allow_short : bool, default False
        If False, weights are constrained non-negative (no short selling).
        If True, weights may be negative.

    Returns
    -------
    PortfolioResult
    """
    if len(bonds) < 2:
        raise ValueError("immunization requires at least two bonds")

    pv_l = present_value_at_yield(liability, y)
    d_l = macaulay_duration(liability, y)
    c_l = convexity(liability, y)

    durations = np.array([b.duration for b in bonds])
    convexities = np.array([b.convexity for b in bonds])

    # Objective: maximize sum_i w_i * C_i  ==  minimize  -sum_i w_i * C_i.
    # (linprog minimizes c @ w, so we negate the convexity vector.)
    c = -convexities

    # Equality constraints: PV match and duration match.
    A_eq = np.vstack([
        np.ones(len(bonds)),  # sum w_i        = PV_L
        durations,            # sum w_i * D_i  = PV_L * D_L
    ])
    b_eq = np.array([pv_l, pv_l * d_l])

    # Bounds: lower bound 0 (no short) or -inf (short allowed).
    lower = None if allow_short else 0.0
    bounds = [(lower, None)] * len(bonds)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        return PortfolioResult(
            weights=np.full(len(bonds), np.nan),
            asset_convexity=np.nan,
            liability_convexity=c_l,
            convexity_satisfied=False,
            success=False,
            message=result.message,
        )

    weights = np.asarray(result.x, dtype=float)
    asset_convexity = float(np.dot(weights, convexities) / pv_l)

    return PortfolioResult(
        weights=weights,
        asset_convexity=asset_convexity,
        liability_convexity=c_l,
        convexity_satisfied=asset_convexity >= c_l - 1e-9,
        success=True,
        message=result.message,
    )