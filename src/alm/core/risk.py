"""Interest rate risk measures: duration and convexity.

These quantify how the present value of a cash flow stream responds to
changes in interest rates, and are the foundation of immunization.

Design note — why standalone functions here, rather than methods on
CashFlow (as present_value is)? This is a deliberate separation of
concerns. A stream's present value is its most intrinsic property, so it
belongs on the object. Duration and convexity, by contrast, are a
risk-analysis view and depend on an external parameter y — the yield level
at which sensitivity is measured, which is the analyst's choice, not
something the cash flow carries. Keeping them in a separate risk module
splits "data representation" (cashflow) from "risk analysis" (risk). A
future effective_duration will live here too, growing this into a complete
risk toolkit. Both designs are defensible; this split extends more cleanly.

All measures here are analytic sensitivities under a single flat yield y
(continuous compounding), matching the textbook definition and easy to
verify exactly.
"""

from __future__ import annotations

import numpy as np

from alm.core.cashflow import CashFlow


def present_value_at_yield(cashflow: CashFlow, y: float) -> float:
    """Present value discounted at a single flat yield y.

    PV(y) = sum_i CF_i * exp(-y * t_i)

    Distinct from CashFlow.present_value(curve), which uses a full term
    structure; here a single yield gives duration and convexity a
    well-defined point of differentiation.
    """
    t = cashflow.times
    cf = cashflow.amounts
    return float(np.sum(cf * np.exp(-y * t)))


def macaulay_duration(cashflow: CashFlow, y: float) -> float:
    """Macaulay duration: the PV-weighted average time to cash flow.

    D_mac = sum_i t_i * (CF_i * exp(-y * t_i)) / PV

    Interpreted as the average horizon (in years) over which present value
    is recovered: a zero-coupon bond's equals its maturity; coupon-paying
    instruments have shorter duration.
    """
    t = cashflow.times
    cf = cashflow.amounts
    pv_weights = cf * np.exp(-y * t)
    pv = np.sum(pv_weights)
    if pv == 0:
        raise ValueError("Macaulay duration is undefined when PV is zero")
    return float(np.sum(t * pv_weights) / pv)


def modified_duration(cashflow: CashFlow, y: float) -> float:
    """Modified duration: relative sensitivity of PV to the yield.

    D_mod = -(1/PV) * dPV/dy

    Under continuous compounding, modified duration equals Macaulay
    duration exactly (the discrete-compounding factor 1/(1+y/m) tends to 1).
    Provided as a separate function for semantic clarity: a caller asking
    for a risk sensitivity should not need to know it coincides with D_mac.
    """
    return macaulay_duration(cashflow, y)


def convexity(cashflow: CashFlow, y: float) -> float:
    """Convexity: the PV-weighted average of t^2, the second-order measure.

    C = (1/PV) * d2PV/dy2 = sum_i t_i^2 * (CF_i * exp(-y * t_i)) / PV

    Mirrors duration but weights by t^2 instead of t, capturing the
    curvature of the PV-yield relationship that duration alone misses.
    """
    t = cashflow.times
    cf = cashflow.amounts
    pv_weights = cf * np.exp(-y * t)
    pv = np.sum(pv_weights)
    if pv == 0:
        raise ValueError("Convexity is undefined when PV is zero")
    return float(np.sum(t**2 * pv_weights) / pv)