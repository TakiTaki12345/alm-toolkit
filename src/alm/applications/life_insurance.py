"""Life insurance and annuity pricing via actuarial present value.

This module joins the two foundational primitives — a mortality table
(who is alive across time) and a yield curve (what money is worth across
time) — to price life-contingent cash flows.

The actuarial present value (APV) of any such product has one shape:

    APV = sum_k  discount(k) * probability(k) * benefit(k)

where probability(k) is the chance the k-th payment actually occurs. The
products differ only in that probability and in the timing convention:

    Insurance (term / whole life): pays on *death*. A death in year k
        (probability  {k-1}p_x * q_{x+k-1}) pays at the end of year k,
        discounted by v^k.

    Life annuity (annuity-due): pays on *survival*. A payment at time k
        (probability  {k}p_x) is made at the start of year k, discounted
        by v^k, for k = 0, 1, 2, ...

Insurance weights by the probability of dying; the annuity by the
probability of surviving — mirror images linked by the identity
A_x = 1 - d * a-due_x, which the tests verify.

Only integer-year (discrete) products are implemented here; fractional
timing is a natural later extension using the life table's fractional
survival.
"""

from __future__ import annotations

import numpy as np

from alm.core.curve import YieldCurve
from alm.core.mortality import LifeTable


def _apv(discounts, probabilities, benefits):
    """Core actuarial present value: sum of discount * probability * benefit.

    All three inputs are aligned arrays over the payment index. Keeping the
    summation in one place means the product functions only need to prepare
    their probability and benefit vectors.
    """
    discounts = np.asarray(discounts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    benefits = np.asarray(benefits, dtype=float)
    return float(np.sum(discounts * probabilities * benefits))


def term_insurance(
    table: LifeTable,
    curve: YieldCurve,
    age: int,
    n: int,
    benefit: float = 1.0,
) -> float:
    """APV of an n-year term insurance on a life aged `age`.

    Pays `benefit` at the end of the year of death, if death occurs within
    n years.

        APV = sum_{k=1}^{n}  v^k * {k-1}p_x * q_{x+k-1} * benefit
    """
    if n < 1:
        raise ValueError("term n must be at least 1")

    ks = np.arange(1, n + 1)
    discounts = curve.discount_factor(ks)
    # Probability of death in year k = survive k-1 years, then die that year.
    probabilities = np.array([
        table.survival(age, k - 1) * table.q_x(age + (k - 1))
        for k in ks
    ])
    benefits = np.full(n, benefit)
    return _apv(discounts, probabilities, benefits)


def whole_life_insurance(
    table: LifeTable,
    curve: YieldCurve,
    age: int,
    benefit: float = 1.0,
) -> float:
    """APV of a whole life insurance on a life aged `age`.

    Pays `benefit` at the end of the year of death, whenever it occurs. This
    is term insurance run to the end of the mortality table (the last age
    with q_x = ... has a final year of coverage).
    """
    # Number of years until the table is exhausted from this age.
    max_years = table.start_age + table.q.size - age
    if max_years < 1:
        raise ValueError(f"age {age} is at or beyond the end of the table")
    return term_insurance(table, curve, age, max_years, benefit=benefit)


def life_annuity_due(
    table: LifeTable,
    curve: YieldCurve,
    age: int,
    n: int | None = None,
    payment: float = 1.0,
) -> float:
    """APV of a life annuity-due on a life aged `age`.

    Pays `payment` at the start of each year while the life survives, for
    k = 0, 1, ..., (n-1) years, or for the whole of life if n is None.

        APV = sum_{k=0}^{n-1}  v^k * {k}p_x * payment

    At k=0 the payment is certain (v^0 = 1, {0}p_x = 1): today's payment.
    """
    if n is None:
        n = table.start_age + table.q.size - age + 1
    if n < 1:
        raise ValueError("annuity term n must be at least 1")

    ks = np.arange(0, n)
    discounts = curve.discount_factor(ks)
    probabilities = np.array([table.survival(age, k) for k in ks])
    payments = np.full(n, payment)
    return _apv(discounts, probabilities, payments)