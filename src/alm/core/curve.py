from __future__ import annotations

import numpy as np


class YieldCurve:
    """A zero-coupon yield curve with linear interpolation.

    The curve is defined by a set of (tenor, rate) sample points, where the
    rates are continuously-compounded zero rates. Rates for arbitrary tenors
    are obtained by linear interpolation between the sample points. This is
    the foundational primitive of the toolkit: everything else (bond pricing,
    risk measures, immunization) discounts through this curve.

    Parameters
    ----------
    tenors : array-like
        Maturities in years, e.g. [1, 2, 5, 10]. Must be strictly increasing.
    rates : array-like
        Continuously-compounded zero rates as decimals (e.g. 0.03 for 3%),
        one per tenor.
    """

    def __init__(self, tenors, rates):
        tenors = np.asarray(tenors, dtype=float)
        rates = np.asarray(rates, dtype=float)

        # Strict validation: unlike a cash flow's timings, disordered tenors
        # are a genuine data error and would silently corrupt np.interp, so
        # we reject rather than sort.
        if tenors.ndim != 1 or rates.ndim != 1:
            raise ValueError("tenors and rates must be one-dimensional")
        if tenors.shape != rates.shape:
            raise ValueError("tenors and rates must have the same length")
        if tenors.size < 2:
            raise ValueError("at least two sample points are required")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")

        self.tenors = tenors
        self.rates = rates

    def zero_rate(self, t):
        """Continuously-compounded zero rate at tenor t (years).

        Between sample points the rate is linearly interpolated. Outside the
        sample range np.interp clamps to the nearest endpoint (flat
        extrapolation) rather than extending the trend.

        Flat extrapolation is a deliberate, conservative choice: beyond the
        curve's ends there is usually no reliable market data, and extending
        the trend would fabricate rates in that region — potentially absurd
        ones (a short-end slope pushed to t=0 could imply negative rates; a
        long-end slope could run off to implausibly high ones). Clamping adds
        no unproven assumptions, which is the safer default in risk work.
        """
        t = np.asarray(t, dtype=float)
        return np.interp(t, self.tenors, self.rates)

    def discount_factor(self, t):
        """Discount factor for time t (years): DF(t) = exp(-r(t) * t).

        Continuous compounding is chosen over discrete 1/(1+r)^t because its
        derivative stays clean, which makes the duration and convexity
        formulas built on top of it far simpler.
        """
        t = np.asarray(t, dtype=float)
        r = self.zero_rate(t)
        return np.exp(-r * t)

    def forward_rate(self, t1, t2):
        """Continuously-compounded forward rate for the future period [t1, t2].

        The forward rate is the rate, agreed today, for lending over a future
        interval. It is not independent data — it is implied by the spot curve
        through no-arbitrage: investing to t2 directly must equal investing to
        t1 and rolling forward at f(t1, t2). Under continuous compounding,

            exp(r2*t2) = exp(r1*t1) * exp(f*(t2 - t1))

        which rearranges to

            f(t1, t2) = (r2*t2 - r1*t1) / (t2 - t1).

        As t2 -> t1 this tends to the instantaneous forward rate; here we
        require t2 > t1 so the discrete forward rate is well defined.
        """
        t1 = float(t1)
        t2 = float(t2)
        if t2 <= t1:
            raise ValueError("t2 must be strictly greater than t1")

        r1 = self.zero_rate(t1)
        r2 = self.zero_rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)