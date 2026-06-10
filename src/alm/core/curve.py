from __future__ import annotations

import numpy as np

class YieldCurve:
    """
    A zero-coupon yield curve with linear interpolation.

    The curve is defined by a set of (tenor, rate) sample points.
    rates: continuously-compounded zero rates.
    Rates for any tenors are obtained by linear interpolation between the sample points.

    ----------Parameters----------
    tenors : array
        Maturities in years, ex. [1, 2, 5, 10].
        It is strictly increasing.
    rates : array
        Continuously-compounded zero rates, Decimals, ex. 0.03 for 3%.
        corresponding to each tenor.
    """

    def __init__(self, tenors, rates):
        tenors = np.asarray(tenors, dtype=float)
        rates = np.asarray(rates, dtype=float)

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
        t = np.asarray(t, dtype=float)
        # Find the rate reference for tenor t by linear interpolation
        # If t is less than min(tenors), return rates[0]
        # If t is greater than max(tenors), return rates[-1]
        return np.interp(t, self.tenors, self.rates)

    def discount_factor(self, t):
        t = np.asarray(t, dtype=float)
        r = self.zero_rate(t)   # DF(t) = exp(-r(t) * t)
        return np.exp(-r * t)   # Return the discount factor for time t (years).