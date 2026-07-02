from __future__ import annotations

import numpy as np


class CashFlow:
    """A stream of dated cash flows.

    Both assets (e.g. bonds) and liabilities (e.g. annuity payments) are
    expressed as cash flow streams, making this the common currency of the
    whole toolkit.

    Design note — data and operations are bound together in one object so
    that valuation logic (present value, and later IRR, duration) lives as
    methods on the stream it acts on. IRR, for instance, is an intrinsic
    property of a cash flow, so it reads most naturally as a method. This
    is the object-oriented principle of high cohesion. Internally the data
    is held as parallel NumPy arrays so those operations stay vectorized.

    Parameters
    ----------
    times : array-like
        Times in years at which cash flows occur. Must be non-negative.
    amounts : array-like
        Cash flow amounts. Positive = inflow, negative = outflow.
    """

    def __init__(self, times, amounts):
        times = np.asarray(times, dtype=float)
        amounts = np.asarray(amounts, dtype=float)

        # Validate on construction so every method downstream can trust the
        # data is well-formed (strict on genuine errors).
        if times.ndim != 1 or amounts.ndim != 1:
            raise ValueError("times and amounts must be one-dimensional")
        if times.shape != amounts.shape:
            raise ValueError("times and amounts must have the same length")
        if times.size == 0:
            raise ValueError("a cash flow must have at least one payment")
        if np.any(times < 0):
            raise ValueError("times must be non-negative")

        # Sort chronologically. Out-of-order input is a matter of convenience,
        # not an error, so we quietly fix it (forgiving on harmless disorder)
        # and let downstream logic assume chronological order.
        order = np.argsort(times)
        self.times = times[order]
        self.amounts = amounts[order]

    def __repr__(self):
        return f"CashFlow(times={self.times.tolist()}, amounts={self.amounts.tolist()})"

    def present_value(self, curve):
        """Present value of the stream under a given yield curve.

        PV = sum_i amount_i * discount_factor(time_i)

        The curve is passed in rather than held as state (dependency
        injection), keeping CashFlow agnostic to how discounting is done.
        """
        discount_factors = curve.discount_factor(self.times)
        return float(np.sum(self.amounts * discount_factors))