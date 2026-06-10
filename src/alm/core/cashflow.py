from __future__ import annotations

import numpy as np

'''
把資料和操作綁在一起，未來要加 IRR、duration、convexity 這些功能時，它們都自然地成為 CashFlow 的方法，
使用者一個物件就能取用所有相關計算，不用記一堆散落的函數。
IRR 是一個好例子，因為 IRR 確實是「現金流自己的性質」，放在類別裡語意最通順。
這就是物件導向裡「高內聚」的概念。
'''

class CashFlow:
    """
    A stream of dated cash flows.

    ----------Parameters----------
    times : array
        Times in years at which cash flows occur. Non-negative.
    amounts : array
        Cash flow amounts. Positive = inflow, negative = outflow.
    """

    def __init__(self, times, amounts):
        times = np.asarray(times, dtype=float)
        amounts = np.asarray(amounts, dtype=float)

        if times.ndim != 1 or amounts.ndim != 1:
            raise ValueError("times and amounts must be one-dimensional")
        if times.shape != amounts.shape:
            raise ValueError("times and amounts must have the same length")
        if times.size == 0:
            raise ValueError("a cash flow must have at least one payment")
        if np.any(times < 0):
            raise ValueError("times must be non-negative")

        # Sort times and amounts simultaneously in ascending order of times
        # Ensure times and amounts are sorted in ascending order.
        order = np.argsort(times)
        self.times = times[order]
        self.amounts = amounts[order]

    def __repr__(self):
        # Return a string representation of the CashFlow object.
        return f"CashFlow(times={self.times.tolist()}, amounts={self.amounts.tolist()})"

    def present_value(self, curve):
        # Return the present value.
        # PV = sum_i  amount_i * discount_factor(time_i)
        discount_factors = curve.discount_factor(self.times)
        return float(np.sum(self.amounts * discount_factors))