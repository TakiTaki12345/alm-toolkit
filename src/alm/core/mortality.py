"""Mortality: life tables and survival probabilities.

A life table is the actuarial counterpart of a yield curve — a foundational,
application-agnostic primitive. Where a curve answers "what is money worth
across time?", a life table answers "who is still alive across time?".
Life-insurance products combine both.

The table is stored internally as annual mortality rates q_x (the probability
that a life aged x dies within one year), which is the natural unit of
actuarial calculation. A life table given as l_x (numbers living) carries the
same information — l_x's absolute scale is meaningless, only its ratios are —
so l_x is offered as an alternative constructor rather than a separate class:
these are two representations of one thing, not two behaviours.

Fractional-year survival requires an assumption about how deaths are spread
within a year. Two standards are supported:

  UDD (uniform distribution of deaths):  s_q_x = s * q_x,  so  s_p_x = 1 - s*q_x
  Constant force of mortality:           s_p_x = (p_x)^s

Constant force mirrors continuous compounding exactly: (p_x)^s = exp(-mu*s)
with mu = -ln(p_x), so the force of mortality plays the role that the interest
rate plays in discounting. UDD is the textbook default.
"""

from __future__ import annotations

import numpy as np


class LifeTable:
    """A life table indexed from a starting age, holding annual rates q_x.

    Parameters
    ----------
    q : array-like
        Annual mortality rates q_x for ages start_age, start_age+1, ...
        Each must lie in [0, 1].
    start_age : int, default 0
        The age corresponding to q[0].
    """

    def __init__(self, q, start_age=0):
        q = np.asarray(q, dtype=float)

        if q.ndim != 1:
            raise ValueError("q must be one-dimensional")
        if q.size == 0:
            raise ValueError("a life table needs at least one rate")
        if np.any(q < 0) or np.any(q > 1):
            raise ValueError("each q_x must be a probability in [0, 1]")

        self.q = q
        self.start_age = int(start_age)

    @classmethod
    def from_lx(cls, l, start_age=0):
        """Build from numbers-living l_x, converting to q_x internally.

        q_x = (l_x - l_{x+1}) / l_x. The final entry of l has no successor,
        so the resulting table has one fewer age than l.
        """
        l = np.asarray(l, dtype=float)
        if l.ndim != 1:
            raise ValueError("l must be one-dimensional")
        if l.size < 2:
            raise ValueError("l must contain at least two ages")
        if np.any(l <= 0):
            raise ValueError("l_x must be positive")
        if np.any(np.diff(l) > 0):
            raise ValueError("l_x must be non-increasing")

        q = (l[:-1] - l[1:]) / l[:-1]
        return cls(q, start_age=start_age)

    def _index(self, age):
        """Array index for a given integer age, with bounds checking."""
        i = int(age) - self.start_age
        if not (0 <= i < self.q.size):
            raise ValueError(
                f"age {age} is outside the table's range "
                f"[{self.start_age}, {self.start_age + self.q.size - 1}]"
            )
        return i

    def q_x(self, age):
        """Probability that a life aged `age` dies within one year."""
        return float(self.q[self._index(age)])

    def p_x(self, age):
        """Probability that a life aged `age` survives one year."""
        return 1.0 - self.q_x(age)

    def survival(self, age, t, assumption="udd"):
        """Probability that a life aged `age` survives t years (t may be fractional).

        Decomposes t into an integer part n and a fractional part s:

            t_p_x = n_p_x * s_p_{x+n}

        The integer part is the cumulative product of annual survival
        probabilities. The fractional part follows the chosen assumption:

            "udd"            : s_p_y = 1 - s * q_y   (deaths uniform in the year)
            "constant_force" : s_p_y = (p_y) ** s    (constant force of mortality)

        Parameters
        ----------
        age : int
            Current (integer) age.
        t : float
            Years survived; must be non-negative.
        assumption : {"udd", "constant_force"}, default "udd"
            Fractional-year death distribution assumption.

        Returns
        -------
        float
        """
        if t < 0:
            raise ValueError("t must be non-negative")
        if assumption not in ("udd", "constant_force"):
            raise ValueError('assumption must be "udd" or "constant_force"')

        n = int(np.floor(t))
        s = t - n

        # Integer years: cumulative product of p_x over ages age .. age+n-1.
        i = self._index(age)
        if i + n > self.q.size:
            raise ValueError(
                f"surviving {n} integer years from age {age} exceeds the table"
            )
        integer_part = float(np.prod(1.0 - self.q[i:i + n]))

        if s == 0.0:
            return integer_part

        # Fractional remainder, applied at the attained age age+n.
        j = self._index(age + n)
        q_y = float(self.q[j])
        if assumption == "udd":
            fractional_part = 1.0 - s * q_y
        else:  # constant_force
            fractional_part = (1.0 - q_y) ** s

        return integer_part * fractional_part

    def deferred_death_prob(self, age, t, assumption="udd"):
        """Probability that a life aged `age` dies within t years: t_q_x = 1 - t_p_x."""
        return 1.0 - self.survival(age, t, assumption=assumption)