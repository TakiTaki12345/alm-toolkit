"""Binomial short-rate tree for pricing interest-rate-sensitive instruments.

Until now the toolkit has assumed deterministic rates: given a curve, a
cash flow's value is a fixed number. But some instruments — most notably
callable bonds — have values that depend on how rates evolve. A binomial
rate tree models that evolution as a discrete stochastic process, then
prices by backward induction over the tree.

The tree is *recombining*: an up-move followed by a down-move lands on the
same node as down-then-up, because the short rate evolves by multiplicative
factors and r*u*d == r*d*u. This collapses the node count from 2^n (a full
binary tree) to (n+1)(n+2)/2 ~ n^2/2, which is what makes tree pricing
tractable, and lets the tree be stored as a simple triangular array rather
than linked node objects.

This module builds and inspects the rate tree; pricing lives alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RateTree:
    """A recombining binomial tree of short rates.

    A node is identified by (i, j): time step i (0..n) and, within that
    step, the j-th node counting from the bottom (0..i), reached by j up-moves
    and (i - j) down-moves. Its short rate is

        r(i, j) = r0 * u**j * d**(i - j).

    Because the tree recombines, every path to (i, j) gives the same rate,
    so the value is unique and the whole step i holds exactly i + 1 nodes.

    Parameters
    ----------
    r0 : float
        Initial short rate at the root (step 0), as a decimal.
    u : float
        Up multiplicative factor per step (u > 1 for a rate that can rise).
    d : float
        Down multiplicative factor per step (0 < d < 1 typically, d < u).
    n_steps : int
        Number of time steps (tree has n_steps + 1 levels, 0..n_steps).
    p : float, default 0.5
        Risk-neutral probability of an up-move. Defaults to 0.5.
    dt : float, default 1.0
        Length of each time step in years.
    """

    r0: float
    u: float
    d: float
    n_steps: int
    p: float = 0.5
    dt: float = 1.0
    rates: list[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self):
        # Validate: probabilities and factors must be economically sensible.
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be a probability in [0, 1]")
        # Rates evolve by multiplicative factors, so u and d must be positive.
        # Negative rates are legitimate but are expressed through a negative
        # r0, never through a negative factor — a negative d would make
        # d**(i-j) oscillate in sign and produce nonsensical rates silently.
        if self.u <= 0 or self.d <= 0:
            raise ValueError("u and d must be positive")
        if self.u <= self.d:
            raise ValueError("up factor u must exceed down factor d")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

        self._build()

    def _build(self):
        """Populate the triangular rate array via r(i, j) = r0 * u^j * d^(i-j).

        rates[i] is a length-(i+1) array holding the short rates at step i,
        ordered from the bottom node (j=0, all down-moves) to the top
        (j=i, all up-moves).
        """
        self.rates = []
        for i in range(self.n_steps + 1):
            j = np.arange(i + 1)
            self.rates.append(self.r0 * self.u**j * self.d**(i - j))

    def short_rate(self, i, j):
        """Short rate at node (i, j)."""
        if not (0 <= i <= self.n_steps):
            raise ValueError(f"step i must be in [0, {self.n_steps}]")
        if not (0 <= j <= i):
            raise ValueError(f"node j must be in [0, {i}] at step {i}")
        return float(self.rates[i][j])

    def node_count(self):
        """Total number of nodes: (n+1)(n+2)/2, the recombining count."""
        n = self.n_steps
        return (n + 1) * (n + 2) // 2

    def __repr__(self):
        return (f"RateTree(n_steps={self.n_steps}, r0={self.r0:.4f}, "
                f"u={self.u:.4f}, d={self.d:.4f}, p={self.p})")

    def as_dense(self) -> np.ndarray:
        """Dense (n+1)x(n+1) view for printing or plotting.

        Entries above the diagonal (j > i, non-existent nodes) are NaN. This
        is a one-off convenience view; pricing never routes through it, so the
        efficient ragged triangular storage remains the working representation.
        """
        n = self.n_steps
        dense = np.full((n + 1, n + 1), np.nan)
        for i, row in enumerate(self.rates):
            dense[i, :i + 1] = row
        return dense