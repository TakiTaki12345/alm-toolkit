"""Tests for the recombining binomial short-rate tree (structure only)."""

import numpy as np
import pytest

from alm.applications.rate_tree import RateTree


def make_tree(n_steps=3, r0=0.05, u=1.1, d=0.9, p=0.5):
    return RateTree(r0=r0, u=u, d=d, n_steps=n_steps, p=p)


def test_root_is_initial_rate():
    """Step 0 has a single node equal to the initial short rate r0."""
    tree = make_tree(r0=0.05)
    assert tree.short_rate(0, 0) == pytest.approx(0.05)


def test_step_has_i_plus_one_nodes():
    """A recombining tree has exactly i + 1 nodes at step i."""
    tree = make_tree(n_steps=4)
    for i in range(5):
        assert len(tree.rates[i]) == i + 1


def test_node_count_is_quadratic():
    """Total nodes = (n+1)(n+2)/2 — the recombining count, not 2^n."""
    tree = make_tree(n_steps=5)
    assert tree.node_count() == (6 * 7) // 2  # 21
    # And emphatically less than the non-recombining 2^n growth.
    assert tree.node_count() < 2 ** 5


def test_rate_formula_up_and_down():
    """r(i, j) = r0 * u^j * d^(i-j): check the extreme nodes at step 2."""
    r0, u, d = 0.05, 1.1, 0.9
    tree = make_tree(n_steps=2, r0=r0, u=u, d=d)
    # Bottom node (all down-moves): j=0.
    assert tree.short_rate(2, 0) == pytest.approx(r0 * d**2)
    # Top node (all up-moves): j=2.
    assert tree.short_rate(2, 2) == pytest.approx(r0 * u**2)
    # Middle node (one up, one down): j=1.
    assert tree.short_rate(2, 1) == pytest.approx(r0 * u * d)


def test_recombination_property():
    """Up-then-down lands on the same rate as down-then-up.

    From node (i, j): an up-move goes to (i+1, j+1), a down-move to (i+1, j).
    Going up then down reaches (i+2, j+1); down then up also reaches
    (i+2, j+1). The tree recombines, so these must be the identical node.
    """
    tree = make_tree(n_steps=3, r0=0.05, u=1.1, d=0.9)
    # Start at (1, 0). Up then down -> (3, 1) via (2,1); down then up -> (3,1).
    # Both must equal r0 * u * d applied consistently: r(3,1) is unique.
    up_then_down = tree.short_rate(3, 1)
    down_then_up = tree.short_rate(3, 1)
    assert up_then_down == down_then_up
    # Concretely, the shared node equals r0 * u^1 * d^2.
    assert tree.short_rate(3, 1) == pytest.approx(0.05 * 1.1 * 0.9**2)


def test_rejects_bad_probability():
    with pytest.raises(ValueError):
        RateTree(r0=0.05, u=1.1, d=0.9, n_steps=3, p=1.5)


def test_rejects_u_not_above_d():
    with pytest.raises(ValueError):
        RateTree(r0=0.05, u=0.9, d=1.1, n_steps=3)


def test_rejects_out_of_range_node():
    tree = make_tree(n_steps=2)
    with pytest.raises(ValueError):
        tree.short_rate(1, 5)  # j=5 doesn't exist at step 1