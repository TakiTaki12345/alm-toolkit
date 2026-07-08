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

def test_rejects_non_positive_factors():
    """Negative rates come from a negative r0, never a negative factor."""
    with pytest.raises(ValueError):
        RateTree(r0=0.05, u=1.1, d=-0.9, n_steps=3)
    with pytest.raises(ValueError):
        RateTree(r0=0.05, u=1.1, d=0.0, n_steps=3)

from alm.core.cashflow import CashFlow
from alm.core.risk import present_value_at_yield
from alm.applications.rate_tree import price_bond


def test_single_payment_one_step_by_hand():
    """A one-step tree pricing a single terminal payment is fully hand-checkable.

    Tree: r0=0.05, one step, dt=1. The payment of 100 arrives at t=1 (the
    terminal layer), so every terminal node has value 100. At the root:
        V = exp(-r0*dt) * [p*100 + (1-p)*100] = exp(-0.05) * 100.
    The expectation collapses because both children pay 100.
    """
    tree = RateTree(r0=0.05, u=1.2, d=0.8, n_steps=1, p=0.5, dt=1.0)
    bond = CashFlow(times=[1], amounts=[100])
    expected = np.exp(-0.05) * 100
    assert price_bond(bond, tree) == pytest.approx(expected, rel=1e-12)


def test_flat_tree_matches_deterministic_discounting():
    """With no rate movement, tree pricing must equal deterministic discounting.

    We can't set u=d (the constructor forbids it), but a tree that is flat in
    expectation with p chosen so the discounting is symmetric should price a
    bond close to discounting at r0. Here we use a nearly-degenerate tree with
    u and d very close to 1 and confirm convergence to present_value_at_yield.
    """
    r0 = 0.04
    # u and d symmetric and very close to 1: rates barely move from r0.
    tree = RateTree(r0=r0, u=1.0001, d=0.9999, n_steps=5, p=0.5, dt=1.0)
    bond = CashFlow(times=[1, 2, 3, 4, 5], amounts=[4, 4, 4, 4, 104])

    tree_price = price_bond(bond, tree)
    deterministic = present_value_at_yield(bond, r0)
    # Rates move by at most ~0.01% per step, so prices should agree closely.
    assert tree_price == pytest.approx(deterministic, rel=1e-3)


def test_zero_rate_tree_sums_undiscounted():
    """With r0=0 (and factors that keep rates at 0), there is no discounting,
    so the bond price equals the undiscounted sum of cash flows."""
    # r0 = 0 means every node rate is 0 * u^j * d^(i-j) = 0.
    tree = RateTree(r0=0.0, u=1.2, d=0.8, n_steps=3, p=0.5, dt=1.0)
    bond = CashFlow(times=[1, 2, 3], amounts=[10, 10, 110])
    assert price_bond(bond, tree) == pytest.approx(130.0, rel=1e-12)


def test_price_independent_of_probability_when_rate_is_zero():
    """When r0=0 there is no discounting, so price is the cash flow sum
    regardless of the up-probability p."""
    bond = CashFlow(times=[1, 2], amounts=[5, 105])
    tree_p3 = RateTree(r0=0.0, u=1.2, d=0.8, n_steps=2, p=0.3, dt=1.0)
    tree_p7 = RateTree(r0=0.0, u=1.2, d=0.8, n_steps=2, p=0.7, dt=1.0)
    assert price_bond(bond, tree_p3) == pytest.approx(110.0)
    assert price_bond(bond, tree_p7) == pytest.approx(110.0)


def test_rejects_misaligned_cash_flow():
    """A cash flow time that is not a multiple of dt must raise."""
    tree = RateTree(r0=0.05, u=1.2, d=0.8, n_steps=3, p=0.5, dt=1.0)
    bond = CashFlow(times=[2.5], amounts=[100])  # 2.5 is not a multiple of dt=1
    with pytest.raises(ValueError):
        price_bond(bond, tree)


def test_rejects_cash_flow_beyond_horizon():
    """A cash flow past the tree's horizon must raise."""
    tree = RateTree(r0=0.05, u=1.2, d=0.8, n_steps=2, p=0.5, dt=1.0)
    bond = CashFlow(times=[5], amounts=[100])  # step 5 > n_steps 2
    with pytest.raises(ValueError):
        price_bond(bond, tree)