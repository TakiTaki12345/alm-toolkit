"""Tests for the LifeTable (mortality) module."""

import numpy as np
import pytest

from alm.core.mortality import LifeTable


def make_table(start_age=40):
    # A small illustrative table: mortality rising with age.
    q = [0.01, 0.012, 0.015, 0.02, 0.03]
    return LifeTable(q, start_age=start_age)


def test_p_x_complements_q_x():
    """p_x = 1 - q_x."""
    table = make_table()
    assert table.p_x(40) == pytest.approx(1 - 0.01)
    assert table.p_x(42) == pytest.approx(1 - 0.015)


def test_survival_one_year_equals_p_x():
    """Surviving exactly one integer year equals p_x."""
    table = make_table()
    assert table.survival(40, 1) == pytest.approx(table.p_x(40), rel=1e-12)


def test_survival_integer_years_is_cumulative_product():
    """t_p_x over integer years is the product of annual survival probs."""
    table = make_table()
    expected = (1 - 0.01) * (1 - 0.012) * (1 - 0.015)  # survive ages 40,41,42
    assert table.survival(40, 3) == pytest.approx(expected, rel=1e-12)


def test_survival_is_monotone_decreasing():
    """Survival probability decreases as the horizon lengthens."""
    table = make_table()
    assert table.survival(40, 1) > table.survival(40, 2) > table.survival(40, 3)


def test_assumptions_agree_on_integer_years():
    """UDD and constant force must give identical results at integer t,
    since the fractional part vanishes."""
    table = make_table()
    for t in [1, 2, 3]:
        udd = table.survival(40, t, assumption="udd")
        cf = table.survival(40, t, assumption="constant_force")
        assert udd == pytest.approx(cf, rel=1e-12)


def test_udd_fractional_is_linear():
    """Under UDD, s_p_x = 1 - s*q_x — linear in s within the year."""
    table = make_table()
    q40 = 0.01
    # Half-year survival from age 40.
    assert table.survival(40, 0.5, assumption="udd") == pytest.approx(1 - 0.5 * q40, rel=1e-12)


def test_constant_force_fractional_is_exponential():
    """Under constant force, s_p_x = (p_x)^s = exp(-mu*s) with mu = -ln(p_x).

    This mirrors continuous compounding, and we cross-check the two forms.
    """
    table = make_table()
    p40 = 1 - 0.01
    s = 0.5
    direct = table.survival(40, s, assumption="constant_force")
    as_power = p40 ** s
    mu = -np.log(p40)
    as_exp = np.exp(-mu * s)
    assert direct == pytest.approx(as_power, rel=1e-12)
    assert direct == pytest.approx(as_exp, rel=1e-12)


def test_fractional_plus_integer_decomposition():
    """t_p_x for t = 2.5 equals 2_p_x * 0.5_p_{x+2}."""
    table = make_table()
    two_year = table.survival(40, 2)
    half_from_42 = table.survival(42, 0.5)
    assert table.survival(40, 2.5) == pytest.approx(two_year * half_from_42, rel=1e-12)


def test_from_lx_matches_direct_q():
    """Building from l_x must reproduce the same survival probabilities as
    building from the equivalent q_x directly."""
    # Choose l_x, derive the implied q_x, build both ways.
    l = [1000.0, 990.0, 978.0, 963.0]  # numbers living at ages 40..43
    table_l = LifeTable.from_lx(l, start_age=40)

    # Implied q_x: (l_x - l_{x+1}) / l_x.
    q_implied = [(1000 - 990) / 1000, (990 - 978) / 990, (978 - 963) / 978]
    table_q = LifeTable(q_implied, start_age=40)

    assert table_l.survival(40, 3) == pytest.approx(table_q.survival(40, 3), rel=1e-12)
    assert table_l.q_x(41) == pytest.approx(table_q.q_x(41), rel=1e-12)


def test_deferred_death_prob_complements_survival():
    """t_q_x = 1 - t_p_x."""
    table = make_table()
    assert table.deferred_death_prob(40, 3) == pytest.approx(1 - table.survival(40, 3), rel=1e-12)


def test_rejects_out_of_range_q():
    with pytest.raises(ValueError):
        LifeTable([0.01, 1.5, 0.02])  # 1.5 is not a valid probability


def test_from_lx_rejects_increasing_lx():
    with pytest.raises(ValueError):
        LifeTable.from_lx([1000, 1010, 990])  # numbers living can't increase