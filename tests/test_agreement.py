"""Agreement and discrimination helpers on synthetic ratings."""

import numpy as np
import pytest

from fdbja.metrics import agreement as ag
from fdbja.metrics.discrimination import auc, point_biserial

# --- discrimination ------------------------------------------------------------

def test_auc_perfect_reversed_and_ties():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5      # all tied -> chance
    # one tie between classes counts one half
    assert auc([0, 1], [0.3, 0.3]) == 0.5
    assert auc([0, 0, 1], [0.1, 0.5, 0.5]) == pytest.approx(0.75)


def test_auc_and_point_biserial_empty_class():
    assert auc([1, 1], [0.1, 0.2]) is None
    assert auc([], []) is None
    assert point_biserial([1, 1, 1], [0.1, 0.2, 0.3]) is None
    assert point_biserial([0, 1], [0.1, 0.2]) is None
    assert point_biserial([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


# --- categorical agreement ------------------------------------------------------

def test_pct_agreement():
    assert ag.pct_agreement([[1, 1, 1], [0, 0]]) == 1.0          # all agree
    assert ag.pct_agreement([[0, 1]]) == 0.0
    assert ag.pct_agreement([[0, 0, 1]]) == pytest.approx(1 / 3)
    assert ag.pct_agreement([[1], []]) is None                    # nothing with 2 raters
    assert ag.pct_agreement([]) is None


def test_fleiss_kappa():
    perfect = np.array([[3, 0], [0, 3], [3, 0]])
    assert ag.fleiss_kappa(perfect) == pytest.approx(1.0)
    # every rater picks the same category on every item: P_e = 1 -> undefined
    assert ag.fleiss_kappa(np.array([[3, 0], [3, 0]])) is None
    assert ag.fleiss_kappa(np.array([[3, 0]])) is None              # one item
    assert ag.fleiss_kappa(np.array([[3, 0], [1, 1]])) is None      # unequal rater counts
    # textbook-style check: half agreement split evenly is below chance
    k = ag.fleiss_kappa(np.array([[1, 1], [1, 1], [1, 1]]))
    assert k == pytest.approx(-1.0)


def test_to_count_table_keeps_the_modal_rater_count():
    table, kept = ag.to_count_table([[0, 1, 1], [1, 1], [0, 0, 0], [2, 0, 1]], n_cat=2)
    assert kept == [0, 2, 3]
    assert table.tolist() == [[1, 2], [3, 0], [1, 1]]          # out-of-range label dropped
    table, kept = ag.to_count_table([], n_cat=2)
    assert table.shape == (0, 2) and kept == []


# --- continuous agreement and correlation -------------------------------------

def test_icc1():
    assert ag.icc1([[1, 1], [5, 5], [3, 3]]) == pytest.approx(1.0)   # within-item variance 0
    assert ag.icc1([[1, 2]]) is None
    assert ag.icc1([[1], [2]]) is None                               # no within-item df
    low = ag.icc1([[1, 5], [5, 1], [1, 5], [5, 1]])
    assert low < 0


def test_spearman_ties_and_short_input():
    assert ag.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ag.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert ag.spearman([1, 2], [1, 2]) is None
    assert ag.spearman([1, 1, 1], [1, 2, 3]) is None                 # constant -> undefined
    # ties get average ranks
    assert ag._rankdata([3, 1, 3, 2]).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_bootstrap_ci_and_partial_spearman():
    x = np.arange(30, dtype=float)
    lo, hi = ag.bootstrap_ci(x, x * 2, n_boot=200)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert ag.bootstrap_ci([1, 2, 3], [1, 2, 3]) is None
    # x and y both driven by z: the partial correlation removes it
    rng = np.random.default_rng(0)
    z = rng.random(200)
    a = z + 0.01 * rng.random(200)
    b = z + 0.01 * rng.random(200)
    assert ag.spearman(a, b) > 0.9
    assert abs(ag.partial_spearman(a, b, [z])) < 0.3
    assert ag.partial_spearman(a, b, []) == pytest.approx(ag._pearson(ag._rankdata(a), ag._rankdata(b)))
    assert ag.partial_spearman([1, 2, 3], [1, 2, 3], [[1, 2, 3]]) is None
