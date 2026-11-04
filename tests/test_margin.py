import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import scoring


@pytest.fixture(scope="module")
def sc():
    return scoring.load_scores()


def test_margin_order_matches_probability_order_on_random_pairs(sc):
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(sc), 400_000), rng.integers(0, len(sc), 400_000)
    p, m = sc["score"].to_numpy(), sc["margin"].to_numpy()
    differ = p[i] != p[j]
    assert differ.sum() > 100_000
    assert (np.sign(p[i] - p[j])[differ] == np.sign(m[i] - m[j])[differ]).all()


def test_margin_is_monotone_in_probability_everywhere(sc):
    s = sc.sort_values("margin")
    assert (np.diff(s["score"].to_numpy()) >= 0).all()


def test_probability_ties_are_resolved_by_margin(sc):
    top = sc[sc["score"] >= 1.0]
    # probability gives one level for these rows; margin separates them into thousands of levels
    assert len(top) > 10_000 * C.SAMPLE_FRAC and top["margin"].nunique() > 1_000 * C.SAMPLE_FRAC


def test_load_scores_is_seeded():
    a, b = scoring.load_scores(), scoring.load_scores()
    pd.testing.assert_frame_equal(a, b)
