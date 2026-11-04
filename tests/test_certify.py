import itertools
import math

import numpy as np
import pytest
from scipy.stats import binomtest

from src.certify import certify, clopper_pearson_upper, missed_counts


def test_hand_checkable_zero_misses():
    # k=0: upper bound solves (1-p)^n = delta -> p = 1 - delta^(1/n)
    assert clopper_pearson_upper(0, 10, 0.10) == pytest.approx(1 - 0.1 ** (1 / 10), abs=1e-12)
    assert clopper_pearson_upper(0, 10, 0.10) == pytest.approx(0.2057, abs=1e-4)


def test_edge_cases():
    assert clopper_pearson_upper(0, 0, 0.1) == 1.0
    assert clopper_pearson_upper(7, 7, 0.1) == 1.0
    assert clopper_pearson_upper(5, 5, 0.05) == 1.0


@pytest.mark.parametrize("k,n,delta", [(0, 10, 0.10), (0, 50, 0.05), (1, 20, 0.10), (3, 40, 0.05),
                                       (17, 100, 0.10), (60, 100, 0.05), (99, 100, 0.10), (500, 1000, 0.05)])
def test_matches_scipy_two_sided_interval(k, n, delta):
    # one-sided bound at 1-delta == upper end of the two-sided exact interval at 1-2*delta
    ref = binomtest(k, n).proportion_ci(confidence_level=1 - 2 * delta, method="exact").high
    assert clopper_pearson_upper(k, n, delta) == pytest.approx(ref, abs=1e-9)


def test_k_equals_n_matches_scipy_value():
    ref = binomtest(10, 10).proportion_ci(confidence_level=0.8, method="exact").high
    assert ref == pytest.approx(1.0)
    assert clopper_pearson_upper(10, 10, 0.10) == 1.0


def test_monotone_in_k():
    for n, delta in itertools.product([10, 50, 300], [0.05, 0.10]):
        b = [clopper_pearson_upper(k, n, delta) for k in range(n + 1)]
        assert all(b[i] <= b[i + 1] + 1e-15 for i in range(n))


def test_decreasing_in_n_at_fixed_rate():
    for rate, delta in itertools.product([0.0, 0.1, 0.5], [0.05, 0.10]):
        b = [clopper_pearson_upper(int(rate * n), n, delta) for n in (10, 20, 50, 100, 1000)]
        assert all(b[i] >= b[i + 1] for i in range(len(b) - 1))


def test_bound_at_least_point_estimate():
    for k, n in [(0, 30), (5, 30), (29, 30)]:
        assert clopper_pearson_upper(k, n, 0.1) >= k / n


def test_per_period_top_k_hand_example():
    # period 0: scores .9 .8 .7 .1 ; positives at .8 and .1  -> K=2 reviews .9,.8 -> .1 missed
    # period 1: scores .5 .4 .3   ; positive at .3           -> K=2 reviews .5,.4 -> .3 missed
    scores = [0.9, 0.8, 0.7, 0.1, 0.5, 0.4, 0.3]
    labels = [0, 1, 0, 1, 0, 0, 1]
    periods = [0, 0, 0, 0, 1, 1, 1]
    assert missed_counts(scores, labels, periods, 2) == (2, 3)
    assert missed_counts(scores, labels, periods, 3) == (1, 3)    # only .1 outside (period 1 fully reviewed)
    assert missed_counts(scores, labels, periods, 4) == (0, 3)    # everything reviewed
    assert missed_counts(scores, labels, periods, 1) == (3, 3)    # top-1 are .9 and .5 (both negatives)
    assert missed_counts(scores, labels, periods, 100) == (0, 3)


def test_ties_broken_by_input_position():
    # three tied entities in one period, K=1: the first in input order is reviewed
    assert missed_counts([0.5, 0.5, 0.5], [1, 0, 0], [0, 0, 0], 1) == (0, 1)
    assert missed_counts([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 0], 1) == (1, 1)
    # a higher score always beats an earlier-positioned tie
    assert missed_counts([0.5, 0.5, 0.6], [1, 0, 0], [0, 0, 0], 1) == (1, 1)


def test_period_order_in_input_does_not_matter():
    scores = [0.3, 0.9, 0.4, 0.8]
    labels = [1, 0, 1, 0]
    # period 1 holds (.3, .4): .4 reviewed, .3 missed; period 0 holds (.9, .8) negatives
    assert missed_counts(scores, labels, [1, 0, 1, 0], 1) == (1, 2)


def test_certify_dict():
    out = certify([0.9, 0.8, 0.7, 0.1, 0.5, 0.4, 0.3], [0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1], 2, 0.10)
    assert out["n_pos"] == 3 and out["n_missed"] == 2 and out["K"] == 2 and out["delta"] == 0.10
    assert out["point_miss_rate"] == pytest.approx(2 / 3)
    assert out["upper_bound"] == pytest.approx(
        binomtest(2, 3).proportion_ci(confidence_level=0.8, method="exact").high, abs=1e-9)


def test_no_positives():
    out = certify([0.1, 0.2], [0, 0], [0, 0], 1, 0.1)
    assert out["n_pos"] == 0 and out["upper_bound"] == 1.0 and math.isnan(out["point_miss_rate"])


def test_fraction_k_is_ceil_of_q_times_n():
    from src.certify import fraction_k
    periods = [0] * 8 + [1] * 10 + [2] * 3
    k = fraction_k(periods, 0.25)
    assert list(k[:8]) == [2] * 8                       # 0.25 * 8 = 2 exactly, no float bump to 3
    assert list(k[8:18]) == [3] * 10                    # ceil(2.5)
    assert list(k[18:]) == [1] * 3                      # ceil(0.75)
    assert list(fraction_k([0] * 1000, 0.001)) == [1] * 1000     # 0.001 * 1000 = 1 (not 2)


def test_per_period_k_array_hand_example():
    # period 0 (4 rows, K=1) and period 1 (3 rows, K=2); positives at .8 / .1 / .3
    scores = [0.9, 0.8, 0.7, 0.1, 0.5, 0.4, 0.3]
    labels = [0, 1, 0, 1, 0, 0, 1]
    periods = [0, 0, 0, 0, 1, 1, 1]
    K = [1, 1, 1, 1, 2, 2, 2]
    assert missed_counts(scores, labels, periods, K) == (3, 3)   # .8 and .1 outside top-1; .3 outside top-2
    assert missed_counts(scores, labels, periods, [2, 2, 2, 2, 3, 3, 3]) == (1, 3)


def test_group_certificates_tiny_example():
    from src.certify import group_certificates
    flags = [True, True, False, False, False, True]
    groups = ["A", "A", "A", "B", "B", "C"]
    out = group_certificates(flags, groups, 0.10)
    assert out["A"]["n_pos"] == 3 and out["A"]["n_missed"] == 2
    assert out["B"]["n_pos"] == 2 and out["B"]["n_missed"] == 0
    assert out["C"]["upper_bound"] == 1.0                       # k = n
    assert out["B"]["upper_bound"] == pytest.approx(1 - 0.1 ** (1 / 2))
    assert out["A"]["upper_bound"] == pytest.approx(
        binomtest(2, 3).proportion_ci(confidence_level=0.8, method="exact").high, abs=1e-9)


def test_max_over_transaction_aggregation_hand_example():
    import pandas as pd
    from src.transaction_model import txn_margin_max_by_entity
    tx = pd.DataFrame({"from_key": [1, 1, 2, 3], "to_key": [2, 3, 3, 1], "period": [0, 0, 0, 1],
                       "txn_margin": [0.5, 2.0, -1.0, 7.0]})
    out = txn_margin_max_by_entity(tx).set_index(["acct_key", "period"])["max_txn_margin"]
    assert out[(1, 0)] == 2.0          # sender of 0.5 and 2.0
    assert out[(2, 0)] == 0.5          # receiver of 0.5, sender of -1.0
    assert out[(3, 0)] == 2.0          # receiver of 2.0 and -1.0
    assert out[(3, 1)] == 7.0 and out[(1, 1)] == 7.0   # a different period is a different entity-period
