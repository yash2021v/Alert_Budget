import numpy as np

from src.coverage_monitor import error_frequency, error_indicator


def _rate(n, b, p, trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    ks = rng.binomial(n, p, trials)
    return np.mean([error_indicator(int(k), n, b) for k in ks])


def test_valid_bound_error_rate_is_at_most_about_ten_percent():
    for n in (5, 20, 100, 1000):
        for b in (0.1, 0.5, 0.9):
            assert _rate(n, b, b) <= 0.10 + 0.01          # p = b is the boundary case
            assert _rate(n, b, b * 0.8) <= 0.10 + 0.01     # conservative bound: even fewer errors


def test_error_rate_is_much_higher_when_true_miss_rate_exceeds_bound():
    for n in (50, 200):
        assert _rate(n, 0.5, 0.8) > 0.8
    assert _rate(100, 0.3, 0.45) > 0.8


def test_zero_positives_is_not_an_error_and_is_excluded_from_averages():
    assert error_indicator(0, 0, 0.5) == 0
    assert error_frequency([1, 0, 1, 0], [10, 5, 0, 8]) == 1 / 3     # the n = 0 step (err 0 or 1) is dropped
    import math
    assert math.isnan(error_frequency([0], [0]))


def test_hand_checkable_values():
    # n = 10, b = 0.5: ppf(0.9) = 7, so 8 misses is an error and 7 is not
    assert error_indicator(7, 10, 0.5) == 0 and error_indicator(8, 10, 0.5) == 1
    assert error_indicator(5, 10, 1.0) == 0                      # bound 1: never an error
