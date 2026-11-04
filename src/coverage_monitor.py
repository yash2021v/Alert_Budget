"""Binomial-based coverage error indicator.

The plain flag "realised miss rate <= bound" is noisy at small n: even a perfectly valid bound covers an OBSERVED rate
only about 82% of the time at delta 0.10. The indicator below is exact instead.

For a step/period t with n_t positives, k_t misses and certificate bound b_t:
    err_t = 1  if  k_t > binom.ppf(1 - delta_err, n_t, b_t),  else 0.
If the bound is valid (true miss rate p <= b_t), k_t ~ Binomial(n_t, p) is stochastically no larger than
Binomial(n_t, b_t), so P(err_t = 1) <= delta_err at any n_t. delta_err = 0.10 is fixed.
n_t = 0 returns err = 0 and is excluded from averages (use `error_frequency`).
"""
import numpy as np
from scipy.stats import binom

DELTA_ERR = 0.10


def error_indicator(k: int, n: int, bound: float, delta_err: float = DELTA_ERR) -> int:
    if n <= 0:
        return 0
    return int(k > binom.ppf(1 - delta_err, n, min(max(bound, 0.0), 1.0)))


def error_frequency(errs, ns) -> float:
    """Mean of the error indicators over steps with at least one positive (n_t = 0 steps are excluded)."""
    errs, ns = np.asarray(errs, dtype=float), np.asarray(ns)
    keep = ns > 0
    return float(errs[keep].mean()) if keep.any() else float("nan")
