"""Certified miss rates for capacity-constrained alert triage (pure functions, no I/O).

Setting. In every period the K highest-scoring entity-periods are reviewed. A positive entity-period is
*missed* if it is not in the top K of its period. The miss rate (false omission rate) is
missed positives / all positives.

Certificate. From the calibration positives, `certify` returns the exact one-sided Clopper-Pearson upper
confidence bound on the true miss rate at confidence 1 - delta.

VALIDITY CONDITION. The bound covers the true miss rate with probability at least 1 - delta ONLY IF the
positives seen at deployment are exchangeable with the calibration positives (each positive is missed
independently with the same probability, for the same fixed score function and K). Temporal drift of the
score distribution, and dependence between consecutive periods for the same account, can break this; those
are exactly what hypotheses H2 and H4 test. The bound says nothing about a drifted deployment.

Tie handling. Within a period, entities are ranked by score descending; ties are broken by input position
(the entity that appears earlier in the input arrays ranks higher). This is deterministic for a fixed input
order but is not random, so callers with many exact ties should shuffle the input order once, up front.
"""
import numpy as np
from scipy.stats import beta


def missed_flags(scores, labels, period_ids, K: int) -> np.ndarray:
    """Boolean array over the POSITIVES (in input order): True if that positive is outside its period's top K.

    Ranking is within each period by score descending, ties broken by input position (earlier = higher).
    A period with fewer than K entities reviews everything.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    periods = np.asarray(period_ids)
    n = len(scores)
    if not (len(labels) == len(periods) == n):
        raise ValueError("scores, labels and period_ids must have the same length")
    pos_idx = np.arange(n)
    # sort by period, then score descending, then input position ascending
    order = np.lexsort((pos_idx, -scores, periods))
    p_sorted = periods[order]
    starts = np.r_[True, p_sorted[1:] != p_sorted[:-1]]
    group_start = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    rank = np.arange(n) - group_start                  # 0-based rank within period
    reviewed = np.empty(n, dtype=bool)
    reviewed[order] = rank < K
    is_pos = labels == 1
    return ~reviewed[is_pos]


def missed_counts(scores, labels, period_ids, K: int) -> tuple[int, int]:
    """Return (n_missed, n_pos): positives outside the per-period top K, and ALL positives (the denominator
    is every positive in the data, not only those in the reviewed set). Tie handling: see `missed_flags`."""
    m = missed_flags(scores, labels, period_ids, K)
    return int(m.sum()), int(len(m))


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided upper confidence bound, at confidence 1 - delta, on a proportion after k "successes"
    (here: misses) in n trials (here: calibration positives).

    P(true proportion <= bound) >= 1 - delta whenever the n trials are i.i.d. Bernoulli (see the module
    docstring for the validity condition). Edge cases: n = 0 -> 1.0; k = n -> 1.0.
    """
    if n <= 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def certify(scores, labels, period_ids, K: int, delta: float) -> dict:
    """Certify the miss rate at budget K from calibration data.

    Returns n_pos, n_missed, point_miss_rate, upper_bound (Clopper-Pearson, confidence 1 - delta), K, delta.
    Valid for exchangeable deployment positives only (module docstring).
    """
    n_missed, n_pos = missed_counts(scores, labels, period_ids, K)
    return {"n_pos": n_pos, "n_missed": n_missed,
            "point_miss_rate": (n_missed / n_pos) if n_pos else float("nan"),
            "upper_bound": clopper_pearson_upper(n_missed, n_pos, delta), "K": int(K), "delta": float(delta)}
