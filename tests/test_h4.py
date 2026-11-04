import numpy as np

from src import certify as CE
from src.h4_elliptic import POOL_START, recovery_lag, run_arm


def _counts(k_by_step, n=50):
    return {s: (k_by_step.get(s, 10), n, 1000) for s in range(25, 50)}


def test_static_pool_is_steps_25_to_34_truncated_by_delay():
    counts = _counts({})
    rows = run_arm("static", 2, 0.1, counts)
    r35, r40 = rows[0], rows[5]
    assert r35["pool_positives"] == 50 * 8               # steps 25..32 only (labels of 33 and 34 have not arrived at t=35, L=2)
    assert r40["pool_positives"] == 50 * 10              # full 25..34 once available
    assert r40["bound"] == CE.clopper_pearson_upper(100, 500, 0.10)


def test_rolling_pool_is_the_five_most_recent_available_steps():
    rows = run_arm("rolling", 2, 0.1, _counts({}))
    r = [x for x in rows if x["step"] == 45][0]
    assert r["pool_positives"] == 50 * 5                 # steps 38..42
    assert all(x["delta_t"] == 0.10 for x in rows)


def test_aci_delta_moves_down_after_errors_and_up_after_none():
    # misses explode from step 40 on -> errors -> delta_t must fall below 0.10 (wider bound); delta never leaves [1e-4, 0.5]
    counts = _counts({s: 50 for s in range(40, 50)})
    rows = run_arm("aci", 0, 0.1, counts, 0.05)
    d = [x["delta_t"] for x in rows]
    assert min(d) < 0.10 and d[-1] < d[0] and all(1e-4 <= v <= 0.5 for v in d)
    quiet = run_arm("aci", 0, 0.1, _counts({s: 0 for s in range(25, 50)}), 0.05)
    assert quiet[-1]["delta_t"] > 0.10 and quiet[-1]["delta_t"] <= 0.5
    assert quiet[0]["delta_t"] == 0.10 and quiet[1]["delta_t"] == 0.10     # no update before an issued bound exists


def test_recovery_lag():
    errs = {s: 0 for s in range(43, 50)}
    assert recovery_lag(errs) == 0
    errs.update({43: 1, 44: 1})
    assert recovery_lag(errs) == 2
    assert recovery_lag({s: 1 for s in range(43, 50)}) is None
