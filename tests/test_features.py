import numpy as np
import pandas as pd

from src import features as F


def _tx(rows):
    """rows: (from, to, period, minute_in_period, amount, fmt, from_bank, to_bank)"""
    df = pd.DataFrame(rows, columns=["from_key", "to_key", "period", "minute", "amount", "payment_format",
                                     "from_bank", "to_bank"])
    df["ts"] = pd.Timestamp("2022-09-01") + pd.to_timedelta(df["period"] * 12 * 60 + df["minute"], unit="m")
    df["amount_paid"] = df["amount"]
    df["amount_received"] = df["amount"]
    df["period"] = df["period"].astype("int16")
    return df


def _tiny():
    # hand-made graph: 0 -> 1 (x3), 0 -> 2, 3 -> 0 ; all in periods 0-1; period 2 is the target
    return _tx([
        (0, 1, 0, 0, 10.0, "ACH", 1, 1),
        (0, 1, 0, 10, 20.0, "ACH", 1, 1),
        (0, 1, 1, 0, 30.0, "Wire", 1, 1),
        (0, 2, 1, 40, 100.0, "ACH", 1, 2),
        (3, 0, 1, 50, 5.0, "Cash", 3, 1),
        (0, 1, 2, 0, 999.0, "ACH", 1, 1),     # target-period tx (must never be used)
    ])


def _panel(accts, p):
    return pd.DataFrame({"acct_key": accts, "period": np.int16(p)})


def test_hand_computed_features():
    tx = _tiny()
    f = F.compute_features(tx, _panel([0, 1, 9], 2)).set_index("acct_key")
    a = f.loc[0]
    assert (a.out_deg, a.in_deg, a.n_tx) == (4, 1, 5)
    assert a.n_counterparties == 3                       # {1, 2, 3}
    assert np.isclose(a.cp_diversity, 3 / 5)
    assert np.isclose(a.in_out_ratio, 2 / 5)
    assert np.isclose(a.amt_mean, (10 + 20 + 30 + 100 + 5) / 5)
    assert a.amt_max == 100
    assert np.isclose(a.cross_bank_share, 2 / 5)         # 0->2 and 3->0 cross banks
    assert a.fmt_ACH == 3 and a.fmt_Wire == 1 and a.fmt_Cash == 1
    assert f.loc[1].in_deg == 3 and f.loc[1].out_deg == 0
    assert f.loc[9].n_tx == 0 and np.isnan(f.loc[9].amt_mean)   # no history -> zero counts, NaN stats


def test_no_future_leakage():
    tx = _tiny()
    p = 2
    full = F.compute_features(tx, _panel([0, 1, 2, 3], p))
    truncated = F.compute_features(tx[tx["period"] < p], _panel([0, 1, 2, 3], p))
    pd.testing.assert_frame_equal(full, truncated)
    # and mutating the future must not change anything either
    mutated = tx.copy()
    mutated.loc[mutated["period"] >= p, ["amount_paid", "amount_received"]] *= 1000
    pd.testing.assert_frame_equal(full, F.compute_features(mutated, _panel([0, 1, 2, 3], p)))


def test_lookback_window_excludes_old_periods():
    tx = _tx([(0, 1, 0, 0, 1.0, "ACH", 1, 1), (0, 1, 5, 0, 1.0, "ACH", 1, 1)])
    f = F.compute_features(tx, _panel([0], 5)).set_index("acct_key")
    assert f.loc[0].n_tx == 0                             # period-0 tx is outside the 3-period lookback
    f = F.compute_features(tx, _panel([0], 3)).set_index("acct_key")
    assert f.loc[0].n_tx == 1


def test_burstiness_regular_vs_bursty():
    reg = _tx([(0, 1, 0, m, 1.0, "ACH", 1, 1) for m in (0, 10, 20, 30, 40)])
    f = F.compute_features(reg, _panel([0], 1)).set_index("acct_key")
    assert np.isclose(f.loc[0].burstiness, -1.0)          # perfectly periodic
