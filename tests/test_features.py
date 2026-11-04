import numpy as np
import pandas as pd
import pytest

from src import features as F

STATS = pd.DataFrame({"mean": [0.0, 0.0], "std": [1.0, 1.0]}, index=["USD", "__global__"])


def _tx(rows):
    """rows: (from, to, period, minute_in_period, amount, fmt, from_bank, to_bank, is_laundering)"""
    df = pd.DataFrame(rows, columns=["from_key", "to_key", "period", "minute", "amount", "payment_format",
                                     "from_bank", "to_bank", "is_laundering"])
    df["ts"] = pd.Timestamp("2022-09-01") + pd.to_timedelta(df["period"] * 12 * 60 + df["minute"], unit="m")
    df["amount_paid"] = df["amount"]
    df["amount_received"] = df["amount"]
    df["pay_currency"] = "USD"
    df["recv_currency"] = "USD"
    df["period"] = df["period"].astype("int16")
    return df.drop(columns=["minute", "amount"])


def _tiny():
    # hand-made graph. periods 0-1 are lookback for target period 2; period 2 is current; period 3 is the future
    return _tx([
        (0, 1, 0, 0, 10.0, "ACH", 1, 1, 0),
        (0, 1, 0, 10, 20.0, "ACH", 1, 1, 0),
        (0, 1, 1, 0, 30.0, "Wire", 1, 1, 0),
        (0, 2, 1, 40, 100.0, "ACH", 1, 2, 1),
        (3, 0, 1, 50, 5.0, "Cash", 3, 1, 0),
        (0, 1, 2, 0, 50.0, "ACH", 1, 1, 0),      # current period
        (4, 0, 2, 30, 7.0, "Wire", 4, 1, 1),     # current period
        (0, 1, 3, 0, 999.0, "ACH", 1, 1, 1),     # future: must never be used for period 2
    ])


def _panel(accts, p):
    return pd.DataFrame({"acct_key": accts, "period": np.int16(p)})


def test_hand_computed_features():
    f = F.compute_features(_tiny(), _panel([0, 1, 7], 2), STATS).set_index("acct_key")
    a = f.loc[0]
    # current period: 0 sends 50 to 1, receives 7 from 4
    assert (a.cur_n_tx, a.cur_out_deg, a.cur_in_deg) == (2, 1, 1)
    assert (a.cur_out_cp, a.cur_in_cp, a.cur_n_counterparties) == (1, 1, 2)
    assert np.isclose(a.cur_amt_mean, (np.log1p(50) + np.log1p(7)) / 2, rtol=1e-5)
    assert np.isclose(a.cur_amt_max, np.log1p(50), rtol=1e-5)
    assert np.isclose(a.cur_amt_total, 51 + 8, rtol=1e-5)           # exp(log1p(x)) = 1 + x
    assert np.isclose(a.cur_cross_bank_share, 1 / 2) and a.cur_n_currencies == 1
    assert a.cur_fmt_ACH == 1 and a.cur_fmt_Wire == 1
    # lookback (periods 0-1): out to 1 (x3) and 2, in from 3
    assert (a.lb_n_tx, a.lb_out_deg, a.lb_in_deg, a.lb_n_counterparties) == (5, 4, 1, 3)
    assert np.isclose(a.lb_cp_diversity, 3 / 5)
    assert np.isclose(a.lb_cross_bank_share, 2 / 5)
    assert a.lb_fmt_ACH == 3 and a.lb_fmt_Wire == 1 and a.lb_fmt_Cash == 1
    assert a.has_history == 1
    assert np.isclose(a.ratio_n_tx, 2 / (5 / 2))                     # lookback has 2 periods available at p=2
    assert np.isclose(a.ratio_out_deg, 1 / (4 / 2))
    # account 1: only a receiver in the lookback; no lookback activity as a sender
    assert f.loc[1].lb_in_deg == 3 and f.loc[1].lb_out_deg == 0
    # account 7: no activity at all -> lookback NaN (not zero), has_history 0
    assert f.loc[7].has_history == 0 and np.isnan(f.loc[7].lb_n_tx) and np.isnan(f.loc[7].ratio_n_tx)


def test_no_future_leakage():
    tx = _tiny()
    p = 2
    full = F.compute_features(tx, _panel([0, 1, 4], p), STATS)
    truncated = F.compute_features(tx[tx["period"] <= p], _panel([0, 1, 4], p), STATS)
    pd.testing.assert_frame_equal(full, truncated)
    mutated = tx.copy()
    mutated.loc[mutated["period"] > p, ["amount_paid", "amount_received"]] *= 1000
    pd.testing.assert_frame_equal(full, F.compute_features(mutated, _panel([0, 1, 4], p), STATS))


def test_label_column_is_ignored():
    tx = _tiny()
    base = F.compute_features(tx, _panel([0, 1, 4], 2), STATS)
    shuffled = tx.copy()
    shuffled["is_laundering"] = np.random.default_rng(0).permutation(shuffled["is_laundering"].to_numpy())
    flipped = tx.assign(is_laundering=1 - tx["is_laundering"])
    pd.testing.assert_frame_equal(base, F.compute_features(shuffled, _panel([0, 1, 4], 2), STATS))
    pd.testing.assert_frame_equal(base, F.compute_features(flipped, _panel([0, 1, 4], 2), STATS))
    pd.testing.assert_frame_equal(base, F.compute_features(tx.drop(columns="is_laundering"), _panel([0, 1, 4], 2), STATS))
    assert not any(c in F.FEATURE_COLS for c in F.LABEL_COLS)


def test_lookback_window_excludes_old_periods():
    tx = _tx([(0, 1, 0, 0, 1.0, "ACH", 1, 1, 0), (0, 1, 5, 0, 1.0, "ACH", 1, 1, 0)])
    f = F.compute_features(tx, _panel([0], 5), STATS).set_index("acct_key")
    assert np.isnan(f.loc[0].lb_n_tx) and f.loc[0].cur_n_tx == 1       # period-0 tx is outside the lookback
    f = F.compute_features(tx, _panel([0], 3), STATS).set_index("acct_key")
    assert f.loc[0].lb_n_tx == 1


def test_burstiness_regular_vs_bursty():
    reg = _tx([(0, 1, 1, m, 1.0, "ACH", 1, 1, 0) for m in (0, 10, 20, 30, 40)])
    f = F.compute_features(reg, _panel([0], 1), STATS).set_index("acct_key")
    assert np.isclose(f.loc[0].cur_burstiness, -1.0)                  # perfectly periodic


def test_amount_stats_use_train_periods_only():
    tx = _tx([(0, 1, 0, 0, 10.0, "ACH", 1, 1, 0), (0, 1, 1, 0, 1e9, "ACH", 1, 1, 0)])
    st = F.fit_amount_stats(tx, 0)
    assert np.isclose(st.loc["USD", "mean"], np.log1p(10.0))
