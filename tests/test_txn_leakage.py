"""Leakage tests for the transaction-level scorer's FEATURES (real data, one period)."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import features as F
from src import transaction_model as TM

PERIOD = 14


@pytest.fixture(scope="module")
def ctx():
    tx, ent, stats, t0, currencies = TM.load_context()
    cur = tx[tx.period == PERIOD]
    ok = pd.MultiIndex.from_arrays([cur.from_key, cur.period]).isin(ent.index) & pd.MultiIndex.from_arrays([cur.to_key, cur.period]).isin(ent.index)
    sample = cur[ok].sample(4000, random_state=0)         # quick mode only has features for sampled accounts
    accts = np.unique(np.r_[sample.from_key, sample.to_key])
    panel = pd.DataFrame({"acct_key": accts, "period": np.int16(PERIOD)})
    return tx, ent, stats, t0, currencies, sample, panel


def _features(sample, tx_hist, panel, stats, t0, currencies):
    ent = F.compute_features(tx_hist, panel, stats).set_index(["acct_key", "period"])
    return TM.txn_features(sample.drop(columns=["is_laundering"], errors="ignore"), ent, stats, t0, currencies)


def test_features_do_not_change_if_later_transactions_are_deleted(ctx):
    tx, ent, stats, t0, currencies, sample, panel = ctx
    full = _features(sample, tx, panel, stats, t0, currencies)
    trunc = _features(sample, tx[tx.period <= PERIOD], panel, stats, t0, currencies)
    pd.testing.assert_frame_equal(full, trunc)
    # and they match the features the model was actually trained/scored with (precomputed entity table)
    stored = TM.txn_features(sample.drop(columns=["is_laundering"]), ent, stats, t0, currencies)
    pd.testing.assert_frame_equal(full, stored, rtol=1e-4, atol=1e-5)


def test_features_do_not_change_if_label_is_shuffled_or_removed(ctx):
    tx, ent, stats, t0, currencies, sample, panel = ctx
    base = _features(sample, tx, panel, stats, t0, currencies)
    shuffled = tx.assign(is_laundering=np.random.default_rng(1).permutation(tx["is_laundering"].to_numpy()))
    removed = tx.drop(columns=["is_laundering"])
    pd.testing.assert_frame_equal(base, _features(sample.assign(is_laundering=1), shuffled, panel, stats, t0, currencies))
    pd.testing.assert_frame_equal(base, _features(sample.drop(columns=["is_laundering"]), removed, panel, stats, t0, currencies))
    assert not any("laundering" in c or "typology" in c for c in base.columns)
