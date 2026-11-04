"""Transaction-level LightGBM scorer for H3 (aggregation variant). Trained on TRAIN-split transactions only.

Features of a transaction in period t:
  own attributes: currency-standardised log amounts (train-only statistics), payment format, currencies,
    cross-bank flag, same-currency flag, position within the period;
  both endpoints' LOOKBACK-ONLY entity features (periods t-3 .. t-1, never the current period), prefixed s_ / r_.
The label (Is Laundering) is the target only; it is never an input.

Training sample: ALL train positives + a random 10% of train negatives (seeded). This is roughly a 1:120
positive:negative ratio instead of ~1:1,200; the model is used only for ranking (margin), so the resulting
probability shift does not matter. One fixed hyperparameter set, no search.
"""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src import features as F
from src import panel as P
from src import split as S

MODEL_PATH = C.RESULTS / "lgbm_txn_frozen.joblib"
SCORES_PATH = C.PROCESSED / "txn_scores.parquet"
NEG_FRAC = 0.10
LB_COLS = [f"lb_{c}" for c in F.BASE_COLS] + ["has_history"]
PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=C.SEED, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)


def _codes(series: pd.Series, categories) -> np.ndarray:
    return pd.Categorical(series, categories=categories).codes.astype("int16")


def txn_features(tx: pd.DataFrame, ent: pd.DataFrame, stats: pd.DataFrame, t0: pd.Timestamp,
                 currencies: list) -> pd.DataFrame:
    """Feature matrix for transactions `tx` (must carry `period`; label columns are not read).
    `ent` = entity-period feature table indexed by (acct_key, period); only its lookback columns are used."""
    own = pd.DataFrame({
        "z_paid": F._z(tx["amount_paid"].to_numpy(), tx["pay_currency"], stats),
        "z_recv": F._z(tx["amount_received"].to_numpy(), tx["recv_currency"], stats),
        "fmt": _codes(tx["payment_format"], F.FORMATS),
        "pay_cur": _codes(tx["pay_currency"], currencies),
        "recv_cur": _codes(tx["recv_currency"], currencies),
        "cross_bank": (tx["from_bank"] != tx["to_bank"]).astype("int8").to_numpy(),
        "same_cur": (tx["pay_currency"] == tx["recv_currency"]).astype("int8").to_numpy(),
        "pos_in_period": (((tx["ts"] - t0).dt.total_seconds() / 3600.0) % C.PERIOD_HOURS / C.PERIOD_HOURS).to_numpy(),
    }, index=tx.index)
    lb = ent[LB_COLS]
    per = tx["period"].to_numpy()
    s = lb.reindex(pd.MultiIndex.from_arrays([tx["from_key"].to_numpy(), per])).add_prefix("s_")
    r = lb.reindex(pd.MultiIndex.from_arrays([tx["to_key"].to_numpy(), per])).add_prefix("r_")
    s.index = r.index = own.index
    return pd.concat([own, s, r], axis=1).astype("float32")


def load_context():
    tx = P.active_transactions()
    ent = F.load_features().set_index(["acct_key", "period"])
    stats = pd.read_parquet(C.PROCESSED / "amount_stats.parquet")
    currencies = sorted(set(tx["pay_currency"]) | set(tx["recv_currency"]))
    return tx, ent, stats, tx["ts"].min().normalize(), currencies


def txn_margin_max_by_entity(tx_scores: pd.DataFrame) -> pd.DataFrame:
    """Maximum transaction score over all transactions where the entity is sender or receiver in the period.
    `tx_scores` columns: from_key, to_key, period, txn_margin. Returns acct_key, period, max_txn_margin."""
    long = pd.concat([
        tx_scores[["from_key", "period", "txn_margin"]].rename(columns={"from_key": "acct_key"}),
        tx_scores[["to_key", "period", "txn_margin"]].rename(columns={"to_key": "acct_key"})])
    return (long.groupby(["acct_key", "period"], as_index=False)["txn_margin"].max()
            .rename(columns={"txn_margin": "max_txn_margin"}))


if __name__ == "__main__":
    tx, ent, stats, t0, currencies = load_context()
    a, b = S.period_bounds(int(tx["period"].max()) + 1)
    train = tx[tx["period"] < a]
    rng = np.random.default_rng(C.SEED)
    keep = (train["is_laundering"].to_numpy() == 1) | (rng.random(len(train)) < NEG_FRAC)
    sample = train[keep]
    y = sample["is_laundering"].to_numpy()
    print(f"train tx {len(train):,}; sample {len(sample):,} = {int(y.sum()):,} positives + {int((y == 0).sum()):,} "
          f"negatives ({NEG_FRAC:.0%} of train negatives; ratio 1:{(y == 0).sum() / y.sum():.0f})")
    Xtr = txn_features(sample.drop(columns=["is_laundering"]), ent, stats, t0, currencies)
    model = lgb.LGBMClassifier(**PARAMS).fit(Xtr, y)
    joblib.dump(model, MODEL_PATH)
    parts = []
    for lo in range(a, int(tx["period"].max()) + 1, 2):                # score calibration+test tx in 2-period chunks
        ch = tx[(tx["period"] >= lo) & (tx["period"] < lo + 2)]
        X = txn_features(ch.drop(columns=["is_laundering"]), ent, stats, t0, currencies)
        parts.append(pd.DataFrame({"tx_id": ch["tx_id"].to_numpy(), "from_key": ch["from_key"].to_numpy(),
                                   "to_key": ch["to_key"].to_numpy(), "period": ch["period"].to_numpy(),
                                   "is_laundering": ch["is_laundering"].to_numpy(),
                                   "txn_margin": model.predict(X, raw_score=True)}))
    sc = pd.concat(parts, ignore_index=True)
    sc.to_parquet(SCORES_PATH)
    for name, m in (("calibration", (sc.period >= a) & (sc.period < b)), ("test", sc.period >= b)):
        d = sc[m]
        print(f"txn model {name}: rows {len(d):,}, positives {int(d.is_laundering.sum()):,}, "
              f"ROC-AUC {roc_auc_score(d.is_laundering, d.txn_margin):.4f}, "
              f"PR-AUC {average_precision_score(d.is_laundering, d.txn_margin):.4f} "
              f"(base rate {d.is_laundering.mean():.4%})")
    imp = pd.Series(model.booster_.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    print("top features (gain share):", (imp / imp.sum()).round(3).head(8).to_dict())
    print("\n*** TXN MODEL FROZEN - do not retune ***")
