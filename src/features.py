"""Entity-period features (v2): current period t plus a trailing 3-period lookback.

Review of period t happens after it closes, so every transaction in period t is available; nothing after
the end of period t is used. The label (Is Laundering) and typology are never read: `compute_features`
strips label columns from the transaction table on entry.

Amounts are log1p-transformed and standardised within each currency using TRAIN-split statistics only
(`fit_amount_stats`); every amount statistic is computed on the standardised values. `amt_total` sums
exp(z) so that it is positive and a current/lookback ratio is meaningful.
Lookback features are NaN (not zero) when the account had no activity in the lookback window.
"""
import numpy as np
import pandas as pd

from src import config as C
from src import panel as P
from src import split as S

LOOKBACK = 3
_FEATURES = C.PROCESSED / "features.parquet"
_STATS = C.PROCESSED / "amount_stats.parquet"
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
LABEL_COLS = ["is_laundering", "typology", "label"]

BASE_COLS = (["n_tx", "out_deg", "in_deg", "in_out_ratio", "out_cp", "in_cp", "n_counterparties", "cp_diversity",
              "n_currencies", "amt_mean", "amt_std", "amt_max", "amt_skew", "amt_total", "cross_bank_share",
              "burstiness"] + [f"fmt_{c.replace(' ', '_')}" for c in FORMATS])
RATIO_COLS = ["ratio_n_tx", "ratio_out_deg", "ratio_amt_total"]
FEATURE_COLS = [f"cur_{c}" for c in BASE_COLS] + [f"lb_{c}" for c in BASE_COLS] + ["has_history"] + RATIO_COLS


def fit_amount_stats(tx: pd.DataFrame, max_train_period: int) -> pd.DataFrame:
    """Per-currency mean/std of log1p(amount) over train-period transactions (both sides). Index = currency;
    the row named '__global__' is the fallback for currencies unseen in training."""
    tr = tx[tx["period"] <= max_train_period]
    s = pd.concat([
        pd.DataFrame({"cur": tr["pay_currency"].to_numpy(), "x": np.log1p(tr["amount_paid"].to_numpy())}),
        pd.DataFrame({"cur": tr["recv_currency"].to_numpy(), "x": np.log1p(tr["amount_received"].to_numpy())}),
    ])
    st = s.groupby("cur")["x"].agg(["mean", "std"])
    st.loc["__global__"] = [s["x"].mean(), s["x"].std()]
    st["std"] = st["std"].fillna(1.0).where(st["std"] > 1e-9, 1.0)
    return st


def _z(amount: np.ndarray, cur: pd.Series, stats: pd.DataFrame) -> np.ndarray:
    g = stats.loc["__global__"]
    mu = cur.map(stats["mean"]).fillna(g["mean"]).to_numpy()
    sd = cur.map(stats["std"]).fillna(g["std"]).to_numpy()
    return (np.log1p(amount) - mu) / sd


def _long_table(tx: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Two rows per transaction, one from each account's perspective."""
    secs = (tx["ts"] - t0).dt.total_seconds().to_numpy()
    cross = (tx["from_bank"] != tx["to_bank"]).to_numpy()
    fmt = tx["payment_format"].to_numpy()
    out = pd.DataFrame({"acct": tx["from_key"].to_numpy(), "cp": tx["to_key"].to_numpy(), "is_out": 1,
                        "z": tx["z_paid"].to_numpy(), "t": secs, "fmt": fmt, "cross": cross,
                        "cur": tx["pay_currency"].to_numpy()})
    inn = pd.DataFrame({"acct": tx["to_key"].to_numpy(), "cp": tx["from_key"].to_numpy(), "is_out": 0,
                        "z": tx["z_recv"].to_numpy(), "t": secs, "fmt": fmt, "cross": cross,
                        "cur": tx["recv_currency"].to_numpy()})
    return pd.concat([out, inn], ignore_index=True)


def window_features(tx_win: pd.DataFrame, accts: np.ndarray, t0: pd.Timestamp) -> pd.DataFrame:
    """Aggregate one window of transactions into per-account features (index = acct_key, only accounts
    that have at least one transaction in the window)."""
    L = _long_table(tx_win, t0)
    L = L[L["acct"].isin(accts)].sort_values(["acct", "t"], kind="stable")
    g = L.groupby("acct")
    f = pd.DataFrame({"n_tx": g.size(), "out_deg": g["is_out"].sum()})
    f["in_deg"] = f["n_tx"] - f["out_deg"]
    f["in_out_ratio"] = (f["in_deg"] + 1) / (f["out_deg"] + 1)
    cp = L.drop_duplicates(["acct", "cp", "is_out"])
    f["out_cp"] = cp[cp["is_out"] == 1].groupby("acct").size()
    f["in_cp"] = cp[cp["is_out"] == 0].groupby("acct").size()
    f[["out_cp", "in_cp"]] = f[["out_cp", "in_cp"]].fillna(0)
    f["n_counterparties"] = L.drop_duplicates(["acct", "cp"]).groupby("acct").size()
    f["cp_diversity"] = f["n_counterparties"] / f["n_tx"]
    f["n_currencies"] = L.drop_duplicates(["acct", "cur"]).groupby("acct").size()
    L = L.assign(z2=L["z"] ** 2, z3=L["z"] ** 3, ez=np.exp(L["z"]))
    s = L.groupby("acct")[["z", "z2", "z3", "ez"]].sum()
    mu = s["z"] / f["n_tx"]
    ex2, ex3 = s["z2"] / f["n_tx"], s["z3"] / f["n_tx"]
    sd = np.sqrt((ex2 - mu ** 2).clip(lower=0))
    f["amt_mean"] = mu
    f["amt_std"] = sd
    f["amt_max"] = g["z"].max()
    f["amt_skew"] = ((ex3 - 3 * mu * ex2 + 2 * mu ** 3) / sd ** 3).where(sd > 1e-6)
    f["amt_total"] = s["ez"]
    f["cross_bank_share"] = g["cross"].mean()
    pf = L.groupby(["acct", "fmt"]).size().unstack(fill_value=0).reindex(columns=FORMATS, fill_value=0)
    pf.columns = [f"fmt_{c.replace(' ', '_')}" for c in pf.columns]
    f = f.join(pf)
    # burstiness B = (sd - mean) / (sd + mean) of inter-event gaps; needs >= 3 events (>= 2 gaps)
    dt = L.groupby("acct")["t"].diff()
    gaps = pd.DataFrame({"acct": L["acct"].to_numpy(), "dt": dt.to_numpy()}).dropna()
    gs = gaps.groupby("acct")["dt"].agg(["mean", "std", "size"])
    sd_gap = gs["std"] * np.sqrt((gs["size"] - 1) / gs["size"])
    denom = sd_gap + gs["mean"]
    f["burstiness"] = ((sd_gap - gs["mean"]) / denom.where(denom > 0)).where(gs["size"] >= 2)
    return f.reindex(columns=BASE_COLS)


def compute_features(tx: pd.DataFrame, panel: pd.DataFrame, stats: pd.DataFrame,
                     lookback: int = LOOKBACK) -> pd.DataFrame:
    """One row per (acct_key, period) of `panel`. `tx` must carry `period`; label columns are dropped."""
    tx = tx.drop(columns=[c for c in LABEL_COLS if c in tx.columns])
    panel = panel[["acct_key", "period"]]
    t0 = tx["ts"].min().normalize()
    tx = tx.assign(z_paid=_z(tx["amount_paid"].to_numpy(), tx["pay_currency"], stats),
                   z_recv=_z(tx["amount_received"].to_numpy(), tx["recv_currency"], stats))
    parts = []
    for p in sorted(panel["period"].unique()):
        accts = panel.loc[panel["period"] == p, "acct_key"].to_numpy()
        n_lb = min(lookback, int(p))
        cur = window_features(tx[tx["period"] == p], accts, t0).reindex(accts)
        lb_tx = tx[(tx["period"] >= p - lookback) & (tx["period"] < p)]
        lb = window_features(lb_tx, accts, t0).reindex(accts) if len(lb_tx) else \
            pd.DataFrame(np.nan, index=accts, columns=BASE_COLS)
        f = pd.concat([cur.add_prefix("cur_"), lb.add_prefix("lb_")], axis=1)
        f["has_history"] = lb["n_tx"].notna().astype(float).to_numpy()
        for name, col in (("ratio_n_tx", "n_tx"), ("ratio_out_deg", "out_deg"), ("ratio_amt_total", "amt_total")):
            avg = lb[col] / n_lb if n_lb > 0 else np.nan
            f[name] = (cur[col] / avg.where(avg > 0)) if n_lb > 0 else np.nan
        f.insert(0, "period", np.int16(p))
        f.index = pd.Index(accts, name="acct_key")
        parts.append(f.reset_index())
    out = pd.concat(parts, ignore_index=True)
    out[FEATURE_COLS] = out[FEATURE_COLS].astype("float32")
    return out[["acct_key", "period"] + FEATURE_COLS]


def load_features() -> pd.DataFrame:
    return pd.read_parquet(_FEATURES)


if __name__ == "__main__":
    import sys
    import time
    tx = P.active_transactions()
    pan = P.build_panel()[["acct_key", "period"]]
    a, _ = S.period_bounds(int(pan["period"].max()) + 1)
    stats = fit_amount_stats(tx, a - 1)           # train periods only, fitted on the full table
    stats.to_parquet(_STATS)
    print(f"amount stats fitted on periods <= {a - 1}: {len(stats) - 1} currencies")
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    if frac < 1.0:
        rng = np.random.default_rng(C.SEED)
        keys = pan["acct_key"].unique()
        keep = rng.choice(keys, int(frac * len(keys)), replace=False)
        pan = pan[pan["acct_key"].isin(keep)]
        tx = tx[tx["from_key"].isin(keep) | tx["to_key"].isin(keep)]
        print(f"sample: {len(keep):,} accounts, {len(pan):,} entity-periods, {len(tx):,} tx")
    t = time.time()
    feats = compute_features(tx, pan, stats)
    print(f"computed {feats.shape} in {time.time() - t:.0f}s")
    dest = _FEATURES if frac >= 1.0 else C.PROCESSED / f"features_sample{int(frac * 100)}.parquet"
    feats.to_parquet(dest)
    print("saved", dest)
    print(feats[FEATURE_COLS].describe().T[["count", "mean", "min", "max"]].to_string())
