"""Leakage-safe entity-period features.

Features for (account a, period p) use ONLY transactions with period in [p-LOOKBACK, p-1]
(strictly before period p starts). Accounts with no history get zero counts and NaN statistics.
"""
import numpy as np
import pandas as pd

from src import config as C
from src import panel as P

LOOKBACK = 3
_FEATURES = C.PROCESSED / "features.parquet"
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]


FEATURE_COLS = ["n_tx", "out_deg", "in_deg", "in_out_ratio", "n_counterparties", "cp_diversity", "amt_mean",
                "amt_std", "amt_max", "amt_skew", "cross_bank_share", "burstiness"] + \
               [f"fmt_{c.replace(' ', '_')}" for c in FORMATS]


def _long_table(tx: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Two rows per transaction, one from each account's perspective."""
    secs = (tx["ts"] - t0).dt.total_seconds().to_numpy()
    cross = (tx["from_bank"] != tx["to_bank"]).to_numpy()
    fmt = tx["payment_format"].to_numpy()
    out = pd.DataFrame({"acct": tx["from_key"].to_numpy(), "cp": tx["to_key"].to_numpy(), "is_out": 1,
                        "amt": tx["amount_paid"].to_numpy(), "t": secs, "fmt": fmt, "cross": cross})
    inn = pd.DataFrame({"acct": tx["to_key"].to_numpy(), "cp": tx["from_key"].to_numpy(), "is_out": 0,
                        "amt": tx["amount_received"].to_numpy(), "t": secs, "fmt": fmt, "cross": cross})
    return pd.concat([out, inn], ignore_index=True)


def window_features(tx_win: pd.DataFrame, accts: np.ndarray, t0: pd.Timestamp) -> pd.DataFrame:
    """Aggregate a lookback window of transactions into per-account features (index = acct_key)."""
    L = _long_table(tx_win, t0)
    L = L[L["acct"].isin(accts)].sort_values(["acct", "t"], kind="stable")
    g = L.groupby("acct")
    f = pd.DataFrame({"n_tx": g.size(), "out_deg": g["is_out"].sum()})
    f["in_deg"] = f["n_tx"] - f["out_deg"]
    f["in_out_ratio"] = (f["in_deg"] + 1) / (f["out_deg"] + 1)
    f["n_counterparties"] = L.drop_duplicates(["acct", "cp"]).groupby("acct").size()
    f["cp_diversity"] = f["n_counterparties"] / f["n_tx"]
    L = L.assign(amt2=L["amt"] ** 2, amt3=L["amt"] ** 3)
    s = L.groupby("acct")[["amt", "amt2", "amt3"]].sum()
    mu = s["amt"] / f["n_tx"]
    ex2, ex3 = s["amt2"] / f["n_tx"], s["amt3"] / f["n_tx"]
    var = (ex2 - mu ** 2).clip(lower=0)
    sd = np.sqrt(var)
    f["amt_mean"] = mu
    f["amt_std"] = sd
    f["amt_max"] = g["amt"].max()
    skew = (ex3 - 3 * mu * ex2 + 2 * mu ** 3) / sd ** 3
    f["amt_skew"] = skew.where(sd > 0)
    f["cross_bank_share"] = g["cross"].mean()
    pf = L.groupby(["acct", "fmt"]).size().unstack(fill_value=0).reindex(columns=FORMATS, fill_value=0)
    pf.columns = [f"fmt_{c.replace(' ', '_')}" for c in pf.columns]
    f = f.join(pf)
    # burstiness B = (sd - mean) / (sd + mean) of inter-event gaps; needs >= 3 events (>= 2 gaps)
    dt = L.groupby("acct")["t"].diff()
    gaps = pd.DataFrame({"acct": L["acct"].to_numpy(), "dt": dt.to_numpy()}).dropna()
    gs = gaps.groupby("acct")["dt"].agg(["mean", "std", "size"])
    sd_gap = gs["std"] * np.sqrt((gs["size"] - 1) / gs["size"])           # population std
    denom = sd_gap + gs["mean"]
    f["burstiness"] = ((sd_gap - gs["mean"]) / denom.where(denom > 0)).where(gs["size"] >= 2)
    return f


def compute_features(tx: pd.DataFrame, panel: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Feature table with one row per (acct_key, period) of `panel`. `tx` must carry `period`."""
    t0 = tx["ts"].min().normalize()
    parts = []
    for p in sorted(panel["period"].unique()):
        accts = panel.loc[panel["period"] == p, "acct_key"].to_numpy()
        win = tx[(tx["period"] >= p - lookback) & (tx["period"] < p)]
        if len(win):
            f = window_features(win, accts, t0)
        else:
            f = pd.DataFrame(index=pd.Index([], name="acct"))
        f = f.reindex(index=accts, columns=FEATURE_COLS)
        f.index.name = "acct_key"
        f.insert(0, "period", np.int16(p))
        parts.append(f.reset_index())
    out = pd.concat(parts, ignore_index=True)
    count_cols = ["n_tx", "out_deg", "in_deg", "n_counterparties"] + [c for c in out if c.startswith("fmt_")]
    out[count_cols] = out[count_cols].fillna(0)
    feat_cols = [c for c in out if c not in ("acct_key", "period")]
    out[feat_cols] = out[feat_cols].astype("float32")
    return out


def load_features() -> pd.DataFrame:
    return pd.read_parquet(_FEATURES)


if __name__ == "__main__":
    import sys
    import time
    tx = P.active_transactions()
    pan = P.build_panel()
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    if frac < 1.0:
        rng = np.random.default_rng(C.SEED)
        keys = pan["acct_key"].unique()
        keep = rng.choice(keys, int(frac * len(keys)), replace=False)
        pan = pan[pan["acct_key"].isin(keep)]
        tx = tx[tx["from_key"].isin(keep) | tx["to_key"].isin(keep)]
        print(f"sample: {len(keep):,} accounts, {len(pan):,} entity-periods, {len(tx):,} tx")
    t = time.time()
    feats = compute_features(tx, pan)
    print(f"computed {feats.shape} in {time.time() - t:.0f}s")
    dest = _FEATURES if frac >= 1.0 else C.PROCESSED / f"features_sample{int(frac * 100)}.parquet"
    feats.to_parquet(dest)
    print("saved", dest)
    print(feats[FEATURE_COLS].describe().T[["count", "mean", "min", "max"]].to_string())
