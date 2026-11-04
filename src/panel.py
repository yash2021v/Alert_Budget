"""Entity-period panel: one row per account that sends/receives >=1 transaction in a period."""
import numpy as np
import pandas as pd

from src import config as C
from src import data

_PANEL = C.PROCESSED / "panel.parquet"


def active_transactions(trans: pd.DataFrame | None = None) -> pd.DataFrame:
    """Transactions in the regular-volume window, with integer `period` (0 = first period)."""
    if trans is None:
        trans = data.load_transactions()
    tx = trans[trans["ts"] < pd.Timestamp(C.ACTIVE_END)].copy()
    t0 = tx["ts"].min().normalize()
    tx["period"] = ((tx["ts"] - t0).dt.total_seconds() // (3600 * C.PERIOD_HOURS)).astype("int16")
    return tx


def _entity_rows(tx: pd.DataFrame) -> pd.DataFrame:
    s = tx[["from_key", "period", "tx_id", "is_laundering"]].rename(columns={"from_key": "acct_key"})
    r = tx[["to_key", "period", "tx_id", "is_laundering"]].rename(columns={"to_key": "acct_key"})
    return pd.concat([s, r], ignore_index=True)


def assign_typology(ent: pd.DataFrame, tx_period: pd.DataFrame) -> pd.Series:
    """Typology per positive entity-period (index = ent index), for grouping only.

    Attempts considered = pattern attempts with a transaction involving the account in that period.
    Most frequent typology among them (counting attempts); ties -> typology of the earliest attempt
    (earliest first-transaction timestamp). Positives with no pattern -> 'UNATTRIBUTED'.
    """
    pat = data.load_patterns().dropna(subset=["tx_id"])
    pat["tx_id"] = pat["tx_id"].astype("int32")
    first_ts = pat.groupby("attempt_id")["ts"].min().rename("attempt_start")
    link = pat[["tx_id", "attempt_id", "typology"]].drop_duplicates(["tx_id", "attempt_id"])
    link = link.merge(first_ts, left_on="attempt_id", right_index=True)
    rows = _entity_rows(tx_period)[["acct_key", "period", "tx_id"]].merge(link, on="tx_id")
    per = rows.drop_duplicates(["acct_key", "period", "attempt_id"])
    cnt = per.groupby(["acct_key", "period", "typology"]).agg(n=("attempt_id", "size"),
                                                              first=("attempt_start", "min")).reset_index()
    cnt = cnt.sort_values(["acct_key", "period", "n", "first"], ascending=[True, True, False, True])
    best = cnt.drop_duplicates(["acct_key", "period"])[["acct_key", "period", "typology"]]
    out = ent[["acct_key", "period"]].merge(best, on=["acct_key", "period"], how="left")["typology"]
    out.index = ent.index
    return out


def build_panel(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and _PANEL.exists():
        return pd.read_parquet(_PANEL)
    tx = active_transactions()
    rows = _entity_rows(tx)
    ent = rows.groupby(["acct_key", "period"], sort=True)["is_laundering"].max().rename("label").reset_index()
    ent["label"] = ent["label"].astype("int8")
    ent["period"] = ent["period"].astype("int16")
    if C.SAMPLE_FRAC < 1.0:                       # quick mode: random sample of accounts (seeded); their features stay exact
        keys = np.sort(ent["acct_key"].unique())
        keep = np.random.default_rng(C.SEED).choice(keys, int(C.SAMPLE_FRAC * len(keys)), replace=False)
        ent = ent[ent["acct_key"].isin(keep)].reset_index(drop=True)
    ent["typology"] = pd.Series(pd.NA, index=ent.index, dtype="object")
    pos = ent["label"] == 1
    typ = assign_typology(ent[pos], tx)
    ent.loc[pos, "typology"] = typ.fillna("UNATTRIBUTED")
    t0 = tx["ts"].min().normalize()
    ent["period_start"] = t0 + pd.to_timedelta(ent["period"].astype("int64") * C.PERIOD_HOURS, unit="h")
    ent.to_parquet(_PANEL)
    return ent


if __name__ == "__main__":
    trans = data.load_transactions()
    tx = active_transactions(trans)
    print(f"window: {tx.ts.min()} -> {tx.ts.max()}  ({tx.ts.dt.normalize().nunique()} days) "
          f"{len(tx):,} tx (of {len(trans):,}); dropped {len(trans) - len(tx):,} tail tx "
          f"incl. {int(trans.is_laundering.sum() - tx.is_laundering.sum())} laundering rows")
    p = build_panel(use_cache=False)
    print(f"period length: {C.PERIOD_HOURS}h -> {p.period.nunique()} periods")
    print(f"entity-periods: {len(p):,}; positives {int(p.label.sum()):,} ({p.label.mean():.4%})")
    g = p.groupby("period").agg(rows=("label", "size"), pos=("label", "sum"))
    g["rate"] = (g.pos / g.rows).round(5)
    print(g.T.to_string())
    pos = p[p.label == 1]
    print(f"UNATTRIBUTED share of positives: {(pos.typology == 'UNATTRIBUTED').mean():.2%}")
    print(pos.typology.value_counts().to_string())
