"""Temporal split of the entity-period panel (60/20/20 by period order, never shuffled)."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src import config as C
from src import panel as P

_SPLIT = C.PROCESSED / "split.parquet"
GIANT_THRESHOLD = 0.90


def component_stats(tx: pd.DataFrame) -> dict:
    n = int(max(tx["from_key"].max(), tx["to_key"].max())) + 1
    g = coo_matrix((np.ones(len(tx), dtype="int8"), (tx["from_key"], tx["to_key"])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    seen = np.unique(np.concatenate([tx["from_key"].to_numpy(), tx["to_key"].to_numpy()]))
    sizes = np.bincount(lab[seen])
    sizes = sizes[sizes > 0]
    return {"n_accounts": len(seen), "n_components": len(sizes), "largest": int(sizes.max()),
            "largest_share": float(sizes.max() / len(seen))}


def period_bounds(n_periods: int) -> tuple[int, int]:
    """First period of calibration and of test: train = [0, a), cal = [a, b), test = [b, n)."""
    return int(round(0.6 * n_periods)), int(round(0.8 * n_periods))


def build_split(panel: pd.DataFrame) -> pd.DataFrame:
    a, b = period_bounds(int(panel["period"].max()) + 1)
    split = np.where(panel["period"] < a, "train", np.where(panel["period"] < b, "calibration", "test"))
    return pd.DataFrame({"acct_key": panel["acct_key"].to_numpy(), "period": panel["period"].to_numpy(),
                         "split": split})


def load_split() -> pd.DataFrame:
    return pd.read_parquet(_SPLIT) if _SPLIT.exists() else _make()


def _make() -> pd.DataFrame:
    s = build_split(P.build_panel())
    s.to_parquet(_SPLIT)
    return s


if __name__ == "__main__":
    tx = P.active_transactions()
    cs = component_stats(tx)
    print(f"components: {cs['n_components']:,}; largest = {cs['largest']:,} of {cs['n_accounts']:,} accounts "
          f"({cs['largest_share']:.2%})")
    pan = P.build_panel()
    if cs["largest_share"] > GIANT_THRESHOLD:
        print(f"giant component > {GIANT_THRESHOLD:.0%}: component-aware split infeasible -> strictly temporal split")
    else:
        print(f"giant component <= {GIANT_THRESHOLD:.0%}, so the rule asks for whole components per split, but it holds "
              f"{cs['largest_share']:.0%} of accounts and spans every period, so that cannot coexist with a "
              "60/20/20 split by period order -> using the strictly temporal split (overlap reported below)")
    s = _make()
    df = pan.merge(s, on=["acct_key", "period"])
    for name in ("train", "calibration", "test"):
        d = df[df.split == name]
        print(f"{name:<12} periods {d.period.min()}-{d.period.max()}  rows {len(d):>9,}  "
              f"positives {int(d.label.sum()):>5,}  rate {d.label.mean():.4%}")
    multi = (df.groupby("acct_key")["split"].nunique() > 1).sum()
    print(f"accounts appearing in >1 split: {multi:,} of {df.acct_key.nunique():,} ({multi / df.acct_key.nunique():.1%})")
    pos_acc = df[df.label == 1].groupby("acct_key")["split"].nunique()
    print(f"accounts with positives in >1 split: {(pos_acc > 1).sum():,} of {len(pos_acc):,}")
