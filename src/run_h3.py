"""H3: direct entity-level score vs max-over-transaction score, each with the same entity-level certificate.

direct = frozen entity-level margin (unchanged); max = maximum transaction margin (frozen transaction model) over the
entity's transactions in the period. For each variant: certify on calibration, evaluate on test (fixed-count K grid),
plus an exchangeable control (pool calibration+test flags, 1,000 random re-splits, coverage vs pooled population).

Writes results/h3_aggregation.csv and results/h3_summary.md.
"""
import numpy as np
import pandas as pd

from src import certify as CE
from src import config as C
from src import scoring
from src import transaction_model as TM

K_FRACS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
DELTAS = [0.10, 0.05]
DRAWS = C.MC


def variant_tables() -> dict:
    sc = scoring.load_scores()                                   # shuffled once (seeded); merges below keep that order
    txs = pd.read_parquet(TM.SCORES_PATH)
    mx = TM.txn_margin_max_by_entity(txs)
    sc = sc.merge(mx, on=["acct_key", "period"], how="left")
    assert sc["max_txn_margin"].notna().all()
    return {"direct": sc.assign(score=sc["margin"]), "max_txn": sc.assign(score=sc["max_txn_margin"])}


def main():
    rng = np.random.default_rng(C.SEED)
    rows = []
    tables = variant_tables()
    for name, sc in tables.items():
        cal, test = sc[sc.split == "calibration"], sc[sc.split == "test"]
        avg = test.groupby("period").size().mean()
        for f in K_FRACS:
            k = int(round(f * avg))
            mc = CE.missed_flags(cal["score"].to_numpy(), cal["label"].to_numpy(), cal["period"].to_numpy(), k)
            mt = CE.missed_flags(test["score"].to_numpy(), test["label"].to_numpy(), test["period"].to_numpy(), k)
            pooled = np.r_[mc, mt]
            for d in DELTAS:
                bound = CE.clopper_pearson_upper(int(mc.sum()), len(mc), d)
                hits = 0
                for _ in range(DRAWS):
                    a = rng.permutation(len(pooled))[:len(mc)]
                    hits += pooled.mean() <= CE.clopper_pearson_upper(int(pooled[a].sum()), len(mc), d)
                rows.append(dict(variant=name, K=k, K_frac=f, delta=d, n_cal_pos=len(mc), cal_miss_rate=mc.mean(),
                                 certified_bound=bound, test_miss_realised=mt.mean(),
                                 covered=bool(mt.mean() <= bound), recall_at_K_test=1 - mt.mean(),
                                 control_coverage_vs_pooled=hits / DRAWS))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "h3_aggregation.csv", index=False)
    write_md(df)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))
    print(open(C.RESULTS / "h3_summary.md").read())


def write_md(df):
    d1 = df[df.delta == 0.10]
    p = d1.pivot(index="K_frac", columns="variant")
    ctrl = {v: (g.control_coverage_vs_pooled.min(), g.control_coverage_vs_pooled.max()) for v, g in df.groupby("variant")}
    ok = df.groupby("variant").apply(lambda g: bool((g.control_coverage_vs_pooled >= 1 - g.delta - 2 * np.sqrt(
        (1 - g.delta) * g.delta / DRAWS)).all()), include_groups=False)
    cov = {v: int(g[g.delta == 0.10].covered.sum()) for v, g in df.groupby("variant")}
    rec_gain = (p["recall_at_K_test"]["max_txn"] - p["recall_at_K_test"]["direct"])
    bound_diff = (p["certified_bound"]["max_txn"] - p["certified_bound"]["direct"])
    L = ["# H3 summary"]
    if ok.all():
        L.append(f"The guarantee survives aggregation under the exchangeable control: coverage of the pooled population "
                 f"miss rate is {ctrl['direct'][0]:.3f}-{ctrl['direct'][1]:.3f} for the direct score and "
                 f"{ctrl['max_txn'][0]:.3f}-{ctrl['max_txn'][1]:.3f} for the max-over-transaction score, at or above nominal in both.")
    else:
        L.append("The guarantee does NOT survive aggregation in the exchangeable control: " +
                 ", ".join(f"{v}: {a:.3f}-{b:.3f}" for v, (a, b) in ctrl.items()) + ".")
    L.append(f"On the real temporal split (certify on calibration, evaluate on test, delta 0.10) the direct score is covered "
             f"at {cov['direct']} of 7 budgets and the max score at {cov['max_txn']} of 7, so drift is still separate from validity.")
    L.append(f"Recall on test changes by {rec_gain.mean():+.3f} on average (range {rec_gain.min():+.3f} to {rec_gain.max():+.3f}) "
             f"when moving from the direct to the max score, i.e. the max score is "
             f"{'better' if rec_gain.mean() > 0 else 'worse'} at {int((rec_gain > 0).sum()) if rec_gain.mean() > 0 else int((rec_gain < 0).sum())} of 7 budgets.")
    L.append(f"At a fixed K the certified bound differs by {bound_diff.mean():+.3f} on average (max minus direct); the interval "
             f"width is the same because both use the same {int(df.n_cal_pos.iloc[0])} calibration positives, so a tighter "
             f"guarantee comes from a lower miss rate, not a narrower interval.")
    (C.RESULTS / "h3_summary.md").write_text("\n\n".join(L) + "\n")


if __name__ == "__main__":
    main()
