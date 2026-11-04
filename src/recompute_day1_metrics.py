"""Recompute the Day 1 test metrics with margin ranking and random tie-breaking. No retraining.
The previous file is kept as results/day1_model_metrics_v2_table_order_ties.json."""
import json

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src import features as F
from src import scoring
from src import train as T

if __name__ == "__main__":
    old = json.loads((C.RESULTS / "day1_model_metrics_v2_table_order_ties.json").read_text())
    sc = scoring.load_scores()                                   # seeded shuffle done here
    te, ca = sc[sc.split == "test"], sc[sc.split == "calibration"]
    avg = te.groupby("period").size().mean()
    ks = {f: max(1, int(round(f * avg))) for f in T.K_FRACTIONS}
    assert list(ks.values()) == list(old["K_values"].values())   # same K values as before
    new = dict(old)
    new["test"] = T.evaluate(te, ks)
    new["calibration_sanity"] = {"roc_auc": float(roc_auc_score(ca.label, ca.margin)),
                                 "pr_auc": float(average_precision_score(ca.label, ca.margin))}
    new["note"] = ("Recomputed in Day 2 close-out: entities ranked by raw margin (log-odds) with one seeded random "
                   "shuffle for residual ties; ROC/PR-AUC also computed on margin. The earlier file "
                   "(day1_model_metrics_v2_table_order_ties.json) ranked by probability, which saturates at 1.0 "
                   "for 4.6% of test rows, and broke ties by table order, which made recall optimistic.")
    (C.RESULTS / "day1_model_metrics.json").write_text(json.dumps(new, indent=2))
    o, n = old["test"], new["test"]
    print(f"ROC-AUC old {o['roc_auc']:.4f} -> new {n['roc_auc']:.4f};  PR-AUC old {o['pr_auc']:.4f} -> new {n['pr_auc']:.4f}")
    print(f"{'K':>6} {'old (table-order ties)':>24} {'new (margin, random ties)':>26}")
    for k in o["recall_at_k"]:
        print(f"{k:>6} {o['recall_at_k'][k]['pooled']:>24.4f} {n['recall_at_k'][k]['pooled']:>26.4f}")
    print("per-typology AUC new:", {t: round(v['roc_auc'], 3) for t, v in n['by_typology'].items()})
    # how bad are the residual ties at the cutoff? size of the tie block that straddles rank K, per period
    for f, k in ks.items():
        sizes = []
        for p, g in te.groupby("period"):
            m = g["margin"].sort_values(ascending=False).to_numpy()
            thr = m[k - 1]
            sizes.append(int((m == thr).sum()))
        print(f"K={k:>5}: tie block at the cutoff, per period (rows sharing the K-th margin): {sizes}")
