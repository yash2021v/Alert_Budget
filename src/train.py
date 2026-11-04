"""Train the LightGBM scorer on the train split only, evaluate on test, then freeze it."""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src import features as F
from src import panel as P
from src import split as S

MODEL_PATH = C.RESULTS / "lgbm_frozen.joblib"
METRICS_PATH = C.RESULTS / "day1_model_metrics.json"
SCORES_PATH = C.PROCESSED / "scores.parquet"
K_FRACTIONS = [0.001, 0.0025, 0.005, 0.01, 0.02]   # of average entity-periods per period (test split)

# Fixed hyperparameters; no search. Imbalance handled with scale_pos_weight = n_neg / n_pos (train).
PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=C.SEED, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)


def load_table() -> pd.DataFrame:
    df = P.build_panel().merge(S.load_split(), on=["acct_key", "period"]).merge(
        F.load_features(), on=["acct_key", "period"])
    return df


def recall_at_k(df: pd.DataFrame, k: int, mask: pd.Series | None = None) -> dict:
    """Per-period top-k by score. Recall = captured positives / positives (pooled and mean over periods)."""
    top = df.sort_values("score", ascending=False).groupby("period").head(k)
    sel = top if mask is None else top[mask.loc[top.index]]
    pos = df if mask is None else df[mask]
    pos = pos[pos["label"] == 1]
    cap = sel[sel["label"] == 1]
    per = (cap.groupby("period").size().reindex(pos["period"].unique(), fill_value=0)
           / pos.groupby("period").size())
    return {"pooled": float(len(cap) / max(len(pos), 1)), "mean_over_periods": float(per.mean())}


def evaluate(df: pd.DataFrame, ks: dict) -> dict:
    y, s = df["label"], df["score"]
    out = {"n_rows": int(len(df)), "n_pos": int(y.sum()),
           "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
           "recall_at_k": {f"K={k}": recall_at_k(df, k) for k in ks.values()}}
    by = {}
    neg = df[df["label"] == 0]
    for t in sorted(df.loc[df.label == 1, "typology"].unique()):
        pos_t = df[(df["label"] == 1) & (df["typology"] == t)]
        sub = pd.concat([pos_t, neg])
        mask = (df["typology"] == t)
        by[t] = {"n_pos": int(len(pos_t)),
                 "roc_auc": float(roc_auc_score(sub["label"], sub["score"])),
                 "pr_auc": float(average_precision_score(sub["label"], sub["score"])),
                 "recall_at_k": {f"K={k}": recall_at_k(df, k, mask)["pooled"] for k in ks.values()}}
    out["by_typology"] = by
    return out


if __name__ == "__main__":
    df = load_table()
    tr, ca, te = (df[df.split == s].copy() for s in ("train", "calibration", "test"))
    n_pos, n_neg = int(tr.label.sum()), int((tr.label == 0).sum())
    params = {**PARAMS, "scale_pos_weight": n_neg / n_pos}
    print(f"train rows {len(tr):,}, positives {n_pos:,}, scale_pos_weight {params['scale_pos_weight']:.1f}")
    model = lgb.LGBMClassifier(**params).fit(tr[F.FEATURE_COLS], tr["label"])
    for d in (ca, te):
        d["score"] = model.predict_proba(d[F.FEATURE_COLS])[:, 1]
    avg_rows = te.groupby("period").size().mean()
    ks = {f: max(1, int(round(f * avg_rows))) for f in K_FRACTIONS}
    metrics = {"params": params, "features": F.FEATURE_COLS, "avg_entity_periods_per_period_test": float(avg_rows),
               "K_values": {f"{f:.2%}": k for f, k in ks.items()}, "test": evaluate(te, ks),
               "calibration_sanity": {"roc_auc": float(roc_auc_score(ca.label, ca.score)),
                                      "pr_auc": float(average_precision_score(ca.label, ca.score))}}
    imp = pd.Series(model.booster_.feature_importance("gain"), index=F.FEATURE_COLS).sort_values(ascending=False)
    metrics["top_feature_gain_share"] = (imp / imp.sum()).round(4).head(15).to_dict()
    C.RESULTS.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, MODEL_PATH)
    cols = ["acct_key", "period", "split", "label", "typology", "score"]
    pd.concat([ca[cols], te[cols]]).to_parquet(SCORES_PATH)
    t = metrics["test"]
    print(f"TEST roc_auc {t['roc_auc']:.4f}  pr_auc {t['pr_auc']:.4f}  (base rate {t['n_pos'] / t['n_rows']:.4%})")
    print("K values:", metrics["K_values"])
    for k, v in t["recall_at_k"].items():
        print(f"  recall@{k}: pooled {v['pooled']:.3f}  mean/period {v['mean_over_periods']:.3f}")
    print("by typology (auc / pr / recall@K pooled):")
    for name, v in t["by_typology"].items():
        print(f"  {name:<15} n={v['n_pos']:>4} auc {v['roc_auc']:.3f} pr {v['pr_auc']:.4f} " +
              " ".join(f"{r:.2f}" for r in v["recall_at_k"].values()))
    print("top 15 features (share of total gain):")
    for n, v in metrics["top_feature_gain_share"].items():
        print(f"  {n:<26}{v:.4f}")
    print("\n*** MODEL FROZEN - do not retune ***")
