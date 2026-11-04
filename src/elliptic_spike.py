"""H4 spike on Elliptic: does certified coverage change around the step-43 dark-market shutdown? No ACI, no tuning.

Labelled nodes only (class 1 = illicit, 2 = licit, "unknown" excluded), native 165 features (time step is NOT a
feature). Split by time step: train 1-24, calibration 25-34, evaluation 35-49. Fixed-parameter LightGBM on train only,
ranked by raw margin. FRACTION budget: K_t = ceil(q * labelled nodes in step t), q = 10% and 20%, because the number
of labelled nodes per step varies from ~200 to ~2,200 and a fixed count would review wildly different shares.
Certify on the calibration steps (delta 0.10), then check each evaluation step. Rule for 'visible change' (fixed in
advance): the covered share of evaluation steps with at least one illicit node drops by >= 25 points after step 43.
"""
import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import certify as CE
from src import config as C

TRAIN, CAL, EVAL = (1, 24), (25, 34), (35, 49)
QS = [0.10, 0.20]
DELTA = 0.10
SHUTDOWN = 43
DRAWS = 1000
PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=C.SEED,
              deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)


def load_labelled() -> pd.DataFrame:
    n = open(C.ELLIPTIC_FEATURES).readline().count(",") + 1
    names = ["txId", "time_step"] + [f"f{i}" for i in range(1, n - 1)]
    t = pacsv.read_csv(C.ELLIPTIC_FEATURES, read_options=pacsv.ReadOptions(column_names=names),
                       convert_options=pacsv.ConvertOptions(column_types={"txId": pa.int64(), "time_step": pa.int32(),
                                                                           **{c: pa.float32() for c in names[2:]}}))
    cls = pd.read_csv(C.ELLIPTIC_CLASSES, dtype={"class": str})
    cls = cls[cls["class"].isin(["1", "2"])].assign(label=lambda d: (d["class"] == "1").astype(int))
    return t.to_pandas().merge(cls[["txId", "label"]], on="txId")


def main():
    rng = np.random.default_rng(C.SEED)
    df = load_labelled()
    feats = [c for c in df.columns if c.startswith("f")]
    assert len(feats) == 165 and "time_step" not in feats
    tr = df[df.time_step.between(*TRAIN)]
    model = lgb.LGBMClassifier(**PARAMS).fit(tr[feats], tr["label"])
    joblib.dump(model, C.RESULTS / "lgbm_elliptic_spike.joblib")
    df["margin"] = model.predict(df[feats], raw_score=True)
    df = df[df.time_step >= CAL[0]].sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)   # seeded tie shuffle
    cal, ev = df[df.time_step.between(*CAL)], df[df.time_step.between(*EVAL)]
    rows, ctrl = [], []
    for q in QS:
        mc = CE.missed_flags(cal["margin"], cal["label"], cal["time_step"], CE.fraction_k(cal["time_step"], q))
        bound = CE.clopper_pearson_upper(int(mc.sum()), len(mc), DELTA)
        allm = CE.missed_flags(df["margin"], df["label"], df["time_step"], CE.fraction_k(df["time_step"], q))
        pos_steps = df.loc[df["label"] == 1, "time_step"].to_numpy()
        for s in range(CAL[0], EVAL[1] + 1):
            g = df[df.time_step == s]
            m = allm[pos_steps == s]
            rows.append(dict(step=s, q=q, budget_type="fraction", role="calibration" if s <= CAL[1] else "evaluation",
                             n_labelled=len(g), n_illicit=int(g.label.sum()), realised_miss=m.mean() if len(m) else np.nan,
                             bound=bound, covered=bool(m.mean() <= bound) if len(m) and s > CAL[1] else np.nan))
        pop, hits = allm.mean(), 0
        for _ in range(DRAWS):
            a = rng.permutation(len(allm))[:len(mc)]
            hits += pop <= CE.clopper_pearson_upper(int(allm[a].sum()), len(mc), DELTA)
        ctrl.append(dict(q=q, n_cal_pos=len(mc), cal_miss=mc.mean(), bound=bound, pooled_miss=pop, coverage=hits / DRAWS))
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "elliptic_spike.csv", index=False)
    pd.DataFrame(ctrl).to_csv(C.RESULTS / "elliptic_spike_control.csv", index=False)
    plot(out)
    write_md(out, pd.DataFrame(ctrl))
    pd.set_option("display.width", 200)
    print(out[out.role == "evaluation"].round(3).to_string(index=False))
    print(pd.DataFrame(ctrl).round(3).to_string(index=False))
    print(open(C.RESULTS / "elliptic_spike.md").read())


def plot(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, q in zip(axes, QS):
        s = out[out.q == q]
        ax.plot(s.step, s.realised_miss, "o-", color="#d9731a", label="realised miss rate")
        ax.plot(s.step, s.bound, "-", color="#1f5fa8", label=f"bound (calibration, delta {DELTA})")
        ax.axvline(SHUTDOWN, color="#c0392b", ls="--", label="step 43 (shutdown)")
        ax.axvspan(CAL[0] - 0.5, CAL[1] + 0.5, color="gray", alpha=0.12, label="calibration steps")
        bad = s[(s.role == "evaluation") & (s.covered == False)]
        ax.scatter(bad.step, bad.realised_miss, s=120, facecolors="none", edgecolors="#c0392b", linewidths=2, label="not covered")
        ax.set_title(f"Budget q = {q:.0%} of labelled nodes per step"); ax.set_xlabel("time step")
    axes[0].set_ylabel("miss rate"); axes[0].legend(fontsize=7)
    fig.suptitle("Elliptic spike: realised miss rate vs certified bound (fraction budget)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "elliptic_spike.png", dpi=130)


def write_md(out, ctrl):
    L = ["# Elliptic spike (H4)",
         "Budget type: FRACTION of labelled nodes per step (K_t = ceil(q * labelled nodes in step t), q = 10% and 20%), "
         "because step sizes vary from about 200 to 2,200 labelled nodes and a fixed count would review very different shares."]
    ev = out[out.role == "evaluation"].dropna(subset=["realised_miss"])
    for q in QS:
        e = ev[ev.q == q]
        pre, post = e[e.step < SHUTDOWN], e[e.step >= SHUTDOWN]
        c = ctrl[ctrl.q == q].iloc[0]
        drop = pre.covered.mean() - post.covered.mean()
        L.append(f"At q = {q:.0%} the bound from steps 25-34 is {c.bound:.3f} (calibration miss rate {c.cal_miss:.3f}); "
                 f"{int(pre.covered.sum())} of {len(pre)} evaluation steps before 43 are covered versus {int(post.covered.sum())} of "
                 f"{len(post)} from 43 on, and the mean realised miss is {pre.realised_miss.mean():.3f} before versus "
                 f"{post.realised_miss.mean():.3f} after.")
        L.append(f"q = {q:.0%}: " + ("coverage VISIBLY changes at the shutdown (covered share drops by "
                 f"{drop:.0%}, rule: at least 25 points)." if drop >= 0.25 else
                 f"coverage does NOT visibly change at the shutdown (covered share changes by {-drop:+.0%}; rule: a drop of "
                 "at least 25 points); we adopt the negative-result framing if this holds at both budgets."))
        L.append(f"Exchangeable control (steps 25-49 pooled, {DRAWS} random re-splits): coverage of the pooled population miss "
                 f"rate {c.coverage:.3f} (nominal 0.90), so the bound itself is valid when time is removed and any "
                 "under-coverage above is drift rather than a code error.")
    (C.RESULTS / "elliptic_spike.md").write_text("\n\n".join(L) + "\n")


if __name__ == "__main__":
    main()
