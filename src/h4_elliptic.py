"""H4 centerpiece on Elliptic: static vs rolling vs ACI certificates under delayed feedback and the step-43 shutdown.

Fixed in advance (nothing tuned after seeing results)
- Labelled nodes only (1 = illicit, 2 = licit), native 165 features; the frozen LightGBM from the spike (trained on steps
  1-24) ranks by raw margin. Evaluation steps 35-49. Fraction budget K_t = ceil(q * labelled nodes in step t), q in {10%, 20%}.
- Delayed feedback L in {0, 2, 4}: when step t is reviewed only labels of steps <= t-L-1 are available; the pool starts at 25.
- STATIC: bound from steps 25..min(34, t-L-1) (the spike's calibration steps, truncated while those labels have not arrived).
- ROLLING: pool = most recent W = 5 available steps, max(25, t-L-5) .. t-L-1; delta fixed at 0.10.
- ACI: same rolling pool with an adaptive delta_t. delta_35 = 0.10 and, literally as specified,
  delta_{t+1} = clip(delta_t + gamma * (0.10 - err_s), 1e-4, 0.5) with s = t-L-1, the most recent step whose labels have arrived,
  err_s = binomial error indicator of s against the bound ISSUED for s. Only evaluated steps (s >= 35) have an issued bound;
  for s < 35 there is no update. gamma = 0.05 is the main setting; gamma in {0.01, 0.10} are a fixed sensitivity check.
- Recovery lag: first step s >= 43 such that the error indicator is 0 at s, s+1, s+2 (all within step 49); lag = s - 43.
Writes results/h4_elliptic_arms.csv, results/h4_elliptic_summary.csv, results/h4_summary.md and two figures.
"""
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import certify as CE
from src import config as C
from src.coverage_monitor import error_frequency, error_indicator
from src.elliptic_spike import load_labelled

QS, LS = [0.10, 0.20], [0, 2, 4]
EVAL = range(35, 50)
POOL_START, W, SHUTDOWN = 25, 5, 43
GAMMAS = {"aci": 0.05, "aci_g0.01": 0.01, "aci_g0.10": 0.10}
DELTA0 = 0.10


def step_counts(q):
    """Per-step (misses, positives, n_labelled) at budget q for steps 25..49, ranking by frozen margin."""
    df = load_labelled()
    feats = [c for c in df.columns if c.startswith("f")]
    model = joblib.load(C.RESULTS / "lgbm_elliptic_spike.joblib")
    df["margin"] = model.predict(df[feats], raw_score=True)
    df = df[df.time_step >= POOL_START].sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    flags = CE.missed_flags(df["margin"], df["label"], df["time_step"], CE.fraction_k(df["time_step"], q))
    steps = df.loc[df["label"] == 1, "time_step"].to_numpy()
    n_lab = df.groupby("time_step").size()
    return {s: (int(flags[steps == s].sum()), int((steps == s).sum()), int(n_lab[s])) for s in range(POOL_START, 50)}


def pool_bound(counts, lo, hi, delta):
    k = sum(counts[s][0] for s in range(lo, hi + 1))
    n = sum(counts[s][1] for s in range(lo, hi + 1))
    return CE.clopper_pearson_upper(k, n, delta), n


def run_arm(arm, L, q, counts, gamma=None):
    rows, delta_t, errs = [], DELTA0, {}
    for t in EVAL:
        if arm == "static":
            lo, hi, d_t = POOL_START, min(34, t - L - 1), DELTA0
        elif arm == "rolling":
            lo, hi, d_t = max(POOL_START, t - L - W), t - L - 1, DELTA0
        else:
            lo, hi, d_t = max(POOL_START, t - L - W), t - L - 1, delta_t
        bound, n_pool = pool_bound(counts, lo, hi, d_t)
        k, n, n_lab = counts[t]
        err = error_indicator(k, n, bound)
        errs[t] = err
        rows.append(dict(arm=arm, gamma=gamma if arm == "aci" else np.nan, L=L, q=q, step=t, n_labelled=n_lab,
                         n_illicit=n, misses=k, bound=bound, delta_t=d_t, pool_positives=n_pool, err=err,
                         realised_miss=k / n if n else np.nan, plain_covered_noisy=bool(k / n <= bound) if n else np.nan))
        if arm == "aci":                                    # delta_{t+1} from s = t-L-1 (needs an issued bound: s >= 35)
            s = t - L - 1
            if s >= EVAL.start:
                delta_t = float(np.clip(delta_t + gamma * (DELTA0 - errs[s]), 1e-4, 0.5))
    return rows


def recovery_lag(errs_by_step):
    for s in range(SHUTDOWN, 50 - 2):
        if all(errs_by_step.get(x, 0) == 0 for x in (s, s + 1, s + 2)):
            return s - SHUTDOWN
    return None


def main():
    allrows = []
    for q in QS:
        counts = step_counts(q)
        for L in LS:
            allrows += run_arm("static", L, q, counts)
            allrows += run_arm("rolling", L, q, counts)
            for name, g in GAMMAS.items():
                rows = run_arm("aci", L, q, counts, g)
                for r in rows:
                    r["arm"] = name
                allrows += rows
    df = pd.DataFrame(allrows)
    df.to_csv(C.RESULTS / "h4_elliptic_arms.csv", index=False)
    summ = []
    for (arm, L, q), g in df.groupby(["arm", "L", "q"]):
        pre, post = g[g.step < SHUTDOWN], g[g.step >= SHUTDOWN]
        lag = recovery_lag(dict(zip(g.step, g.err)))
        summ.append(dict(arm=arm, L=L, q=q, err_freq_before=error_frequency(pre.err, pre.n_illicit),
                         err_freq_after=error_frequency(post.err, post.n_illicit),
                         mean_bound_before=pre.bound.mean(), mean_bound_after=post.bound.mean(),
                         max_bound_after=post.bound.max(), recovery_lag="none" if lag is None else lag))
    sm = pd.DataFrame(summ)
    sm.to_csv(C.RESULTS / "h4_elliptic_summary.csv", index=False)
    plot_center(df)
    plot_all(df)
    write_md(sm, df)
    pd.set_option("display.width", 220)
    print(sm.round(3).to_string(index=False))
    print(open(C.RESULTS / "h4_summary.md").read())


def plot_center(df):
    L, q = 2, 0.10
    g = df[(df.L == L) & (df.q == q)]
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.3])
    for i, (arm, title) in enumerate([("static", "STATIC (steps 25-34)"), ("rolling", "ROLLING (W = 5)"),
                                      ("aci", "ACI (W = 5, gamma = 0.05)")]):
        ax = fig.add_subplot(gs[0, i])
        a = g[g.arm == arm]
        ax.plot(a.step, a.realised_miss, "o-", color="#d9731a", label="realised miss")
        ax.plot(a.step, a.bound, "s-", color="#1f5fa8", label="bound")
        bad = a[a.err == 1]
        ax.scatter(bad.step, bad.realised_miss, s=130, facecolors="none", edgecolors="#c0392b", linewidths=2, label="error indicator = 1")
        ax.axvline(SHUTDOWN, color="#c0392b", ls="--")
        ax.set_title(title); ax.set_ylim(0, 1.05); ax.set_xlabel("time step")
        if i == 0:
            ax.set_ylabel("miss rate"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, :])
    a = g[g.arm == "aci"]
    ax.plot(a.step, a.delta_t, "o-", color="#1b7f3b")
    ax.axvline(SHUTDOWN, color="#c0392b", ls="--"); ax.set_ylabel("ACI delta_t"); ax.set_xlabel("time step")
    fig.suptitle(f"H4 centerpiece: Elliptic, delay L = {L}, budget q = {q:.0%} of labelled nodes per step (dashed = step 43)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "h4_centerpiece.png", dpi=130)


def plot_all(df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    colors = {"static": "#7f7f7f", "rolling": "#1f5fa8", "aci": "#1b7f3b"}
    for i, q in enumerate(QS):
        for j, L in enumerate(LS):
            ax = axes[i, j]
            g = df[(df.L == L) & (df.q == q)]
            r = g[g.arm == "static"]
            ax.plot(r.step, r.realised_miss, "o-", color="#d9731a", label="realised miss")
            for arm in ("static", "rolling", "aci"):
                a = g[g.arm == arm]
                ax.plot(a.step, a.bound, "-", color=colors[arm], label=f"{arm} bound")
            ax.axvline(SHUTDOWN, color="#c0392b", ls="--")
            ax.set_title(f"L = {L}, q = {q:.0%}", fontsize=9)
            if i == 1: ax.set_xlabel("time step")
            if j == 0: ax.set_ylabel("miss rate")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("H4: all delays and budgets (ACI gamma = 0.05; dashed = step 43)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "h4_all_delays.png", dpi=130)


def write_md(sm, df):
    m = sm[(sm.L == 2) & (sm.q == 0.10)].set_index("arm")
    L = ["# H4 summary (Elliptic, delay L = 2, q = 10% unless stated)"]
    for arm in ("static", "rolling", "aci"):
        r = m.loc[arm]
        L.append(f"{arm.upper()}: the error indicator fires at a rate of {r.err_freq_before:.2f} before step 43 and {r.err_freq_after:.2f} "
                 f"from step 43 on (target 0.10), with mean bound {r.mean_bound_before:.3f} before and {r.mean_bound_after:.3f} after "
                 f"(max {r.max_bound_after:.3f}) and recovery lag {r.recovery_lag}.")
    a = m.loc["aci"]
    if a.max_bound_after >= 0.9:
        L.append(f"ACI's bound reaches {a.max_bound_after:.3f} after the shutdown, i.e. it nearly gives up on tightness: "
                 "restoring coverage can cost bound tightness, which is a finding rather than a failure.")
    else:
        L.append(f"ACI's bound after the shutdown peaks at {a.max_bound_after:.3f}, so any coverage restored does not require a bound near 1.")
    sens = sm[(sm.L == 2) & (sm.q == 0.10) & sm.arm.str.startswith("aci")].set_index("arm")
    L.append("Sensitivity of ACI to gamma (fixed values, not for selection): error frequency after step 43 is " +
             ", ".join(f"{a_}: {sens.loc[a_, 'err_freq_after']:.2f}" for a_ in sens.index) + ".")
    e = sm[(sm.q == 0.10)].pivot(index="L", columns="arm", values="err_freq_after")
    L.append("Effect of the delay L at q = 10% (error frequency from step 43 on): " +
             "; ".join(f"L = {l}: static {e.loc[l, 'static']:.2f}, rolling {e.loc[l, 'rolling']:.2f}, ACI {e.loc[l, 'aci']:.2f}" for l in e.index) + ".")
    (C.RESULTS / "h4_summary.md").write_text("\n\n".join(L) + "\n")


if __name__ == "__main__":
    main()
