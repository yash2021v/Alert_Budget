"""Is the temporal under-coverage a time shift? Per-period miss rates, plus a control experiment that removes
the time order by pooling calibration and test miss flags and re-splitting them at random.

Writes results/shift_diagnostic.csv, results/figures/shift_diagnostic.png, results/shift_diagnostic.md.
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import certify as CE
from src import config as C
from src import scoring
from src.run_budget_curve import K_FRACS

DELTAS = [0.10, 0.05]
DRAWS = C.MC


def split_flags(df, k):
    """Per-positive miss flags (positives in table order) and the period of each positive."""
    m = CE.missed_flags(df["margin"].to_numpy(), df["label"].to_numpy(), df["period"].to_numpy(), k)
    return m, df.loc[df["label"] == 1, "period"].to_numpy()


def main():
    rng = np.random.default_rng(C.SEED)
    sc = scoring.load_scores()
    cal, test = sc[sc.split == "calibration"], sc[sc.split == "test"]
    avg = test.groupby("period").size().mean()
    ks = {f: int(round(f * avg)) for f in K_FRACS}
    rows = []
    # ---- per-period table (periods 12-19)
    per = sc.groupby("period").agg(n_entity_periods=("label", "size"), n_pos=("label", "sum"))
    per["pos_rate"] = per["n_pos"] / per["n_entity_periods"]
    per["split"] = sc.groupby("period")["split"].first()
    flags = {}
    for f, k in ks.items():
        mc, pc = split_flags(cal, k)
        mt, pt = split_flags(test, k)
        flags[f] = (mc, mt)
        miss = pd.Series(np.r_[mc, mt]).groupby(np.r_[pc, pt]).mean()
        for p, r in per.iterrows():
            rows.append(dict(section="per_period", K=k, K_frac=f, period=int(p), split=r["split"],
                             n_entity_periods=int(r["n_entity_periods"]), n_pos=int(r["n_pos"]),
                             pos_rate=r["pos_rate"], miss_rate=miss[p]))
    # ---- temporal vs control
    summary = []
    for f, k in ks.items():
        mc, mt = flags[f]
        pooled = np.r_[mc, mt]
        n_c, pop = len(mc), pooled.mean()
        for d in DELTAS:
            t_bound = CE.clopper_pearson_upper(int(mc.sum()), n_c, d)
            cov_pop = cov_rem = 0
            for _ in range(DRAWS):
                perm = rng.permutation(len(pooled))
                a, b = perm[:n_c], perm[n_c:]
                ub = CE.clopper_pearson_upper(int(pooled[a].sum()), n_c, d)
                cov_pop += pop <= ub
                cov_rem += pooled[b].mean() <= ub
            row = dict(section="control_vs_temporal", K=k, K_frac=f, delta=d, n_cal_pos=n_c, n_test_pos=len(mt),
                       cal_miss_rate=mc.mean(), test_miss_rate=mt.mean(), pooled_miss_rate=pop,
                       temporal_bound=t_bound, temporal_covered=bool(mt.mean() <= t_bound),
                       control_coverage_vs_pooled=cov_pop / DRAWS, control_coverage_vs_remainder=cov_rem / DRAWS,
                       draws=DRAWS)
            rows.append(row)
            summary.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "shift_diagnostic.csv", index=False)
    S = pd.DataFrame(summary)
    plot(df, S)
    write_md(S, df)
    pd.set_option("display.width", 200)
    print(df[df.section == "per_period"].pivot(index="period", columns="K", values="miss_rate")
          .join(per[["split", "n_entity_periods", "n_pos", "pos_rate"]]).loc[12:].round(4).to_string())
    print(S[["K", "delta", "cal_miss_rate", "test_miss_rate", "temporal_bound", "temporal_covered",
             "control_coverage_vs_pooled", "control_coverage_vs_remainder"]].round(4).to_string(index=False))


def plot(df, S):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    pp = df[df.section == "per_period"]
    for k, g in pp.groupby("K"):
        ax[0].plot(g["period"], g["miss_rate"], "o-", ms=3, label=f"K={k}")
    ax[0].axvline(15.5, color="gray", ls="--")
    ax[0].text(13.5, 0.02, "calibration", ha="center", fontsize=8)
    ax[0].text(17.5, 0.02, "test", ha="center", fontsize=8)
    ax[0].set_xlabel("period (12h blocks)"); ax[0].set_ylabel("miss rate"); ax[0].legend(fontsize=7)
    ax[0].set_title("Miss rate by period (periods 12-19)")
    s = S[S.delta == 0.10]
    x = s["K_frac"] * 100
    ax[1].plot(x, s["control_coverage_vs_pooled"], "o-", label="control: vs pooled population rate")
    ax[1].plot(x, s["control_coverage_vs_remainder"], "s-", label="control: vs remainder's observed rate")
    ax[1].scatter(x, s["temporal_covered"].astype(float), marker="x", s=70, c="#c0392b", label="temporal: covered (1) / not (0)")
    ax[1].axhline(0.90, color="k", ls=":", label="nominal 0.90")
    ax[1].set_xscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([f"{v:g}" for v in x])
    ax[1].set_xlabel("budget K (% of entity-periods per period)"); ax[1].set_ylabel("coverage")
    ax[1].set_title("Coverage at delta = 0.10: random split vs time split"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "shift_diagnostic.png", dpi=130)


def per_sizes(df):
    pp = df[(df.section == "per_period") & (df.K == df.K.min())]
    c, t = pp[pp.period <= 15], pp[pp.period >= 16]
    return {"c": (int(c.n_entity_periods.min()), int(c.n_entity_periods.max())),
            "t": (int(t.n_entity_periods.min()), int(t.n_entity_periods.max()))}


def write_md(S, df):
    se = lambda d: np.sqrt((1 - d) * d / DRAWS)
    n = S["K"].nunique()
    lines = ["# Shift diagnostic"]
    cov = {d: int(S[(S.delta == d) & S.temporal_covered].shape[0]) for d in DELTAS}
    lines.append(f"Certifying on the calibration split and checking the test split (time order kept), the bound is covered "
                 f"at {cov[0.10]} of {n} budgets at delta 0.10 and {cov[0.05]} of {n} at delta 0.05.")
    ok = {}
    for d in DELTAS:
        s = S[S.delta == d]
        ok[d] = bool((s["control_coverage_vs_pooled"] >= 1 - d - 2 * se(d)).all())
        lines.append(f"With the time order removed (pooled calibration and test flags, re-split at random {DRAWS} times), "
                     f"coverage of the pooled population rate is {s.control_coverage_vs_pooled.min():.3f}-"
                     f"{s.control_coverage_vs_pooled.max():.3f} at delta {d} (nominal {1 - d:.2f}), and coverage of the "
                     f"remainder's observed rate is {s.control_coverage_vs_remainder.min():.3f}-"
                     f"{s.control_coverage_vs_remainder.max():.3f} (that comparison cannot reach nominal by construction).")
    pp = df[df.section == "per_period"]
    mid = pp[pp.K == sorted(pp.K.unique())[3]]
    c, t = mid[mid.period <= 15], mid[mid.period >= 16]
    lines.append(f"Per period, the positive rate is {c.pos_rate.mean():.3%} in calibration periods 12-15 versus "
                 f"{t.pos_rate.mean():.3%} in test periods 16-19, and the miss rate at K={sorted(pp.K.unique())[3]} averages "
                 f"{c.miss_rate.mean():.3f} versus {t.miss_rate.mean():.3f}.")
    sz = per_sizes(df)
    lines.append(f"Period volume also differs: calibration periods hold {sz['c'][0]:,}-{sz['c'][1]:,} entity-periods versus "
                 f"{sz['t'][0]:,}-{sz['t'][1]:,} in test, and because K is a fixed absolute count a busy period reviews a "
                 f"smaller share of its entities, so the miss rate moves with volume.")
    if all(ok.values()) and cov[0.10] < n:
        lines.append("Coverage returns to nominal once the time order is removed while the temporal split under-covers, "
                     "which points to a shift between the calibration and test periods rather than a bug in the "
                     "certification code.")
    elif not all(ok.values()):
        lines.append("Coverage does NOT return to nominal even with the time order removed, so the temporal under-coverage "
                     "is not explained by time shift alone and the certification code or flag dependence needs a closer look.")
    else:
        lines.append("The temporal split also covers at every budget here, so no shift effect is visible at this setting.")
    lines.append("This control cannot tell score drift from a change in the mix of positives; it only shows that the "
                 "miss flags are not exchangeable across time.")
    (C.RESULTS / "shift_diagnostic.md").write_text("\n\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
