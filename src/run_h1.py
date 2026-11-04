"""H1: group-conditional certificates.

Typology is only known for positives, so there are NO per-typology thresholds: the reviewed set is the usual
top-K by margin per period. For each typology group g (8 typologies + UNATTRIBUTED) we certify on the calibration
positives of g and compare with g's realised test miss rate; we also check the POOLED bound against each group.
An exchangeable protocol (pool g's calibration+test positives, 1,000 random re-splits) separates "pooled bound
under-serves the group / temporal drift" from sampling variation.

Writes results/coverage_by_typology.csv, results/h1_bound_width.csv, results/figures/h1_typology.png.
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import certify as CE
from src import config as C
from src import scoring

K_FRACS = [0.005, 0.01, 0.02, 0.05]
DELTAS = [0.10, 0.05]
DRAWS = 1000
LOW_N = 30


def flags_and_groups(df, k):
    m = CE.missed_flags(df["margin"].to_numpy(), df["label"].to_numpy(), df["period"].to_numpy(), k)
    return m, df.loc[df["label"] == 1, "typology"].to_numpy()


def main():
    rng = np.random.default_rng(C.SEED)
    sc = scoring.load_scores()
    cal, test = sc[sc.split == "calibration"], sc[sc.split == "test"]
    avg = test.groupby("period").size().mean()
    rows = []
    for f in K_FRACS:
        k = int(round(f * avg))
        mc, gc = flags_and_groups(cal, k)
        mt, gt = flags_and_groups(test, k)
        for d in DELTAS:
            pooled = CE.clopper_pearson_upper(int(mc.sum()), len(mc), d)
            cg = CE.group_certificates(mc, gc, d)
            for g, c in cg.items():
                t = mt[gt == g]
                test_miss = t.mean() if len(t) else float("nan")
                pool = np.r_[mc[gc == g], t]
                n_c = c["n_pos"]
                cov, cal_miss = float("nan"), np.full(3, np.nan)
                if 0 < n_c < len(pool):
                    hits, cms = 0, []
                    for _ in range(DRAWS):
                        a = rng.permutation(len(pool))[:n_c]
                        hits += pool.mean() <= CE.clopper_pearson_upper(int(pool[a].sum()), n_c, d)
                        cms.append(pool[a].mean())
                    cov = hits / DRAWS
                    cal_miss = np.array([np.mean(cms), *np.percentile(cms, [25, 75])])
                rows.append(dict(typology=g, K=k, K_frac=f, delta=d, n_cal_pos=n_c, n_test_pos=len(t),
                                 group_bound=c["upper_bound"], pooled_bound=pooled, group_cal_miss=c["point_miss_rate"],
                                 group_test_miss=test_miss, group_covered=bool(test_miss <= c["upper_bound"]),
                                 pooled_covers_group=bool(test_miss <= pooled), low_n=n_c < LOW_N,
                                 exch_coverage=cov, exch_cal_miss_mean=cal_miss[0], exch_cal_miss_q25=cal_miss[1],
                                 exch_cal_miss_q75=cal_miss[2]))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "coverage_by_typology.csv", index=False)
    w = df[["typology", "K", "delta", "n_cal_pos", "group_cal_miss", "group_bound", "low_n"]].copy()
    w["bound_width"] = w["group_bound"] - w["group_cal_miss"]
    w.sort_values(["delta", "K", "n_cal_pos"]).to_csv(C.RESULTS / "h1_bound_width.csv", index=False)
    plot(df)
    pd.set_option("display.width", 250)
    for d in DELTAS:
        s = df[(df.delta == d) & (df.K_frac == 0.01)]
        print(f"\n--- K_frac 1%, delta {d} ---")
        print(s[["typology", "n_cal_pos", "n_test_pos", "group_bound", "pooled_bound", "group_cal_miss", "group_test_miss",
                 "group_covered", "pooled_covers_group", "exch_coverage", "exch_cal_miss_mean", "exch_cal_miss_q25",
                 "exch_cal_miss_q75"]].round(3).to_string(index=False))
    for d in DELTAS:
        s = df[df.delta == d]
        print(f"\ndelta {d}: group bound covers {int(s.group_covered.sum())}/{len(s)} (typology, K) cells; "
              f"pooled bound covers {int(s.pooled_covers_group.sum())}/{len(s)}; "
              f"exchangeable coverage range {s.exch_coverage.min():.3f}-{s.exch_coverage.max():.3f} "
              f"(nominal {1 - d}); low-n groups: {sorted(s[s.low_n].typology.unique())}")


def plot(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for ax, f in zip(axes.ravel(), K_FRACS):
        s = df[(df.delta == 0.10) & (df.K_frac == f)].reset_index(drop=True)
        x = np.arange(len(s))
        b1 = ax.bar(x - 0.2, s["group_bound"], 0.4, color="#1f5fa8", label="group bound (90%)")
        b2 = ax.bar(x + 0.2, s["group_test_miss"], 0.4, color="#d9731a", label="realised test miss")
        for i, low in enumerate(s["low_n"]):
            if low:
                b1[i].set_hatch("//"); b2[i].set_hatch("//")
        ax.axhline(s["pooled_bound"].iloc[0], color="k", ls="--", lw=1, label="pooled bound")
        ax.set_xticks(x); ax.set_xticklabels(s["typology"], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"K = {f:.1%} of period size ({s['K'].iloc[0]})", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("H1: group-conditional bound vs realised miss rate (delta 0.10; hatched = fewer than 30 calibration positives)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "h1_typology.png", dpi=130)


if __name__ == "__main__":
    main()
