"""Certify on the CALIBRATION split, then check the realised miss rate on the TEST split (same K, per-period top K).

Nothing is adjusted if some budgets are not covered: drift between calibration and test is a result.
Writes results/miss_rate_certified_vs_realised.csv and results/figures/budget_curve.png.
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import certify as CE
from src import config as C

K_FRACS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
DELTAS = [0.10, 0.05]
OUT = C.RESULTS / "miss_rate_certified_vs_realised.csv"


def build_curve(scores: pd.DataFrame) -> pd.DataFrame:
    # shuffle once so exact score ties are not resolved by table order (see certify.py)
    cal = scores[scores["split"] == "calibration"].sample(frac=1.0, random_state=C.SEED)
    test = scores[scores["split"] == "test"].sample(frac=1.0, random_state=C.SEED)
    avg = test.groupby("period").size().mean()          # same base as Day 1 K values (155, 388, ...)
    rows = []
    for f in K_FRACS:
        K = int(round(f * avg))
        t_missed, t_pos = CE.missed_counts(test["score"], test["label"], test["period"], K)
        for d in DELTAS:
            c = CE.certify(cal["score"], cal["label"], cal["period"], K, d)
            realised = t_missed / t_pos
            rows.append(dict(K=K, K_frac=f, delta=d, n_cal_pos=c["n_pos"], cal_miss_rate=c["point_miss_rate"],
                             certified_bound=c["upper_bound"], n_test_pos=t_pos, test_miss_rate_realised=realised,
                             covered=bool(realised <= c["upper_bound"]), recall_at_K_test=1 - realised))
    return pd.DataFrame(rows)


def plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, d in zip(axes, DELTAS):
        s = df[df["delta"] == d]
        ax.plot(s["K_frac"] * 100, s["certified_bound"], "o-", color="#1f5fa8", label="certified upper bound")
        ax.plot(s["K_frac"] * 100, s["cal_miss_rate"], ":", color="#1f5fa8", alpha=0.6, label="calibration miss rate")
        ax.plot(s["K_frac"] * 100, s["test_miss_rate_realised"], "s-", color="#d9731a", label="realised test miss rate")
        bad = s[~s["covered"]]
        ax.scatter(bad["K_frac"] * 100, bad["test_miss_rate_realised"], s=140, facecolors="none",
                   edgecolors="#c0392b", linewidths=2, label="not covered")
        ax.set_xscale("log")
        ax.set_xticks(s["K_frac"] * 100)
        ax.set_xticklabels([f"{x:g}" for x in s["K_frac"] * 100])
        ax.set_xlabel("budget K (% of entity-periods per period)")
        ax.set_title(f"Nominal confidence {1 - d:.0%} (delta = {d})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("miss rate")
    axes[0].legend(fontsize=8)
    fig.suptitle("Certified vs realised miss rate (calibrate on calibration split, evaluate on test split)")
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "budget_curve.png", dpi=130)


if __name__ == "__main__":
    df = build_curve(pd.read_parquet(C.PROCESSED / "scores.parquet"))
    df.to_csv(OUT, index=False)
    plot(df)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for d in DELTAS:
        s = df[df["delta"] == d]
        print(f"\ndelta={d}: covered K = {s.loc[s.covered, 'K'].tolist()}; "
              f"NOT covered K = {s.loc[~s.covered, 'K'].tolist()}")
