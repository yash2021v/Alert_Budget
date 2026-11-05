"""Build report/figures: Fig1 (validity gate, redrawn from the CSV), Fig2 (fixed-count vs fraction budget), and copies of Fig3-5.
Use --fig1-fig2 to regenerate only Fig1 and Fig2."""
import shutil

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C

OUT = C.ROOT / "report" / "figures"


def fig2():
    fx = pd.read_csv(C.RESULTS / "miss_rate_certified_vs_realised.csv")
    fx = fx[(fx.scorer == "direct") & np.isclose(fx.delta, 0.10)]
    fr = pd.read_csv(C.RESULTS / "miss_rate_fraction_budget.csv")
    fr = fr[np.isclose(fr.delta, 0.10)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, df, title in ((axes[0], fx, "Fixed-count budget (primary definition)"), (axes[1], fr, "Fraction-of-volume budget (diagnostic)")):
        x = df.K_frac * 100
        ax.plot(x, df.certified_bound, "o-", color="#1f5fa8", label="certified bound (90%)")
        ax.plot(x, df.test_miss_rate_realised, "s-", color="#d9731a", label="realised test miss")
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x]); ax.set_title(title, fontsize=10)
        ax.set_xlabel("budget (% of entity-periods per period)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("miss rate"); axes[0].legend(fontsize=8)
    fig.suptitle("Budget curve: certified bound vs realised miss (temporal split)\n"
                 "Direct scorer, delta 0.10; calibrate on periods 12-15, evaluate on periods 16-19", fontsize=11, x=0.5, ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "Fig2_budget_fixed_vs_fraction.png", dpi=200, bbox_inches="tight")


def fig1():
    """Fig 1 from the committed gate CSV (no rerun of the gate)."""
    from src.validity_gate import plot
    plot(pd.read_csv(C.RESULTS / "validity_gate.csv"), OUT / "Fig1_validity_gate.png")


if __name__ == "__main__":
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    F = C.FIGURES
    if "--fig1-fig2" not in sys.argv:                 # default: copy the other figures too
        for src, dst in (("h1_typology_primary.png", "Fig3_h1_typology_primary.png"),
                         ("h2_synthetic_delay.png", "Fig4_h2_synthetic_delay.png"), ("h4_centerpiece.png", "Fig5_h4_centerpiece.png")):
            shutil.copy(F / src, OUT / dst)
    fig1()
    fig2()
    print(sorted(p.name for p in OUT.iterdir()))
