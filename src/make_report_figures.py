"""Copy the key figures to report/figures as Fig1..Fig5 (Fig2 is generated here: fixed-count vs fraction budget)."""
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
    fig.suptitle("Direct scorer, calibrate on periods 12-15, evaluate on periods 16-19: bound is below the realised miss at every budget under both definitions")
    fig.tight_layout()
    fig.savefig(OUT / "Fig2_budget_fixed_vs_fraction.png", dpi=130)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    F = C.FIGURES
    for src, dst in (("validity_gate.png", "Fig1_validity_gate.png"), ("h1_typology_primary.png", "Fig3_h1_typology_primary.png"),
                     ("h2_synthetic_delay.png", "Fig4_h2_synthetic_delay.png"), ("h4_centerpiece.png", "Fig5_h4_centerpiece.png")):
        shutil.copy(F / src, OUT / dst)
    fig2()
    print(sorted(p.name for p in OUT.iterdir()))
