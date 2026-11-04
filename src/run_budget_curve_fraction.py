"""Fraction-of-volume budget arm: in each period t review K_t = ceil(q * n_t) entity-periods (n_t = that period's
entity-period count). Same grid of q, same certify-on-calibration / evaluate-on-test protocol as the fixed-count
curve, which remains the PRIMARY budget definition.

Writes results/miss_rate_fraction_budget.csv and results/volume_decomposition.md.
"""
import pandas as pd

from src import certify as CE
from src import config as C
from src import scoring
from src.run_budget_curve import DELTAS, K_FRACS

OUT = C.RESULTS / "miss_rate_fraction_budget.csv"


def build_fraction_curve(scores: pd.DataFrame) -> pd.DataFrame:
    cal = scores[scores["split"] == "calibration"]
    test = scores[scores["split"] == "test"]
    rows = []
    for q in K_FRACS:
        k_cal, k_test = CE.fraction_k(cal["period"], q), CE.fraction_k(test["period"], q)
        t_missed, t_pos = CE.missed_counts(test["margin"], test["label"], test["period"], k_test)
        for d in DELTAS:
            c = CE.certify(cal["margin"], cal["label"], cal["period"], k_cal, d)
            realised = t_missed / t_pos
            rows.append(dict(budget_type="fraction", K=int(round(k_test.mean())), K_frac=q, delta=d,
                             n_cal_pos=c["n_pos"], cal_miss_rate=c["point_miss_rate"], certified_bound=c["upper_bound"],
                             n_test_pos=t_pos, test_miss_rate_realised=realised,
                             covered=bool(realised <= c["upper_bound"]), recall_at_K_test=1 - realised))
    return pd.DataFrame(rows)


def write_md(fixed: pd.DataFrame, frac: pd.DataFrame):
    cov = {(t, d): int(df[(df.delta == d)].covered.sum()) for t, df in (("fixed", fixed), ("fraction", frac))
           for d in (0.10, 0.05)}
    n = fixed["K_frac"].nunique()
    f10, r10 = fixed[fixed.delta == 0.10], frac[frac.delta == 0.10]
    gap_fixed = (f10["test_miss_rate_realised"].to_numpy() - f10["certified_bound"].to_numpy())
    gap_frac = (r10["test_miss_rate_realised"].to_numpy() - r10["certified_bound"].to_numpy())
    lines = [
        "# Volume decomposition",
        f"Under the fixed-count budget (the primary definition) {cov[('fixed', 0.10)]} of {n} budgets are covered at delta 0.10 "
        f"and {cov[('fixed', 0.05)]} of {n} at delta 0.05; under the fraction-of-volume budget {cov[('fraction', 0.10)]} of {n} "
        f"and {cov[('fraction', 0.05)]} of {n}.",
        f"The realised-minus-bound gap at delta 0.10 averages {gap_fixed.mean():+.3f} (range {gap_fixed.min():+.3f} to "
        f"{gap_fixed.max():+.3f}) for fixed counts and {gap_frac.mean():+.3f} (range {gap_frac.min():+.3f} to "
        f"{gap_frac.max():+.3f}) for fractions.",
    ]
    if cov[("fraction", 0.10)] > cov[("fixed", 0.10)]:
        lines.append("Making the budget a share of each period's volume recovers coverage at some budgets, so part of the "
                     "fixed-count under-coverage is period-volume variation.")
    elif cov[("fraction", 0.10)] == cov[("fixed", 0.10)]:
        lines.append("Making the budget a share of each period's volume does not change how many budgets are covered, "
                     "so period-volume variation alone does not explain the under-coverage.")
    else:
        lines.append("Making the budget a share of each period's volume covers fewer budgets than fixed counts.")
    if gap_frac.mean() < gap_fixed.mean():
        lines.append("The gap shrinks under fractions, which is the volume part of the shift; whatever remains is score "
                     "or positive-mix drift between calibration and test.")
    else:
        lines.append("The gap does not shrink under fractions, so volume is not the main driver and the remaining gap "
                     "reflects score or positive-mix drift between calibration and test.")
    lines.append("Fixed-count stays the primary budget definition; the fraction arm is a diagnostic and was not used to "
                 "change it.")
    (C.RESULTS / "volume_decomposition.md").write_text("\n\n".join(lines) + "\n")


if __name__ == "__main__":
    sc = scoring.load_scores()
    frac = build_fraction_curve(sc)
    frac.to_csv(OUT, index=False)
    fixed = pd.read_csv(C.RESULTS / "miss_rate_certified_vs_realised.csv")
    if "scorer" in fixed.columns:                  # the fraction arm is run on the direct scorer: compare like with like
        fixed = fixed[fixed.scorer == "direct"]
    write_md(fixed, frac)
    print(frac.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(open(C.RESULTS / "volume_decomposition.md").read())
