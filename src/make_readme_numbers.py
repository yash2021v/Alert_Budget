"""Print the README 'Key results' table (markdown) from the committed result CSVs. No number is typed by hand.
Usage: python -m src.make_readme_numbers   (reads results/, or results_quick/ with ALERTBUDGET_MODE=quick)
"""
import json

import numpy as np
import pandas as pd

from src import config as C


def read(name):
    return pd.read_csv(C.RESULTS / name)


def _fmt(v, floatfmt):
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() and abs(v) >= 100 else floatfmt.format(v)
    return str(v)


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(v, floatfmt) for v in r) + " |")
    return "\n".join(out)


def key_numbers() -> pd.DataFrame:
    rows = []
    primary = json.loads((C.RESULTS / "headline_scorer.json").read_text())["primary"]
    gate = read("validity_gate.csv")
    g = gate[gate.gating]
    syn = g[g.check == "synthetic"]
    cur = read("miss_rate_certified_vs_realised.csv")
    cp = cur[(cur.scorer == primary) & np.isclose(cur.delta, 0.10)]
    gap = cp.test_miss_rate_realised - cp.certified_bound
    rows.append(("Valid when exchangeable, broken by time (IBM, primary scorer, delta 0.10)",
                 f"{int(g['pass'].sum())}/{len(g)} gate configurations pass (synthetic coverage {syn.coverage.min():.3f}-{syn.coverage.max():.3f} vs nominal 0.90/0.95), "
                 f"but on the temporal split the bound covers the realised test miss at {int(cp.covered.sum())}/{len(cp)} budgets "
                 f"(realised minus bound {gap.min():+.3f} to {gap.max():+.3f})",
                 "results/validity_gate.csv; results/miss_rate_certified_vs_realised.csv"))
    h3 = read("h3_aggregation.csv")
    h = h3[np.isclose(h3.delta, 0.10) & (h3.K_frac == 0.01)].set_index("variant")
    rows.append(("Aggregation (max over transactions vs direct), K = 1% of period",
                 f"recall {h.loc['max_txn', 'recall_at_K_test']:.3f} vs {h.loc['direct', 'recall_at_K_test']:.3f}; exchangeable-control coverage "
                 f"{h3[h3.variant == 'max_txn'].control_coverage_vs_pooled.min():.3f}-{h3[h3.variant == 'max_txn'].control_coverage_vs_pooled.max():.3f} (valid)",
                 "results/h3_aggregation.csv (recall_at_K_test, control_coverage_vs_pooled)"))
    h1 = read("coverage_by_typology_primary.csv")
    u = h1[(h1.typology == "UNATTRIBUTED") & np.isclose(h1.delta, 0.10)]
    rows.append(("Group-conditional (H1, primary scorer, delta 0.10)",
                 f"UNATTRIBUTED miss {u.group_pooled_population_miss.min():.2f}-{u.group_pooled_population_miss.max():.2f} vs pooled bound "
                 f"{u.pooled_bound.min():.2f}-{u.pooled_bound.max():.2f}; pooled-bound exchangeable coverage {u.pooled_exchangeable_coverage.max():.2f}",
                 "results/coverage_by_typology_primary.csv"))
    h4 = read("h4_elliptic_summary.csv")
    e = h4[(h4.L == 2) & (h4.q == 0.10)].set_index("arm")
    rows.append(("Elliptic shutdown (L = 2, q = 10%): error frequency before -> from step 43",
                 "; ".join(f"{a} {e.loc[a, 'err_freq_before']:.2f} -> {e.loc[a, 'err_freq_after']:.2f}"
                           for a in ("static", "rolling", "aci", "aci_offset")) + f" (target 0.10; recovery lag at L = 2, q = 10%: " + ", ".join(f"{a} {e.loc[a, 'recovery_lag']}" for a in ("static", "rolling", "aci", "aci_offset")) + ")",
                 "results/h4_elliptic_summary.csv"))
    h2 = read("h2_synthetic_delay.csv")
    a = h2[h2.t == "all"].set_index(["d", "L"]).coverage
    rows.append(("Label delay (synthetic, delta 0.10)",
                 f"control d=0: {a[(0.0, 0)]:.3f}-{a[(0.0, 4)]:.3f}; slow drift d=0.01: L=0 {a[(0.01, 0)]:.3f}, L=2 {a[(0.01, 2)]:.3f}, L=4 {a[(0.01, 4)]:.3f}",
                 "results/h2_synthetic_delay.csv (coverage, t = all)"))
    return pd.DataFrame(rows, columns=["Result", "Number", "Source"])


if __name__ == "__main__":
    print(md_table(key_numbers()))
