"""OPTIONAL Step 5: IBM delay check. THIN SAMPLE (4 evaluation periods, 16-19) and SUPPLEMENTARY to H2 (synthetic) and H4 (Elliptic).

Primary scorer (results/headline_scorer.json), fixed-count K grid, delta = 0.10, binomial error indicator. For evaluation
period t and delay L the calibration pool is all labelled periods 12 .. t-L-1 (skipped if empty).
Writes results/ibm_delay_check.csv and results/ibm_delay_check.md.
"""
import json

import numpy as np
import pandas as pd

from src import certify as CE
from src import config as C
from src.coverage_monitor import error_frequency, error_indicator
from src.run_h3 import K_FRACS, variant_tables

DELTA, LS, POOL_START = 0.10, [0, 2], 12


def main():
    primary = json.loads((C.RESULTS / "headline_scorer.json").read_text())["primary"]
    sc = variant_tables()[primary]
    sc = sc[sc.split.isin(["calibration", "test"])]
    avg = sc[sc.split == "test"].groupby("period").size().mean()
    rows = []
    for f in K_FRACS:
        k = int(round(f * avg))
        flags = CE.missed_flags(sc["score"].to_numpy(), sc["label"].to_numpy(), sc["period"].to_numpy(), k)
        per = sc.loc[sc["label"] == 1, "period"].to_numpy()
        cnt = {p: (int(flags[per == p].sum()), int((per == p).sum())) for p in range(POOL_START, 20)}
        for L in LS:
            for t in range(16, 20):
                lo, hi = POOL_START, t - L - 1
                if hi < lo:
                    continue
                kp = sum(cnt[p][0] for p in range(lo, hi + 1))
                npool = sum(cnt[p][1] for p in range(lo, hi + 1))
                b = CE.clopper_pearson_upper(kp, npool, DELTA)
                kt, nt = cnt[t]
                rows.append(dict(scorer=primary, K=k, K_frac=f, L=L, period=t, pool_periods=f"{lo}-{hi}", pool_positives=npool,
                                 n_pos=nt, misses=kt, bound=b, err=error_indicator(kt, nt, b),
                                 realised_miss=kt / nt, plain_covered_noisy=bool(kt / nt <= b)))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "ibm_delay_check.csv", index=False)
    s = df.groupby("L").apply(lambda g: pd.Series({"error_frequency": error_frequency(g.err, g.n_pos),
                                                   "plain_covered_share": g.plain_covered_noisy.mean(), "cells": len(g)}),
                              include_groups=False)
    by_k = df.groupby(["L", "K"]).err.mean().unstack(0)
    md = ["# IBM delay check (supplementary, thin sample)", "",
          "NOTE: only 4 evaluation periods (16-19) per delay, so each cell is a handful of binomial draws; this is supplementary to "
          "the synthetic H2 dose-response and the Elliptic H4 arms and should not be read as evidence on its own.", "",
          f"Scorer: {primary}. Delta 0.10, error indicator with delta_err 0.10 (target error frequency 0.10).", "",
          s.round(3).to_string(), "",
          "Error frequency by K and delay:", "", by_k.round(2).to_string()]
    (C.RESULTS / "ibm_delay_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
