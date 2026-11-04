"""H1 completion: how badly does the POOLED bound under-serve each group when time is removed?

For each (K, delta) the calibration and test per-positive miss flags are pooled (all groups) and re-split at random 1,000
times into a calibration-sized part and a remainder. The pooled bound comes from the calibration-sized part (all groups
mixed). Added to results/coverage_by_typology.csv:
  pooled_exchangeable_coverage          share of re-splits where pooled bound >= the group's miss rate in the pooled population
                                        (proxy for the group's true miss rate; no temporal drift involved)
  pooled_exchangeable_coverage_remainder  same, against the group's observed miss rate in the remainder (noisy, descriptive)
  group_pooled_population_miss          that proxy (group's miss rate over all calibration+test positives)
group_cal_miss (the group's calibration miss rate) is already in the file.
"""
import numpy as np
import pandas as pd

from src import certify as CE
from src import config as C
from src import scoring
from src.run_h1 import DELTAS, DRAWS, K_FRACS, flags_and_groups, load_table, suffix_of

if __name__ == "__main__":
    import sys
    primary = "--primary" in sys.argv
    rng = np.random.default_rng(C.SEED)
    path = C.RESULTS / f"coverage_by_typology{suffix_of(primary)}.csv"
    df = pd.read_csv(path).drop(columns=["pooled_exchangeable_coverage", "pooled_exchangeable_coverage_remainder",
                                         "group_pooled_population_miss"], errors="ignore")
    sc = load_table(primary)
    cal, test = sc[sc.split == "calibration"], sc[sc.split == "test"]
    avg = test.groupby("period").size().mean()
    new = []
    for f in K_FRACS:
        k = int(round(f * avg))
        mc, gc = flags_and_groups(cal, k)
        mt, gt = flags_and_groups(test, k)
        P, G = np.r_[mc, mt], np.r_[gc, gt]
        n_c = len(mc)
        proxy = {g: P[G == g].mean() for g in np.unique(G)}
        for d in DELTAS:
            hit = {g: 0 for g in proxy}
            hit_rem = {g: 0 for g in proxy}
            for _ in range(DRAWS):
                perm = rng.permutation(len(P))
                a, b = perm[:n_c], perm[n_c:]
                bound = CE.clopper_pearson_upper(int(P[a].sum()), n_c, d)
                for g in proxy:
                    hit[g] += proxy[g] <= bound
                    rem = P[b][G[b] == g]
                    hit_rem[g] += (rem.mean() <= bound) if len(rem) else 0
            for g in proxy:
                new.append(dict(typology=g, K=k, delta=d, pooled_exchangeable_coverage=hit[g] / DRAWS,
                                pooled_exchangeable_coverage_remainder=hit_rem[g] / DRAWS, group_pooled_population_miss=proxy[g]))
    df = df.merge(pd.DataFrame(new), on=["typology", "K", "delta"], how="left")
    df.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    s = df[(df.delta == 0.10) & (df.K_frac == 0.01)]
    print(s[["typology", "K", "n_cal_pos", "group_cal_miss", "group_pooled_population_miss", "pooled_bound", "group_test_miss",
             "pooled_covers_group", "pooled_exchangeable_coverage", "pooled_exchangeable_coverage_remainder"]].round(3).to_string(index=False))

    if primary:
        # h1_summary.md. Definitions fixed in advance: a group is "under-served by the pooled bound" in the exchangeable protocol
        # if pooled_exchangeable_coverage < 0.90 (nominal at delta 0.10), and on the temporal split if pooled_covers_group is False.
        def summarise(frame, label):
            d = frame[np.isclose(frame["delta"], 0.10)]
            out = []
            for g in ("UNATTRIBUTED", "BIPARTITE", "SCATTER-GATHER"):
                x = d[d.typology == g]
                out.append(f"{g}: exchangeable under-served at {int((x.pooled_exchangeable_coverage < 0.90).sum())} of {len(x)} budgets "
                           f"(coverage {x.pooled_exchangeable_coverage.min():.2f}-{x.pooled_exchangeable_coverage.max():.2f}), temporal split "
                           f"under-served at {int((~x.pooled_covers_group).sum())} of {len(x)}")
            return f"{label}: " + "; ".join(out) + "."
        direct = pd.read_csv(C.RESULTS / "coverage_by_typology.csv")
        lines = ["# H1 summary (delta 0.10, K = 0.5%, 1%, 2%, 5% of period size)",
                 "Definitions (fixed in advance): 'under-served in the exchangeable protocol' = pooled_exchangeable_coverage < 0.90; "
                 "'under-served on the temporal split' = the pooled bound is below the group's realised test miss rate.",
                 summarise(df, "PRIMARY scorer (max-over-transaction; post-hoc rerun)"),
                 summarise(direct, "DIRECT scorer (secondary)")]
        u = df[(np.isclose(df.delta, 0.10)) & (df.typology == "UNATTRIBUTED")]
        lines.append(f"UNATTRIBUTED positives have a calibration miss rate of {u.group_cal_miss.min():.2f}-{u.group_cal_miss.max():.2f} under the primary "
                     f"scorer against pooled bounds of {u.pooled_bound.min():.2f}-{u.pooled_bound.max():.2f}, so the pooled certificate "
                     "under-serves the group that no pattern file explains even when time order is removed.")
        (C.RESULTS / "h1_summary.md").write_text("\n\n".join(lines) + "\n")
        print("\n".join(lines))
