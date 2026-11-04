"""Validity gate for the certificate: does the Clopper-Pearson bound achieve its nominal coverage?

1. Synthetic exchangeable data with a known truth (normal scores, AUC ~0.85, 20 periods of ~162k entities,
   positive rate 0.26%, plus a small-sample regime with ~50 calibration positives).
2. Real calibration-split scores, with the positives randomly halved (forced exchangeability on the real
   score distribution): certify on one half A. GATING check: the truth is the miss rate of the whole calibration
   population the halves were drawn from. INFORMATIONAL check (not gating): the truth is the observed miss rate
   of the other half B; B's own sampling noise (n/2 positives) makes coverage of even a perfectly valid bound
   top out near Phi(z_delta / sqrt 2) (~0.82 at delta=0.10, ~0.88 at 0.05), so it cannot meet nominal.

GATE: realised coverage >= nominal - 2 binomial standard errors in every configuration.
Writes results/validity_gate.csv and results/figures/validity_gate.png.
"""
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import certify as CE

N_ENT = 162_000                       # entities per period (real panel: 3,245,076 / 20)
N_PERIODS = 20
POS_RATE = 0.0026
SMALL_POS_TOTAL = 50                  # small-sample regime: ~50 calibration positives in total
AUC = 0.85
MU = np.sqrt(2) * norm.ppf(AUC)       # pos ~ N(MU, 1), neg ~ N(0, 1)  ->  AUC = Phi(MU / sqrt 2)
K_FRACS = [0.0025, 0.01, 0.05]
DELTAS = [0.10, 0.05]
TRIALS = 1000
TRUTH_PERIODS = 20000                 # fresh periods for the "true" miss rate (2,000 was too noisy: its error
                                      # is ~10% of the estimator SD and shifted coverage by a few points)


def draw_period(rng, n_ent, pi):
    n_pos = rng.binomial(n_ent, pi)
    return rng.standard_normal(n_ent - n_pos), rng.standard_normal(n_pos) + MU


def fast_missed(neg, pos, ks):
    """Missed positives for each K in one period (continuous scores -> no ties). The K-th largest score is the
    review threshold; a positive is missed iff its score is below it."""
    allv = np.concatenate([neg, pos])
    kmax = max(ks)
    top = np.partition(allv, len(allv) - kmax)[len(allv) - kmax:]
    top.sort()
    return [int((pos < top[kmax - k]).sum()) for k in ks]


def crosscheck(rng):
    """The fast path must agree with certify.missed_counts on the same (shuffled) data."""
    ks = [40, 160]
    for _ in range(3):
        neg, pos = draw_period(rng, 8000, 0.01)
        scores = np.concatenate([neg, pos])
        labels = np.r_[np.zeros(len(neg), int), np.ones(len(pos), int)]
        perm = rng.permutation(len(scores))
        got = fast_missed(neg, pos, ks)
        for k, g in zip(ks, got):
            assert CE.missed_counts(scores[perm], labels[perm], np.zeros(len(scores), int), k)[0] == g


def synthetic(rng, regime, pi):
    ks = [int(round(f * N_ENT)) for f in K_FRACS]
    miss = np.zeros(len(ks))
    npos = 0
    for _ in range(TRUTH_PERIODS):
        neg, pos = draw_period(rng, N_ENT, pi)
        miss += fast_missed(neg, pos, ks)
        npos += len(pos)
    truth = miss / npos
    cov = {(k, d): 0 for k in ks for d in DELTAS}
    pos_counts = []
    for _ in range(TRIALS):
        m = np.zeros(len(ks), dtype=int)
        n = 0
        for _ in range(N_PERIODS):
            neg, pos = draw_period(rng, N_ENT, pi)
            m += fast_missed(neg, pos, ks)
            n += len(pos)
        pos_counts.append(n)
        for i, k in enumerate(ks):
            for d in DELTAS:
                cov[(k, d)] += truth[i] <= CE.clopper_pearson_upper(int(m[i]), n, d)
    rows = []
    for i, (f, k) in enumerate(zip(K_FRACS, ks)):
        for d in DELTAS:
            rows.append(dict(check="synthetic", regime=regime, K_frac=f, K=k, delta=d, trials=TRIALS,
                             mean_n_pos=float(np.mean(pos_counts)), truth_miss_rate=float(truth[i]),
                             coverage=cov[(k, d)] / TRIALS))
    return rows


def real_halves(rng):
    sc = pd.read_parquet(C.PROCESSED / "scores.parquet")
    cal = sc[sc["split"] == "calibration"].sample(frac=1.0, random_state=C.SEED)   # shuffle: tie order is random
    avg = cal.groupby("period").size().mean()
    rows = []
    for f in K_FRACS:
        k = int(round(f * avg))
        m = CE.missed_flags(cal["score"].to_numpy(), cal["label"].to_numpy(), cal["period"].to_numpy(), k)
        n = len(m)
        pop = {d: 0 for d in DELTAS}
        other = {d: 0 for d in DELTAS}
        for _ in range(TRIALS):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            for d in DELTAS:
                ub = CE.clopper_pearson_upper(int(m[a].sum()), len(a), d)
                pop[d] += m.mean() <= ub
                other[d] += m[b].mean() <= ub
        for d in DELTAS:
            base = dict(K_frac=f, K=k, delta=d, trials=TRIALS, mean_n_pos=float(n), truth_miss_rate=float(m.mean()))
            rows.append(dict(check="real_halves_population", regime="calibration_split", gating=True,
                             coverage=pop[d] / TRIALS, **base))
            rows.append(dict(check="real_halves_other_half", regime="calibration_split", gating=False,
                             coverage=other[d] / TRIALS, expected_ceiling=float(norm.cdf(norm.ppf(1 - d) / np.sqrt(2))),
                             **base))
    return rows


def main():
    rng = np.random.default_rng(C.SEED)
    t = time.time()
    crosscheck(rng)
    print("fast simulation agrees with certify.missed_counts")
    rows = synthetic(rng, "main", POS_RATE)
    print(f"main regime done ({time.time() - t:.0f}s)", flush=True)
    rows += synthetic(rng, "small_sample", SMALL_POS_TOTAL / (N_PERIODS * N_ENT))
    print(f"small-sample regime done ({time.time() - t:.0f}s)", flush=True)
    rows += real_halves(rng)
    df = pd.DataFrame(rows)
    df["gating"] = df["gating"].fillna(True).astype(bool)
    df["nominal"] = 1 - df["delta"]
    df["se"] = np.sqrt(df["nominal"] * (1 - df["nominal"]) / df["trials"])
    df["threshold"] = df["nominal"] - 2 * df["se"]
    df["pass"] = df["coverage"] >= df["threshold"]
    gate = df[df["gating"]]
    C.RESULTS.mkdir(exist_ok=True)
    df.to_csv(C.RESULTS / "validity_gate.csv", index=False)
    plot(df)
    show = df.assign(config=df["check"] + "/" + df["regime"] + " K=" + df["K_frac"].map("{:.2%}".format)
                     + " d=" + df["delta"].astype(str))
    print(show[["config", "K", "mean_n_pos", "truth_miss_rate", "coverage", "nominal", "se", "pass"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    info = df[~df["gating"]]
    print("\nInformational (not gating) real_halves_other_half rows: coverage vs theoretical ceiling for a valid bound")
    print(info[["K", "delta", "coverage", "expected_ceiling"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if gate["pass"].all():
        print(f"\nVALIDITY GATE: PASS ({len(gate)} gating configurations at or above nominal - 2 SE)")
    else:
        print("\nVALIDITY GATE: FAIL")
        print(gate[~gate["pass"]].to_string(index=False))
        sys.exit(1)


def plot(df):
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = [f"{r.check}/{r.regime}  K={r.K_frac:.2%}  δ={r.delta}" for r in df.itertuples()]
    y = np.arange(len(df))[::-1]
    ax.hlines(y, df["threshold"], df["nominal"], color="#bbbbbb", lw=6, label="nominal − 2 SE … nominal")
    ax.scatter(df["coverage"], y, c=np.where(~df["gating"], "#888888", np.where(df["pass"], "#1b7f3b", "#c0392b")), zorder=3, label="realised coverage")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("coverage of true miss rate by certified bound")
    ax.set_title("Validity gate: realised vs nominal coverage (gray = informational, not gating)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "validity_gate.png", dpi=130)


if __name__ == "__main__":
    main()
