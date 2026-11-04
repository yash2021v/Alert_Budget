"""H2 synthetic dose-response: how stale can label feedback be before the certificate stops covering?

Design (fixed in advance): Day 2 synthetic generator (negatives N(0,1) stationary; positives N(mu_t, 1)), 30 periods of
162,000 entities, positive rate 0.26%, K = 1% of entities (1,620) per period. Positives drift toward the negatives:
mu_t = max(0, MU - d * max(0, t - 19)) with MU giving AUC 0.85 at t <= 19; i.e. the drift starts at period 20, the first
evaluated period, and cannot pass the negatives' mean (clamp at 0). d in {0, 0.05, 0.10, 0.20}; d = 0 is the control.
At period t with delay L the certificate uses only the positives of periods t-L-4 .. t-L-1 (W = 4), delta = 0.10, and is
compared with the TRUE miss rate at period t (computed analytically from the mixture, a noise-free large-sample
reference). 1,000 trials; t in the last 10 periods (20-29).

Writes results/h2_synthetic_delay.csv, results/figures/h2_synthetic_delay.png, results/h2_summary.md.
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import certify as CE
from src import config as C
from src import validity_gate as G

N_PERIODS, EVAL = 30, range(20, 30)
DRIFT_START = 19
W, DELTA, TRIALS = 4, 0.10, C.MC
DS_ORIG, DS_EXT = [0.0, 0.05, 0.10, 0.20], [0.005, 0.01, 0.02]    # DS_EXT: post-hoc finer grid added after Day 4 (floor effect)
DS, LS = DS_ORIG + DS_EXT, [0, 2, 4]
K = int(round(0.01 * G.N_ENT))


def mu_at(t, d):
    return max(0.0, G.MU - d * max(0, t - DRIFT_START))


def true_miss(mu, pi=G.POS_RATE, n=G.N_ENT, k=K):
    """P(positive is outside the top-K): top-K threshold tau solves (1-pi) sf(tau) + pi sf(tau - mu) = K/n."""
    f = lambda tau: (1 - pi) * norm.sf(tau) + pi * norm.sf(tau - mu) - k / n
    return float(norm.cdf(brentq(f, -10, 10) - mu))


def main(extended=False):
    """extended=False: the original grid (writes the CSV). extended=True: POST-HOC finer grid d in {0.005, 0.01, 0.02},
    appended to the CSV (same protocol)."""
    ds = DS_EXT if extended else DS_ORIG
    grid = "extended_post_hoc" if extended else "original"
    if extended:                                       # no positive-score clipping at these d
        for d in ds:
            assert 30 * d <= 0.6 + 1e-12 and G.MU - 30 * d > 0, f"clipping would occur at d={d}"
    rng = np.random.default_rng(C.SEED + (1 if extended else 0))
    first = min(EVAL) - max(LS) - W
    truth = {(d, t): true_miss(mu_at(t, d)) for d in ds for t in EVAL}
    print(f"analytic true miss rate, d=0: {true_miss(mu_at(20, 0.0)):.4f} (Day 2 simulated reference at K=1,620: 0.8101)")
    cov = {(d, L): np.zeros((TRIALS, len(EVAL)), bool) for d in ds for L in LS}
    bnd = {(d, L): np.zeros((TRIALS, len(EVAL))) for d in ds for L in LS}
    for d in ds:
        for tr in range(TRIALS):
            miss, npos = {}, {}
            for p in range(first, max(EVAL)):
                neg, pos = _draw(rng, mu_at(p, d))
                miss[p], npos[p] = G.fast_missed(neg, pos, [K])[0], len(pos)
            for L in LS:
                for j, t in enumerate(EVAL):
                    ps = range(t - L - W, t - L)
                    b = CE.clopper_pearson_upper(sum(miss[p] for p in ps), sum(npos[p] for p in ps), DELTA)
                    bnd[(d, L)][tr, j] = b
                    cov[(d, L)][tr, j] = truth[(d, t)] <= b
        print(f"d={d} done", flush=True)
    rows = []
    for d in ds:
        for L in LS:
            for j, t in enumerate(EVAL):
                c = cov[(d, L)][:, j].mean()
                rows.append(dict(d=d, L=L, t=t, coverage=c, se=np.sqrt(c * (1 - c) / TRIALS),
                                 mean_bound=bnd[(d, L)][:, j].mean(), mean_true_miss=truth[(d, t)], trials=TRIALS, grid=grid))
            per_trial = cov[(d, L)].mean(axis=1)
            rows.append(dict(d=d, L=L, t="all", coverage=per_trial.mean(), se=per_trial.std(ddof=1) / np.sqrt(TRIALS),
                             mean_bound=bnd[(d, L)].mean(), mean_true_miss=np.mean([truth[(d, t)] for t in EVAL]),
                             trials=TRIALS, grid=grid))
    new = pd.DataFrame(rows)
    path = C.RESULTS / "h2_synthetic_delay.csv"
    if extended:
        old = pd.read_csv(path)
        old = old[old["grid"] == "original"]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df.to_csv(path, index=False)
    df = pd.read_csv(path)
    plot(df)
    write_md(df)
    a = df[df.t == "all"]
    print(a.round(4).to_string(index=False))
    print(open(C.RESULTS / "h2_summary.md").read())


def _draw(rng, mu):
    n_pos = rng.binomial(G.N_ENT, G.POS_RATE)
    return rng.standard_normal(G.N_ENT - n_pos), rng.standard_normal(n_pos) + mu


def plot(df):
    a = df[df.t == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for d in sorted(a.d.unique()):
        s = a[a.d == d].sort_values("L")
        ext = (s.grid == "extended_post_hoc").any()
        ax.errorbar(s.L, s.coverage, yerr=2 * s.se, marker="o", mfc="white" if ext else None, ls="--" if ext else "-",
                    capsize=3, label=f"d = {d}" + (" (post-hoc)" if ext else ""))
    ax.axhline(1 - DELTA, color="k", ls=":", label="nominal 0.90")
    ax.set_xticks(LS); ax.set_xlabel("label delay L (periods)"); ax.set_ylabel("coverage of the true miss rate (periods 20-29)")
    ax.set_title("Coverage vs delay, one line per drift speed d"); ax.legend(fontsize=7); ax.set_ylim(-0.02, 1.02)
    ax = axes[1]
    for L in LS:
        s = a[a.L == L].sort_values("d")
        x = s.d.replace(0.0, 0.002)
        ax.errorbar(x, s.coverage, yerr=2 * s.se, marker="o", capsize=3, label=f"L = {L}")
        pe = s[s.grid == "extended_post_hoc"]
        ax.scatter(pe.d, pe.coverage, s=90, facecolors="none", edgecolors="k", zorder=3)
    ax.set_xscale("log"); ax.axhline(1 - DELTA, color="k", ls=":")
    ax.set_xlabel("drift speed d (SD per period; d = 0 drawn at 0.002); circled = post-hoc grid")
    ax.set_title("Coverage vs drift speed, one line per delay"); ax.legend(fontsize=7); ax.set_ylim(-0.02, 1.02)
    fig.suptitle("H2 synthetic dose-response (error bars = 2 SE; dashed/hollow = post-hoc extended grid)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "h2_synthetic_delay.png", dpi=130)


def write_md(df):
    a = df[df.t == "all"].set_index(["d", "L"])
    c = lambda d, L: a.loc[(d, L), "coverage"]
    se = lambda d, L: a.loc[(d, L), "se"]
    ds = sorted(a.index.get_level_values(0).unique())
    ctrl = [c(0.0, L) for L in LS]
    L_ = [f"Control (d = 0): coverage is {min(ctrl):.3f}-{max(ctrl):.3f} at delays 0, 2 and 4 against a nominal 0.90, so delay alone does not "
          "break the certificate when positives are exchangeable."]
    small = [d for d in ds if 0 < d <= 0.02]
    L_.append("Finer drift grid (added post hoc because d >= 0.05 sits on the floor): coverage at L = 0 / 2 / 4 is " +
              "; ".join(f"d = {d}: {c(d, 0):.3f} / {c(d, 2):.3f} / {c(d, 4):.3f}" for d in small) + ".")
    # rule fixed before looking at the extended results: a delay effect is claimed only if, at some small d, coverage at L = 0
    # exceeds coverage at L = 4 by more than 3 combined standard errors AND coverage at L = 0 is not itself at the floor (< 0.05)
    eff = [d for d in small if c(d, 0) - c(d, 4) > 3 * np.hypot(se(d, 0), se(d, 4)) and c(d, 0) > 0.05]
    if eff:
        L_.append("Delay effect: at d = " + ", ".join(str(d) for d in eff) + " coverage at L = 0 exceeds coverage at L = 4 by more than "
                  "3 combined standard errors (" + "; ".join(f"d = {d}: {c(d, 0) - c(d, 4):+.3f}" for d in eff) + "), so staler feedback does "
                  "cost coverage once drift is slow enough for coverage to be measurable.")
    else:
        L_.append("Delay effect: at no small d does coverage at L = 0 exceed coverage at L = 4 by more than 3 combined standard errors while "
                  "L = 0 is off the floor, so these data show NO clear effect of the delay.")
    floor = [d for d in ds if d > 0 and c(d, 0) < 0.05]
    L_.append(("Coverage is already below 0.05 at L = 0 for d = " + ", ".join(str(d) for d in floor) + ", so the dose-response in delay "
               "cannot be seen there (floor effect).") if floor else "Coverage at L = 0 stays above 0.05 for all small d.")
    L_.append(f"With about {int(round(4 * G.POS_RATE * G.N_ENT))} positives in the pool the interval is roughly +-0.02 wide, so the certificate "
              "tolerates only drift that moves the true miss rate by less than about that much over the pool-plus-delay window; "
              "larger drifts need monitoring and adaptation (H4) rather than a staleness limit.")
    (C.RESULTS / "h2_summary.md").write_text("# H2 synthetic summary\n\n" + "\n\n".join(L_) + "\n")


if __name__ == "__main__":
    import sys
    main(extended="--extended" in sys.argv)
