# AlertBudget

Budget-aware AML alert triage with statistically certified miss rates

## How to run

```bash
source .venv/bin/activate
python -m src.data && python -m src.panel && python -m src.split
python -m src.features && python -m src.train && python -m src.elliptic
python -m pytest -q
streamlit run app/app.py
```

## Data and modelling choices

- **Entity-period.** Entity = account key (bank id as integer + account). A row exists for each account that sends or receives at least one transaction in a period. Periods are 12-hour blocks (the regular-volume window has only 10 days, fewer than 14), giving 20 periods. Label = 1 if any laundering transaction involves the account in that period, as sender or receiver.
- **Window.** The IBM HI-Small generator stops emitting normal traffic after 2022-09-10 (under 500 transactions per day versus 200k-1.1M) while laundering rows keep trickling in, so only transactions before 2022-09-11 are used (`ACTIVE_END` in `src/config.py`). This drops 655 of 5,177 laundering rows.
- **Typology.** Used for grouping and reporting only, never as a feature. For a positive entity-period it is the most frequent typology among the pattern attempts the account takes part in during that period (ties: earliest attempt). `UNATTRIBUTED` marks positives whose laundering transactions belong to no attempt in `HI-Small_Patterns.txt`; about 46% of positive entity-periods.
- **Split.** Strictly temporal by period order: train periods 0-11, calibration 12-15, test 16-19, never shuffled. The largest connected component holds 72% of accounts (under the 90% threshold) but spans every period, so keeping whole components in one split cannot coexist with an order-based split; accounts therefore overlap across splits (71% appear in more than one).
- **Features (v2, current).** For entity-period t: features from all transactions in period t (the review happens after t closes, so they are legitimately available) plus the same set over the trailing 3 periods before t, a `has_history` flag, and current/lookback ratios (transaction count, out-degree, total amount). Lookback features are NaN, not zero, when there is no history. Nothing after the end of period t is used, and the label and typology are never read (the feature code drops them on entry; tests cover future deletion, label shuffling and a hand-built graph). Amounts are log1p-transformed and standardised per currency with train-split statistics only.
- **Ablation (v1).** v1 used only the 3 periods before t (lookback-only) and reached test ROC-AUC 0.61. Its metrics are kept in `results/day1_model_metrics_v1_lookback_only.json` and its model in `results/lgbm_v1_lookback_only.joblib` (git-ignored) for the report.
- **pandas pin.** `pandas>=2.2,<3` (currently 2.3.3).

## Certification

- **Reviewed set / miss.** In each period the top K entity-periods by score are reviewed (K is an absolute count per period). A positive entity-period is *missed* if it is outside its period's top K. Miss rate (false omission rate) = missed positives / all positives. Entities are ranked by the frozen model's raw margin (log-odds, `margin` in `scores.parquet`), not by probability; residual exact ties are broken by one seeded shuffle of the table (`src/scoring.py`) and then by input position (`src/certify.py`).
- **Certificate.** From the calibration positives, `certify` returns the exact one-sided Clopper-Pearson upper bound on the true miss rate at confidence 1 - delta (primary delta = 0.10, also 0.05).
- **Validity condition.** The bound holds with probability at least 1 - delta only if positives seen at deployment are exchangeable with the calibration positives. Temporal drift and dependence between consecutive periods for the same account can break this; H2 and H4 test that later.
- **Validity gate.** `python -m src.validity_gate` simulates exchangeable data with a known truth (1,000 trials per configuration) and halves the real calibration positives; it writes `results/validity_gate.csv` and `results/figures/validity_gate.png` and prints PASS or FAIL (realised coverage must be at least nominal - 2 SE).
- **Budget curve.** `python -m src.run_budget_curve` certifies on the calibration split and compares with the realised miss rate on the test split, writing `results/miss_rate_certified_vs_realised.csv` and `results/figures/budget_curve.png`.

### Ranking and Day 1 recall correction

Probabilities from the frozen model saturate at exactly 1.0 for 4.6% of test rows, so top-K by probability fell inside a block of thousands of ties and the Day 1 recall figures depended on table order (tie-broken by table order they were optimistic compared with random tie-breaking). Ranking by the raw margin removes this: the tie block at each cutoff is only 1-45 rows. Recomputed with margin and random tie-breaking, test recall is 2.0%, 5.0%, 13.3%, 32.5% and 48.0% at K = 155, 388, 776, 1552, 3104 (ROC-AUC 0.865, PR-AUC 0.046). The earlier file is kept as `results/day1_model_metrics_v2_table_order_ties.json`.

### Validity gate specification

The gate specification was corrected after a first run, because comparing a bound with the observed rate of another sample cannot reach nominal coverage even for a valid bound (ceiling about 0.82 at delta 0.10 and 0.88 at 0.05). The corrected gate compares against the true or whole-population miss rate: synthetic data against a 20,000-period reference, and real scores by certifying on a random 10% subsample of calibration positives against the miss rate of all calibration positives. The half-split rows are kept as non-gating (`gating = false` in `results/validity_gate.csv`, with `expected_coverage_theory`). The gate was not changed after its results were seen.

## Day 3: fraction budget, H1, H3, Elliptic spike

- **Fraction-of-volume budget (diagnostic).** `src/run_budget_curve_fraction.py` reviews K_t = ceil(q * n_t) entity-periods in period t (n_t = that period's entity-period count), with the same grid of q, deltas and calibrate-then-test protocol. Fixed-count K remains the primary budget definition; both CSVs carry a `budget_type` column and `results/volume_decomposition.md` compares them.
- **H1: group-conditional certificates (`src/run_h1.py`).** Typology is known only for positives, so per-typology thresholds cannot be built (a threshold needs a typology for every entity, negatives included). The reviewed set is the usual top-K by margin per period; each typology group (8 plus UNATTRIBUTED) is certified with Clopper-Pearson on its own calibration positives and compared with its realised test miss rate, and the pooled bound is checked against each group. An exchangeable protocol (pool the group's calibration and test positives, 1,000 random re-splits) separates sampling variation from temporal drift. Groups with fewer than 30 calibration positives are flagged `low_n`, not dropped.
- **H3 (reduced): direct vs max-over-transaction (`src/transaction_model.py`, `src/run_h3.py`).** The learned aggregator was cut under the schedule fallback; only the simple aggregation is compared. A transaction-level LightGBM (trained on train-split transactions: all positives plus 10% of negatives; the transaction's own attributes plus both endpoints' lookback-only features; one fixed hyperparameter set) is aggregated to the entity-period by taking the maximum transaction margin, and the same entity-level certificate is applied with an exchangeable control.
- **Elliptic spike (H4, `src/elliptic_spike.py`).** Labelled nodes only, native 165 features, train steps 1-24, calibration 25-34, evaluation 35-49, fixed-parameter LightGBM ranked by margin, fraction budget K_t = ceil(q * labelled nodes in step t) at q = 10% and 20% because step sizes vary widely. Per-step coverage is checked against the bound from the calibration steps, with the step-43 shutdown marked, plus an exchangeable control on steps 25-49. No ACI and no tuning.

## Day 4: headline scorer

The primary IBM scorer was chosen on the calibration split only (mean recall over the K grid; see `results/headline_scorer.md`): the **max-over-transaction** score (frozen transaction LightGBM, maximum margin over the entity's transactions in the period). The entity-level direct score stays as the secondary comparison. `results/miss_rate_certified_vs_realised.csv` now has a `scorer` column (primary rows first) and the dashboard defaults to the primary scorer with the other as a toggle. The transaction model's features pass the same leakage tests as the entity features (`tests/test_txn_leakage.py`): they do not change when later transactions are deleted or the label is shuffled or removed. Earlier diagnostics (shift diagnostic, fraction-budget arm, H1) were run on the direct scorer.

### Day 4: error indicator, H2, H4 arms, results freeze

- **Error indicator** (`src/coverage_monitor.py`): err = 1 if misses k > binom.ppf(0.9, n, bound). A valid bound fires with probability at most 0.10 at any n; the plain "realised <= bound" flag is kept only as a noisy descriptive column.
- **H2 synthetic** (`src/h2_synthetic.py`): positives drift toward negatives from period 20; certificate uses the 4 most recent labelled periods at delay L in {0, 2, 4}; d = 0 is the control.
- **H4 arms** (`src/h4_elliptic.py`): STATIC, ROLLING (5-step pool) and ACI (gamma 0.05; 0.01 and 0.10 as fixed sensitivity checks) on Elliptic under delayed feedback, fraction budgets of 10% and 20%.
- **H1 completion** (`src/h1_complete.py`): adds the pooled-bound coverage under exchangeable re-splits per group. H1 was run on the direct (secondary) scorer.
- **IBM delay check** (`src/ibm_delay_check.py`): thin supplementary check on periods 16-19.
- Everything under `results/` at the git tag `results-frozen` is final; no new experiments after that tag.
