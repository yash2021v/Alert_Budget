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

- **Reviewed set / miss.** In each period the top K entity-periods by score are reviewed (K is an absolute count per period). A positive entity-period is *missed* if it is outside its period's top K. Miss rate (false omission rate) = missed positives / all positives. Score ties are broken by input order after a one-time random shuffle (`src/certify.py`).
- **Certificate.** From the calibration positives, `certify` returns the exact one-sided Clopper-Pearson upper bound on the true miss rate at confidence 1 - delta (primary delta = 0.10, also 0.05).
- **Validity condition.** The bound holds with probability at least 1 - delta only if positives seen at deployment are exchangeable with the calibration positives. Temporal drift and dependence between consecutive periods for the same account can break this; H2 and H4 test that later.
- **Validity gate.** `python -m src.validity_gate` simulates exchangeable data with a known truth (1,000 trials per configuration) and halves the real calibration positives; it writes `results/validity_gate.csv` and `results/figures/validity_gate.png` and prints PASS or FAIL (realised coverage must be at least nominal - 2 SE).
- **Budget curve.** `python -m src.run_budget_curve` certifies on the calibration split and compares with the realised miss rate on the test split, writing `results/miss_rate_certified_vs_realised.csv` and `results/figures/budget_curve.png`.
