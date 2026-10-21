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
- **Features.** Computed only from the 3 periods strictly before the entity-period (tested for leakage).
- **pandas pin.** `pandas>=2.2,<3` (currently 2.3.3).
