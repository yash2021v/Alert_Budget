# H3 summary

The guarantee survives aggregation under the exchangeable control: coverage of the pooled population miss rate is 0.962-0.996 for the direct score and 0.964-0.995 for the max-over-transaction score, at or above nominal in both.

On the real temporal split (certify on calibration, evaluate on test, delta 0.10) the direct score is covered at 0 of 7 budgets and the max score at 0 of 7, so drift is still separate from validity.

Recall on test changes by +0.174 on average (range +0.092 to +0.265) when moving from the direct to the max score, i.e. the max score is better at 7 of 7 budgets.

At a fixed K the certified bound differs by -0.190 on average (max minus direct); the interval width is the same because both use the same 1750 calibration positives, so a tighter guarantee comes from a lower miss rate, not a narrower interval.
