# Shift diagnostic

Certifying on the calibration split and checking the test split (time order kept), the bound is covered at 0 of 7 budgets at delta 0.10 and 0 of 7 at delta 0.05.

With the time order removed (pooled calibration and test flags, re-split at random 1000 times), coverage of the pooled population rate is 0.962-0.973 at delta 0.1 (nominal 0.90), and coverage of the remainder's observed rate is 0.789-0.825 (that comparison cannot reach nominal by construction).

With the time order removed (pooled calibration and test flags, re-split at random 1000 times), coverage of the pooled population rate is 0.989-0.996 at delta 0.05 (nominal 0.95), and coverage of the remainder's observed rate is 0.854-0.906 (that comparison cannot reach nominal by construction).

Per period, the positive rate is 0.296% in calibration periods 12-15 versus 0.326% in test periods 16-19, and the miss rate at K=1552 averages 0.607 versus 0.670.

Period volume also differs: calibration periods hold 141,364-156,391 entity-periods versus 73,893-232,090 in test, and because K is a fixed absolute count a busy period reviews a smaller share of its entities, so the miss rate moves with volume.

Coverage returns to nominal once the time order is removed while the temporal split under-covers, which points to a shift between the calibration and test periods rather than a bug in the certification code.

This control cannot tell score drift from a change in the mix of positives; it only shows that the miss flags are not exchangeable across time.
