# Elliptic spike (H4)

Budget type: FRACTION of labelled nodes per step (K_t = ceil(q * labelled nodes in step t), q = 10% and 20%), because step sizes vary from about 200 to 2,200 labelled nodes and a fixed count would review very different shares.

At q = 10% the bound from steps 25-34 is 0.542 (calibration miss rate 0.524); 8 of 8 evaluation steps before 43 are covered versus 1 of 7 from 43 on, and the mean realised miss is 0.179 before versus 0.781 after.

q = 10%: coverage VISIBLY changes at the shutdown (covered share drops by 86%, rule: at least 25 points).

Exchangeable control (steps 25-49 pooled, 1000 random re-splits): coverage of the pooled population miss rate 0.971 (nominal 0.90), so the bound itself is valid when time is removed and any under-coverage above is drift rather than a code error.

At q = 20% the bound from steps 25-34 is 0.190 (calibration miss rate 0.175); 7 of 8 evaluation steps before 43 are covered versus 1 of 7 from 43 on, and the mean realised miss is 0.082 before versus 0.485 after.

q = 20%: coverage VISIBLY changes at the shutdown (covered share drops by 73%, rule: at least 25 points).

Exchangeable control (steps 25-49 pooled, 1000 random re-splits): coverage of the pooled population miss rate 0.970 (nominal 0.90), so the bound itself is valid when time is removed and any under-coverage above is drift rather than a code error.
