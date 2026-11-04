# H4 summary (Elliptic, delay L = 2, q = 10% unless stated)

STATIC: the error indicator fires at a rate of 0.00 before step 43 and 0.71 from step 43 on (target 0.10), with mean bound 0.547 before and 0.542 after (max 0.542) and recovery lag none.

ROLLING: the error indicator fires at a rate of 0.12 before step 43 and 0.86 from step 43 on (target 0.10), with mean bound 0.410 before and 0.257 after (max 0.340) and recovery lag none.

ACI: the error indicator fires at a rate of 0.12 before step 43 and 0.86 from step 43 on (target 0.10), with mean bound 0.409 before and 0.276 after (max 0.409) and recovery lag none.

ACI's bound after the shutdown peaks at 0.409, so any coverage restored does not require a bound near 1.

Sensitivity of ACI to gamma (fixed values, not for selection): error frequency after step 43 is aci: 0.86, aci_g0.01: 0.86, aci_g0.10: 0.86.

Effect of the delay L at q = 10% (error frequency from step 43 on): L = 0: static 0.71, rolling 0.57, ACI 0.57; L = 2: static 0.71, rolling 0.86, ACI 0.86; L = 4: static 0.71, rolling 0.86, ACI 0.86.
