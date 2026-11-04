# H4 summary (Elliptic, delay L = 2, q = 10% unless stated)

STATIC: the error indicator fires at a rate of 0.00 before step 43 and 0.71 from step 43 on (target 0.10), with mean bound 0.547 before and 0.542 after (max 0.542) and recovery lag none.

ROLLING: the error indicator fires at a rate of 0.12 before step 43 and 0.86 from step 43 on (target 0.10), with mean bound 0.410 before and 0.257 after (max 0.340) and recovery lag none.

ACI: the error indicator fires at a rate of 0.12 before step 43 and 0.86 from step 43 on (target 0.10), with mean bound 0.409 before and 0.276 after (max 0.409) and recovery lag none.

ACI's bound after the shutdown peaks at 0.409, so any coverage restored does not require a bound near 1.

Sensitivity of ACI to gamma (fixed values, not for selection): error frequency after step 43 is aci: 0.86, aci_g0.01: 0.86, aci_g0.10: 0.86.

Effect of the delay L at q = 10% (error frequency from step 43 on): L = 0: static 0.71, rolling 0.57, ACI 0.57, ACI-offset 0.57; L = 2: static 0.71, rolling 0.86, ACI 0.86, ACI-offset 0.86; L = 4: static 0.71, rolling 0.86, ACI 0.86, ACI-offset 0.86.

Why delta has little authority (results/bound_vs_delta.csv): at phat = 0.2 and n = 400 the Clopper-Pearson bound is 0.228 at delta 0.1 and only 0.282 at delta 0.0001, so shrinking delta cannot follow a jump in the miss rate from about 0.2 to about 0.8.

--- POST-HOC ARM (added after the Day 4 freeze) ---

ACI-OFFSET (post-hoc): error frequency 0.12 before step 43 and 0.86 from step 43 on, mean bound 0.410 before and 0.398 after (max 0.680), recovery lag none.

ACI-offset does NOT recover coverage at L = 2, q = 10% under the pre-set rule (error frequency after step 43 <= 0.20 with a finite recovery lag); across the 6 (L, q) settings it meets the rule in 0.

Offset sensitivity to gamma_o (fixed values): error frequency after step 43 aci_offset: 0.86 (mean bound after 0.398), aci_offset_g0.05: 0.86 (mean bound after 0.327), aci_offset_g0.20: 0.57 (mean bound after 0.537).
