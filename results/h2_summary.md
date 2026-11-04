# H2 synthetic summary

Control (d = 0): coverage is 0.916-0.920 at delays 0, 2 and 4 against a nominal 0.90, so delay alone does not break the certificate when positives are exchangeable.

Finer drift grid (added post hoc because d >= 0.05 sits on the floor): coverage at L = 0 / 2 / 4 is d = 0.005: 0.833 / 0.773 / 0.731; d = 0.01: 0.753 / 0.603 / 0.504; d = 0.02: 0.539 / 0.290 / 0.206.

Delay effect: at d = 0.005, 0.01, 0.02 coverage at L = 0 exceeds coverage at L = 4 by more than 3 combined standard errors (d = 0.005: +0.102; d = 0.01: +0.248; d = 0.02: +0.332), so staler feedback does cost coverage once drift is slow enough for coverage to be measurable.

Coverage is already below 0.05 at L = 0 for d = 0.1, so the dose-response in delay cannot be seen there (floor effect).

With about 1685 positives in the pool the interval is roughly +-0.02 wide, so the certificate tolerates only drift that moves the true miss rate by less than about that much over the pool-plus-delay window; larger drifts need monitoring and adaptation (H4) rather than a staleness limit.
