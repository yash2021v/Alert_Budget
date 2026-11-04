# H2 synthetic summary

With no drift (d = 0, the control) coverage is 0.916-0.920 across delays 0, 2 and 4 against a nominal 0.90, so staleness alone does not break the certificate when positives are exchangeable.

At d = 0.05 coverage goes 0.091 -> 0.057 -> 0.057 as the delay grows from 0 to 2 to 4 periods (mean bound 0.859 at L = 0 versus 0.833 at L = 4).

At d = 0.1 coverage goes 0.008 -> 0.007 -> 0.006 as the delay grows from 0 to 2 to 4 periods (mean bound 0.886 at L = 0 versus 0.841 at L = 4).

At d = 0.2 coverage goes 0.071 -> 0.000 -> 0.000 as the delay grows from 0 to 2 to 4 periods (mean bound 0.918 at L = 0 versus 0.855 at L = 4).

With drift the certificate fails at every delay, including L = 0: coverage is only 0.000-0.091 for d > 0, because the pool of about 1685 positives gives an interval roughly +-0.02 wide, which any systematic drift between the pool periods and period t exceeds.

By comparison the effect of the delay is small: moving from L = 0 to L = 4 changes coverage by -0.036 on average over the drifting cases, while the step from d = 0 to d = 0.05 changes it by -0.825 at L = 0 (d = 0.2 is partly muddied because positives hit the negatives' mean and the clamp makes the process stationary again).

So the dose-response is steep in the presence of drift and weak in the delay: with a large calibration pool, stale feedback is harmless only when the process is stationary, and any drift needs a monitor and adaptation (H4) rather than a staleness limit.
