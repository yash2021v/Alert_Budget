# Headline scorer (decided on the calibration split only)

Rule (fixed in advance): higher mean recall over the K grid on the CALIBRATION split wins; test results were not used.

| scorer | K=155 | K=388 | K=776 | K=1552 | K=3104 | K=4657 | K=7761 | mean |
|---|---|---|---|---|---|---|---|---|
| direct | 0.027 | 0.070 | 0.247 | 0.395 | 0.516 | 0.616 | 0.760 | 0.3759 |
| max_txn | 0.265 | 0.370 | 0.461 | 0.562 | 0.699 | 0.777 | 0.840 | 0.5675 |

**Decision: primary = max_txn** (max-over-transaction scorer (frozen transaction LightGBM, max margin per entity-period)); secondary = direct. All IBM results from here on use the primary scorer; the other is kept as a comparison.
