# H1 summary (delta 0.10, K = 0.5%, 1%, 2%, 5% of period size)

Definitions (fixed in advance): 'under-served in the exchangeable protocol' = pooled_exchangeable_coverage < 0.90; 'under-served on the temporal split' = the pooled bound is below the group's realised test miss rate.

PRIMARY scorer (max-over-transaction; post-hoc rerun): UNATTRIBUTED: exchangeable under-served at 4 of 4 budgets (coverage 0.00-0.00), temporal split under-served at 4 of 4; BIPARTITE: exchangeable under-served at 1 of 4 budgets (coverage 0.29-1.00), temporal split under-served at 2 of 4; SCATTER-GATHER: exchangeable under-served at 0 of 4 budgets (coverage 1.00-1.00), temporal split under-served at 0 of 4.

DIRECT scorer (secondary): UNATTRIBUTED: exchangeable under-served at 4 of 4 budgets (coverage 0.00-0.00), temporal split under-served at 4 of 4; BIPARTITE: exchangeable under-served at 2 of 4 budgets (coverage 0.00-0.98), temporal split under-served at 4 of 4; SCATTER-GATHER: exchangeable under-served at 0 of 4 budgets (coverage 1.00-1.00), temporal split under-served at 3 of 4.

UNATTRIBUTED positives have a calibration miss rate of 0.33-0.94 under the primary scorer against pooled bounds of 0.17-0.55, so the pooled certificate under-serves the group that no pattern file explains even when time order is removed.
