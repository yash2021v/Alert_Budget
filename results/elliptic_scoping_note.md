# Elliptic scoping note

1. Elliptic is used only for the drift experiment (H4); the IBM AML entity-period panel carries H1-H3.
2. It is scored with its native anonymized features (165 columns); there are no raw amounts or counterparties to engineer from.
3. Labels are coded 1 = illicit, 2 = licit; nodes labelled "unknown" (157,205 of 203,769) are excluded from calibration and scoring.
4. 46,564 labelled nodes remain: 39,877 in time steps 1-42 (11.0% illicit) and 6,687 in steps 43-49 (2.5% illicit).
5. The dark-market shutdown (around step 43) is the drift event: calibrate on early steps, then track coverage over time across it.
