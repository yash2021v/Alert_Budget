"""Step 0: choose the primary (headline) IBM scorer using CALIBRATION data only.

Rule fixed in advance: the scorer with the higher mean recall over the fixed-count K grid on the calibration split
is primary; the other stays as a secondary comparison. Test results are not looked at here.
Writes results/headline_scorer.md and results/headline_scorer.json.
"""
import json

import numpy as np

from src import certify as CE
from src import config as C
from src.run_h3 import K_FRACS, variant_tables

if __name__ == "__main__":
    tables = variant_tables()
    test_avg = tables["direct"].query("split == 'test'").groupby("period").size().mean()   # same K grid as before
    res = {}
    for name, sc in tables.items():
        cal = sc[sc.split == "calibration"]
        rec = []
        for f in K_FRACS:
            k = int(round(f * test_avg))
            m = CE.missed_flags(cal["score"].to_numpy(), cal["label"].to_numpy(), cal["period"].to_numpy(), k)
            rec.append(1 - m.mean())
        res[name] = {"recall_by_K": dict(zip([int(round(f * test_avg)) for f in K_FRACS], np.round(rec, 4).tolist())),
                     "mean_calibration_recall": float(np.mean(rec))}
    primary = max(res, key=lambda n: res[n]["mean_calibration_recall"])
    secondary = [n for n in res if n != primary][0]
    names = {"direct": "entity-level direct scorer (frozen entity LightGBM margin)",
             "max_txn": "max-over-transaction scorer (frozen transaction LightGBM, max margin per entity-period)"}
    lines = ["# Headline scorer (decided on the calibration split only)", "",
             "Rule (fixed in advance): higher mean recall over the K grid on the CALIBRATION split wins; test results were not used.", "",
             "| scorer | " + " | ".join(f"K={k}" for k in res["direct"]["recall_by_K"]) + " | mean |",
             "|---|" + "---|" * (len(res["direct"]["recall_by_K"]) + 1)]
    for n, r in res.items():
        lines.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in r["recall_by_K"].values()) + f" | {r['mean_calibration_recall']:.4f} |")
    lines += ["", f"**Decision: primary = {primary}** ({names[primary]}); secondary = {secondary}. "
              "All IBM results from here on use the primary scorer; the other is kept as a comparison."]
    (C.RESULTS / "headline_scorer.md").write_text("\n".join(lines) + "\n")
    (C.RESULTS / "headline_scorer.json").write_text(json.dumps({"primary": primary, "secondary": secondary, **res}, indent=2))
    print("\n".join(lines))
