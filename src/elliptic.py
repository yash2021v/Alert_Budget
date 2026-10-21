"""Elliptic loader and scoping summary (used only for the H4 drift experiment)."""
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from src import config as C

LABELS = {"1": "illicit", "2": "licit", "unknown": "unknown"}


def load_elliptic_labels() -> pd.DataFrame:
    """txId, time_step, label in {illicit, licit, unknown} (raw file codes 1 = illicit, 2 = licit)."""
    first = open(C.ELLIPTIC_FEATURES).readline().count(",") + 1
    names = ["txId", "time_step"] + [f"f{i}" for i in range(1, first - 1)]
    t = pacsv.read_csv(C.ELLIPTIC_FEATURES, read_options=pacsv.ReadOptions(column_names=names),
                       convert_options=pacsv.ConvertOptions(include_columns=["txId", "time_step"],
                                                            column_types={"txId": pa.int64(), "time_step": pa.int32()}))
    cls = pd.read_csv(C.ELLIPTIC_CLASSES, dtype={"class": str})
    cls["label"] = cls["class"].map(LABELS)
    return t.to_pandas().merge(cls[["txId", "label"]], on="txId", how="left")


if __name__ == "__main__":
    d = load_elliptic_labels()
    assert d["label"].notna().all()
    lab = d[d["label"] != "unknown"]
    g = lab.groupby("time_step")["label"].agg(n="size", illicit=lambda s: (s == "illicit").sum())
    g["illicit_share"] = (g["illicit"] / g["n"]).round(4)
    print(g.to_string())
    before, after = lab[lab.time_step <= 42], lab[lab.time_step >= 43]
    print(f"labelled nodes total {len(lab):,} (excluded unknown: {(d.label == 'unknown').sum():,})")
    print(f"steps 1-42: {len(before):,} labelled, illicit share {(before.label == 'illicit').mean():.3%}")
    print(f"steps 43-49: {len(after):,} labelled, illicit share {(after.label == 'illicit').mean():.3%}")
    note = f"""# Elliptic scoping note

1. Elliptic is used only for the drift experiment (H4); the IBM AML entity-period panel carries H1-H3.
2. It is scored with its native anonymized features (165 columns); there are no raw amounts or counterparties to engineer from.
3. Labels are coded 1 = illicit, 2 = licit; nodes labelled "unknown" ({(d.label == 'unknown').sum():,} of {len(d):,}) are excluded from calibration and scoring.
4. {len(lab):,} labelled nodes remain: {len(before):,} in time steps 1-42 ({(before.label == 'illicit').mean():.1%} illicit) and {len(after):,} in steps 43-49 ({(after.label == 'illicit').mean():.1%} illicit).
5. The dark-market shutdown (around step 43) is the drift event: calibrate on early steps, then track coverage over time across it.
"""
    (C.RESULTS / "elliptic_scoping_note.md").write_text(note)
