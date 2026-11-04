"""Add the frozen model's raw margin (log-odds) to data/processed/scores.parquet. No retraining."""
import joblib
import numpy as np
import pandas as pd

from src import config as C
from src import features as F
from src import train as T

if __name__ == "__main__":
    model = joblib.load(T.MODEL_PATH)
    df = T.load_table()
    df["margin"] = model.predict(df[F.FEATURE_COLS], raw_score=True)
    sc = pd.read_parquet(T.SCORES_PATH).drop(columns=["margin"], errors="ignore")
    sc = sc.merge(df[["acct_key", "period", "margin"]], on=["acct_key", "period"], how="left")
    assert sc["margin"].notna().all()
    # consistency with the stored probabilities (same frozen model)
    prob = 1 / (1 + np.exp(-sc["margin"]))
    print("max |sigmoid(margin) - score|:", float((prob - sc["score"]).abs().max()))
    sc.to_parquet(T.SCORES_PATH)
    for split in ("train", "calibration", "test"):
        m = df.loc[df.split == split, "margin"]
        extra = ""
        if split != "train":
            s = sc[sc.split == split]
            extra = (f"; rows with probability exactly 1.0: {int((s.score >= 1.0).sum()):,}; "
                     f"exact margin ties (rows sharing a margin within a period): "
                     f"{int(s.duplicated(['period', 'margin'], keep=False).sum())}")
        print(f"{split:<12} rows {len(m):>9,} margin range [{m.min():.2f}, {m.max():.2f}]"
              f"; distinct margins {m.nunique():,}{extra}")
    print("scores.parquet now has columns:", list(sc.columns))
