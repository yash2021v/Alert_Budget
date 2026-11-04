"""Loader for the frozen model's scores. Entities are ranked by `margin` (LightGBM raw score / log-odds), not by
the probability: probabilities saturate at exactly 1.0 for ~4.6% of test rows, which makes top-K ill-defined.

Residual exact ties in `margin` are broken by ONE seeded shuffle of the table (here); certify.py then breaks
ties by input position, so the order is random but reproducible (seed = config.SEED).
"""
import pandas as pd

from src import config as C

SCORES_PATH = C.PROCESSED / "scores.parquet"
RANK_COL = "margin"


def load_scores(split: str | None = None) -> pd.DataFrame:
    sc = pd.read_parquet(SCORES_PATH)
    if RANK_COL not in sc.columns:
        raise RuntimeError("scores.parquet has no `margin` column; run `python -m src.add_margin` first")
    if split is not None:
        sc = sc[sc["split"] == split]
    return sc.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
