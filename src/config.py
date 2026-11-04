"""Paths and run mode. Environment variables:
  DATA_DIR            folder holding the seven raw dataset files (default: <repo>/data/raw)
  ALERTBUDGET_MODE    "full" (default) or "quick": quick uses a 15% account sample, 100 Monte Carlo draws instead of 1,000, and
                      writes to results_quick/ and data/processed_quick/ so committed results are never overwritten
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("DATA_DIR", ROOT / "data" / "raw")).expanduser()
MODE = os.environ.get("ALERTBUDGET_MODE", "full")
QUICK = MODE == "quick"
SAMPLE_FRAC = 0.15 if QUICK else 1.0
MC = 100 if QUICK else 1000                      # Monte Carlo trials / random re-splits per configuration
RESULTS = ROOT / ("results_quick" if QUICK else "results")
FIGURES = RESULTS / "figures"

# Raw files
ELLIPTIC_FEATURES = RAW / "elliptic_txs_features.csv"
ELLIPTIC_CLASSES = RAW / "elliptic_txs_classes.csv"
ELLIPTIC_EDGES = RAW / "elliptic_txs_edgelist.csv"
IBM_TRANS = RAW / "HI-Small_Trans.csv"
IBM_ACCOUNTS = RAW / "HI-Small_accounts.csv"
IBM_PATTERNS = RAW / "HI-Small_Patterns.txt"
SAML_D = RAW / "SAML-D.csv"

PROCESSED = ROOT / "data" / ("processed_quick" if QUICK else "processed")
SEED = 42

# The IBM HI-Small generator stops emitting normal traffic after 2022-09-10 (<500 tx/day vs 200k-1.1M)
# while laundering rows keep trickling in. Panel / split / features only use transactions before this.
ACTIVE_END = "2022-09-11"
PERIOD_HOURS = 12   # <14 distinct regular-volume days -> 12-hour blocks

for _d in (RESULTS, FIGURES, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)
