from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Raw files (data/raw holds symlinks to the original dataset folder)
ELLIPTIC_FEATURES = RAW / "elliptic_txs_features.csv"
ELLIPTIC_CLASSES = RAW / "elliptic_txs_classes.csv"
ELLIPTIC_EDGES = RAW / "elliptic_txs_edgelist.csv"
IBM_TRANS = RAW / "HI-Small_Trans.csv"
IBM_ACCOUNTS = RAW / "HI-Small_accounts.csv"
IBM_PATTERNS = RAW / "HI-Small_Patterns.txt"
SAML_D = RAW / "SAML-D.csv"

PROCESSED = ROOT / "data" / "processed"
SEED = 42

# The IBM HI-Small generator stops emitting normal traffic after 2022-09-10 (<500 tx/day vs 200k-1.1M)
# while laundering rows keep trickling in. Panel / split / features only use transactions before this.
ACTIVE_END = "2022-09-11"
PERIOD_HOURS = 12   # <14 distinct regular-volume days -> 12-hour blocks
