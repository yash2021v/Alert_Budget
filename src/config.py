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
