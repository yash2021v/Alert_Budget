"""Shared loaders for the IBM AML (HI-Small) data. Every other script imports from here.

Conventions
- Bank ids are integers everywhere (transactions zero-pad them, the accounts file does not).
- Account key = (bank_id, account) -> a stable int `acct_key`, assigned by sorting the unique
  (bank_id, account) pairs seen in the transactions. The sorted order makes it reproducible.
"""
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from src import config as C

TRANS_COLS = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account", "Amount Received",
              "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
_TRANS_CACHE = C.PROCESSED / "transactions.parquet"
_INDEX_CACHE = C.PROCESSED / "account_index.parquet"

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")


def _read_raw_transactions() -> pd.DataFrame:
    string_cols = ["From Bank", "From Account", "To Bank", "To Account"]
    t = pacsv.read_csv(
        C.IBM_TRANS,
        read_options=pacsv.ReadOptions(column_names=TRANS_COLS, skip_rows=1),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in string_cols}),
    )
    return t.to_pandas()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M"),
        "from_bank": df["From Bank"].astype("int32"),
        "from_acct": df["From Account"],
        "to_bank": df["To Bank"].astype("int32"),
        "to_acct": df["To Account"],
        "amount_received": df["Amount Received"],
        "recv_currency": df["Receiving Currency"],
        "amount_paid": df["Amount Paid"],
        "pay_currency": df["Payment Currency"],
        "payment_format": df["Payment Format"],
        "is_laundering": df["Is Laundering"].astype("int8"),
    })
    return out


def account_index(trans: pd.DataFrame | None = None) -> pd.DataFrame:
    """Table (bank_id, account, acct_key) over every account seen in the transactions."""
    if _INDEX_CACHE.exists():
        return pd.read_parquet(_INDEX_CACHE)
    if trans is None:
        trans = load_transactions()
    pairs = pd.concat([
        trans[["from_bank", "from_acct"]].set_axis(["bank_id", "account"], axis=1),
        trans[["to_bank", "to_acct"]].set_axis(["bank_id", "account"], axis=1),
    ]).drop_duplicates().sort_values(["bank_id", "account"], ignore_index=True)
    pairs["acct_key"] = np.arange(len(pairs), dtype="int32")
    pairs.to_parquet(_INDEX_CACHE)
    return pairs


def load_transactions(use_cache: bool = True) -> pd.DataFrame:
    """Cleaned transactions with `tx_id` (row number in the raw file), `from_key`, `to_key`."""
    if use_cache and _TRANS_CACHE.exists():
        return pd.read_parquet(_TRANS_CACHE)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df = _normalise(_read_raw_transactions())
    df.insert(0, "tx_id", np.arange(len(df), dtype="int32"))
    idx = account_index_from(df)
    for side, b, a in (("from", "from_bank", "from_acct"), ("to", "to_bank", "to_acct")):
        m = df[[b, a]].set_axis(["bank_id", "account"], axis=1).merge(idx, on=["bank_id", "account"], how="left")
        df[f"{side}_key"] = m["acct_key"].to_numpy()
    df.to_parquet(_TRANS_CACHE)
    idx.to_parquet(_INDEX_CACHE)
    return df


def account_index_from(trans: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.concat([
        trans[["from_bank", "from_acct"]].set_axis(["bank_id", "account"], axis=1),
        trans[["to_bank", "to_acct"]].set_axis(["bank_id", "account"], axis=1),
    ]).drop_duplicates().sort_values(["bank_id", "account"], ignore_index=True)
    pairs["acct_key"] = np.arange(len(pairs), dtype="int32")
    return pairs


def load_accounts() -> pd.DataFrame:
    """Accounts table with int bank_id and `acct_key` (NA for accounts that never transact)."""
    acc = pd.read_csv(C.IBM_ACCOUNTS, dtype=str)
    acc = acc.rename(columns={"Bank Name": "bank_name", "Bank ID": "bank_id", "Account Number": "account",
                              "Entity ID": "entity_id", "Entity Name": "entity_name"})
    acc["bank_id"] = acc["bank_id"].astype("int32")
    idx = account_index()
    acc = acc.merge(idx, on=["bank_id", "account"], how="left")
    acc["acct_key"] = acc["acct_key"].astype("Int32")
    return acc


def load_patterns(trans: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per transaction line inside a laundering attempt.

    Columns: attempt_id (1-based, file order), typology, tx_id (matched transactions row, <NA> if no
    match), plus the parsed fields. Matching key: timestamp, banks, accounts, both amounts, currencies, format.
    """
    if trans is None:
        trans = load_transactions()
    rows, cur, aid = [], None, 0
    for ln in C.IBM_PATTERNS.read_text().splitlines():
        m = _BEGIN.match(ln)
        if m:
            aid += 1
            cur = m.group(1)
        elif ln.startswith("END LAUNDERING ATTEMPT"):
            cur = None
        elif cur and ln.count(",") == 10:
            rows.append((aid, cur, *ln.split(",")))
    pat = pd.DataFrame(rows, columns=["attempt_id", "typology"] + TRANS_COLS)
    for c in ("Amount Received", "Amount Paid", "Is Laundering"):
        pat[c] = pd.to_numeric(pat[c])
    pat = _normalise(pat).join(pat[["attempt_id", "typology"]])
    key = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amount_received", "recv_currency",
           "amount_paid", "pay_currency", "payment_format"]
    tkey = trans[["tx_id"] + key].drop_duplicates(key, keep="first")  # 9 exact dup rows in raw data
    pat = pat.merge(tkey, on=key, how="left")
    pat["tx_id"] = pat["tx_id"].astype("Int32")
    return pat[["attempt_id", "typology", "tx_id"] + key + ["is_laundering"]]


if __name__ == "__main__":
    t = load_transactions()
    p = load_patterns(t)
    print(f"transactions: {len(t):,}; accounts seen: {len(account_index()):,}")
    print(f"pattern rows: {len(p):,}; matched {p.tx_id.notna().sum():,}; unmatched {p.tx_id.isna().sum():,}")
    print(p.groupby("typology").agg(attempts=("attempt_id", "nunique"), rows=("attempt_id", "size")))
