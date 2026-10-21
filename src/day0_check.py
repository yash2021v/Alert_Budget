"""Day 0: verify every raw dataset loads and write results/day0_data_summary.md."""
import io
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT = C.RESULTS / "day0_data_summary.md"
_buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _buf.write(s + "\n")


def block(obj):
    say("```\n" + str(obj) + "\n```")


def size(p):
    return f"{p.stat().st_size / 1e6:,.1f} MB"


def describe(df, name):
    say(f"- rows: **{len(df):,}**, columns: {df.shape[1]}")
    say("- column names:", list(df.columns) if df.shape[1] <= 30 else f"{list(df.columns[:8])} ... {list(df.columns[-3:])}")
    say("\nDtypes:")
    block(df.dtypes.value_counts().to_string() if df.shape[1] > 30 else df.dtypes.to_string())
    say("First 3 rows:")
    block(df.head(3).iloc[:, :12].to_string())
    nulls = df.isna().sum()
    say("Null counts:", "none" if nulls.sum() == 0 else "")
    if nulls.sum():
        block(nulls[nulls > 0].to_string())


def timed(f):
    t = time.time()
    r = f()
    say(f"- load time: {time.time() - t:.1f}s")
    return r


# ---------------------------------------------------------------- IBM AML
def ibm_trans():
    say("\n## IBM AML HI-Small_Trans.csv\n")
    say("- file size:", size(C.IBM_TRANS))
    cols = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account", "Amount Received",
            "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
    say("- note: the raw header has two columns both named `Account`; renamed here to From Account / To Account")
    ro = pacsv.ReadOptions(column_names=cols, skip_rows=1)
    co = pacsv.ConvertOptions(column_types={"From Bank": pa.string(), "To Bank": pa.string(),
                                            "From Account": pa.string(), "To Account": pa.string()})
    df = timed(lambda: pacsv.read_csv(C.IBM_TRANS, read_options=ro, convert_options=co).to_pandas())
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    describe(df, "trans")
    say(f"- time range: {df.Timestamp.min()} -> {df.Timestamp.max()}")
    pos = int(df["Is Laundering"].sum())
    say(f"- Is Laundering positives: **{pos:,}** ({pos / len(df):.4%}); negatives {len(df) - pos:,}")
    accts = pd.concat([df["From Bank"] + "|" + df["From Account"], df["To Bank"] + "|" + df["To Account"]])
    say(f"- unique accounts (bank+account): **{accts.nunique():,}**; unique account numbers only: "
        f"{pd.concat([df['From Account'], df['To Account']]).nunique():,}")
    say("- payment formats (count / laundering rate):")
    g = df.groupby("Payment Format")["Is Laundering"].agg(["size", "mean"])
    block(g.sort_values("size", ascending=False).to_string())
    say(f"- duplicate rows (all columns): {int(df.duplicated().sum()):,}")
    return df


def ibm_accounts(df):
    say("\n## IBM AML HI-Small_accounts.csv\n")
    say("- file size:", size(C.IBM_ACCOUNTS))
    acc = timed(lambda: pd.read_csv(C.IBM_ACCOUNTS, dtype=str))
    describe(acc, "accounts")
    say(f"- unique Account Number: {acc['Account Number'].nunique():,}; unique Entity ID: {acc['Entity ID'].nunique():,}; "
        f"unique Bank ID: {acc['Bank ID'].nunique():,}")
    say("- note: trans zero-pads bank ids (e.g. `010`), accounts does not (`10`); bank ids are compared as integers")
    key = set(acc["Bank ID"].astype(int).astype(str) + "|" + acc["Account Number"])
    tx_from = df["From Bank"].astype(int).astype(str) + "|" + df["From Account"]
    tx_to = df["To Bank"].astype(int).astype(str) + "|" + df["To Account"]
    txacc = set(tx_from.unique()) | set(tx_to.unique())
    say("- **Join**: accounts.(Bank ID, Account Number) <-> trans.(From Bank, From Account) and (To Bank, To Account); "
        "Entity ID groups accounts by owning entity.")
    say(f"- transaction accounts found in accounts table: {len(txacc & key):,} / {len(txacc):,}")
    say(f"- accounts-table rows never appearing in trans: {len(key - txacc):,}")
    say(f"- (Bank ID, Account Number) duplicates in accounts table: {int(acc.duplicated(['Bank ID', 'Account Number']).sum()):,}")


def ibm_patterns(df):
    say("\n## IBM AML HI-Small_Patterns.txt\n")
    say("- file size:", size(C.IBM_PATTERNS))
    lines = C.IBM_PATTERNS.read_text().splitlines()
    say(f"- total lines: {len(lines):,}\n\nFirst 40 lines:")
    block("\n".join(lines[:40]))
    begin = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")
    attempts, cur, rows = Counter(), None, []
    for ln in lines:
        m = begin.match(ln)
        if m:
            cur = m.group(1)
            attempts[cur] += 1
            aid = sum(attempts.values())
        elif ln.startswith("END LAUNDERING ATTEMPT"):
            cur = None
        elif cur and ln.strip():
            rows.append((aid, cur, ln))
    say("Attempts per typology:")
    block("\n".join(f"{k:<16}{v:>6}" for k, v in attempts.most_common()) + f"\n{'TOTAL':<16}{sum(attempts.values()):>6}")
    say(f"- transaction lines inside attempts: {len(rows):,}")
    say("""
**Markup:** each laundering attempt is a block `BEGIN LAUNDERING ATTEMPT - <TYPE>[: detail]` ... `END LAUNDERING ATTEMPT - <TYPE>`.
Between the markers each line is a transaction in exactly the same 11-field CSV layout as `HI-Small_Trans.csv`
(Timestamp, From Bank, From Account, To Bank, To Account, Amount Received, Receiving Currency, Amount Paid,
Payment Currency, Payment Format, Is Laundering). There is no transaction ID, so **linking = join on all of those fields**
(timestamp + from bank/account + to bank/account + amounts + currencies + format); attempt id and typology are then attached
to the matching trans row(s).
""")
    pat = pd.DataFrame([(a, t, *l.split(",")) for a, t, l in rows if l.count(",") == 10],
                       columns=["attempt_id", "typology", "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
                                "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format",
                                "Is Laundering"])
    pat["Timestamp"] = pd.to_datetime(pat["Timestamp"], format="%Y/%m/%d %H:%M")
    for c in ["Amount Received", "Amount Paid"]:
        pat[c] = pat[c].astype(float)
    pat["Is Laundering"] = pat["Is Laundering"].astype(int)
    key = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account", "Amount Received",
           "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format"]
    trans_key = df[key + ["Is Laundering"]].reset_index().rename(columns={"index": "trans_row"})
    m = pat.merge(trans_key, on=key, how="left", suffixes=("", "_t"))
    matched = m["trans_row"].notna()
    say(f"- parsed pattern rows: {len(pat):,}; matched to a trans row: {int(matched.sum()):,} "
        f"({matched.mean():.2%}); unmatched: {int((~matched).sum()):,}")
    say(f"- distinct trans rows matched: {m['trans_row'].nunique():,}; "
        f"trans positives total: {int(df['Is Laundering'].sum()):,}")
    ambig = m.groupby(list(pat.columns[:1]) + key)["trans_row"].nunique()
    say(f"- pattern rows matching >1 trans row (ambiguous key): {int((ambig > 1).sum()):,}")
    pos_unlinked = set(df.index[df["Is Laundering"] == 1]) - set(m["trans_row"].dropna().astype(int))
    say(f"- Is Laundering=1 trans rows NOT in any pattern attempt: {len(pos_unlinked):,}")


# ---------------------------------------------------------------- Elliptic
def elliptic():
    say("\n## Elliptic\n")
    for p in (C.ELLIPTIC_FEATURES, C.ELLIPTIC_CLASSES, C.ELLIPTIC_EDGES):
        say(f"- {p.name}: {size(p)}")
    say("\n### elliptic_txs_features.csv (no header)\n")
    first = open(C.ELLIPTIC_FEATURES).readline().split(",")
    say(f"- first line has {len(first)} fields, first value `{first[0]}` is numeric -> confirms no header")
    ncol = len(first)
    names = ["txId", "time_step"] + [f"f{i}" for i in range(1, ncol - 1)]
    co = pacsv.ConvertOptions(column_types={"txId": pa.int64(), "time_step": pa.int32(),
                                            **{n: pa.float32() for n in names[2:]}})
    feats = timed(lambda: pacsv.read_csv(C.ELLIPTIC_FEATURES, read_options=pacsv.ReadOptions(column_names=names),
                                         convert_options=co).to_pandas())
    describe(feats, "features")
    say(f"- node count (rows): **{len(feats):,}**; unique txId: {feats.txId.nunique():,}")
    say(f"- time step range: {feats.time_step.min()} - {feats.time_step.max()} ({feats.time_step.nunique()} distinct)")
    say(f"- feature columns (excluding txId, time step): {ncol - 2}  "
        "(anonymized: 93 local + 72 aggregated neighbour features per the Elliptic paper; no raw amounts, addresses or counterparties)")
    cls = pd.read_csv(C.ELLIPTIC_CLASSES)
    say("\n### elliptic_txs_classes.csv\n")
    say(f"- rows: {len(cls):,}; columns: {list(cls.columns)}")
    block(cls["class"].value_counts().to_string())
    cls_all = feats[["txId", "time_step"]].merge(cls, on="txId", how="outer", indicator=True)
    say(f"- txId join features<->classes: {cls_all['_merge'].value_counts().to_dict()}")
    say("- label coding in this file: `1` = illicit, `2` = licit, `unknown` = unlabelled (standard Elliptic coding; "
        "the file holds numeric codes, not the words illicit/licit)")
    say("Class counts per time step:")
    block(pd.crosstab(cls_all.time_step, cls_all["class"]).to_string())
    edges = pd.read_csv(C.ELLIPTIC_EDGES)
    say("\n### elliptic_txs_edgelist.csv\n")
    say(f"- edges: {len(edges):,}; columns: {list(edges.columns)}")
    ids = set(feats.txId)
    say(f"- edge endpoints missing from features: {int((~edges.txId1.isin(ids)).sum() + (~edges.txId2.isin(ids)).sum()):,}")
    say("- null counts: " + str(edges.isna().sum().to_dict()))


# ---------------------------------------------------------------- SAML-D
def saml():
    say("\n## SAML-D.csv (out of scope for the 5-day plan; load check only)\n")
    say("- file size:", size(C.SAML_D))
    t = timed(lambda: pacsv.read_csv(C.SAML_D))
    say(f"- rows: **{t.num_rows:,}**; columns: {t.column_names}")
    say(f"- dtypes (arrow): {dict(zip(t.column_names, map(str, t.schema.types)))}")


if __name__ == "__main__":
    C.RESULTS.mkdir(exist_ok=True)
    say("# Day 0 data summary\n")
    say(f"Generated {time.strftime('%Y-%m-%d %H:%M')}; pandas {pd.__version__}, pyarrow {pa.__version__}")
    df = ibm_trans()
    ibm_accounts(df)
    ibm_patterns(df)
    elliptic()
    saml()
    OUT.write_text(_buf.getvalue())
    print(f"\nWrote {OUT}")
