import pytest

from src import data


@pytest.fixture(scope="module")
def trans():
    return data.load_transactions()


def test_bank_ids_are_ints_and_columns_renamed(trans):
    assert str(trans["from_bank"].dtype).startswith("int")
    assert {"from_acct", "to_acct", "from_key", "to_key"} <= set(trans.columns)
    assert trans[["from_key", "to_key"]].notna().all().all()


def test_accounts_join_matches_sensible_fraction(trans):
    acc = data.load_accounts()
    matched = acc["acct_key"].notna().mean()
    assert 0.95 < matched <= 1.0          # accounts table slightly larger than the active set
    n_keys = len(data.account_index())
    assert acc["acct_key"].nunique() == acc["acct_key"].notna().sum()  # no duplicate keys
    # every transacting account exists in the accounts table
    assert acc["acct_key"].nunique() == n_keys


def test_patterns_link_to_transactions(trans):
    p = data.load_patterns(trans)
    assert p["tx_id"].notna().mean() > 0.99
    assert p["typology"].nunique() == 8
    linked = trans.set_index("tx_id").loc[p["tx_id"].dropna().astype(int), "is_laundering"]
    assert (linked == 1).all()
