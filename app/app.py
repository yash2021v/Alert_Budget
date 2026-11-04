from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results"

st.set_page_config(page_title="AlertBudget", layout="wide")
st.title("AlertBudget")
st.caption("Budget-aware AML alert triage with statistically certified miss rates")

k = st.sidebar.slider("Daily alert budget K", min_value=100, max_value=1000, value=500, step=50)
delay = st.sidebar.selectbox("Label delay (periods)", [0, 2, 4])
delta = st.sidebar.selectbox("Confidence 1 - delta", [0.10, 0.05], format_func=lambda d: f"{1 - d:.0%}")
st.sidebar.write(f"K = **{k}**, delay = **{delay}** periods")

VALIDITY = ("Validity: the certified bound holds with probability >= 1 - delta only if deployment positives are "
            "exchangeable with the calibration positives; temporal drift or dependence between periods can break it.")


def read(name: str):
    path = RESULTS / name
    if not path.exists():
        st.info(f"Results not ready yet: results/{name} has not been produced.")
        return None
    return pd.read_csv(path)


def at_delta(df: pd.DataFrame) -> pd.DataFrame:
    return df[np.isclose(df["delta"], delta)]


def nearest(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the computed K closest to the slider value."""
    ks = df["K"].unique()
    return df[df["K"] == ks[np.abs(ks - k).argmin()]]


def miss_rate_panel():
    st.subheader("Certified vs realised miss rate")
    df = read("miss_rate_certified_vs_realised.csv")
    if df is None:
        return
    scorers = list(dict.fromkeys(df["scorer"]))                      # primary scorer is written first
    names = {"max_txn": "max-over-transaction", "direct": "entity-level direct"}
    scorer = st.sidebar.selectbox("Scorer", scorers, format_func=lambda s: names.get(s, s) + (" (primary)" if s == scorers[0] else ""))
    d = at_delta(df[df["scorer"] == scorer]).sort_values("K")
    row = nearest(d).iloc[0]
    long = d.melt(id_vars=["K"], value_vars=["certified_bound", "test_miss_rate_realised"],
                  var_name="series", value_name="miss_rate")
    base = alt.Chart(long).encode(x=alt.X("K:Q", scale=alt.Scale(type="log"), title="Budget K per period"),
                                  y=alt.Y("miss_rate:Q", title="Miss rate", scale=alt.Scale(domain=[0, 1])),
                                  color=alt.Color("series:N", title=None))
    hi = alt.Chart(pd.DataFrame({"K": [row["K"]]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="K:Q")
    st.altair_chart(base.mark_line(point=True) + hi, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Nearest computed K (slider {k})", f"{int(row['K'])}")
    c2.metric(f"Certified bound ({1 - delta:.0%})", f"{row['certified_bound']:.3f}")
    c3.metric("Realised test miss rate", f"{row['test_miss_rate_realised']:.3f}",
              "covered" if row["covered"] else "NOT covered", delta_color="normal" if row["covered"] else "inverse")
    st.caption(VALIDITY)


def typology_panel():
    st.subheader("Coverage by typology")
    df = read("coverage_by_typology.csv")
    if df is None:
        return
    d = nearest(at_delta(df)).copy()
    d["group"] = np.where(d["low_n"], d["typology"] + " (low-n)", d["typology"])
    long = d.melt(id_vars=["group", "low_n"], value_vars=["group_bound", "group_test_miss"], var_name="series",
                  value_name="miss_rate")
    long["series"] = long["series"].map({"group_bound": "group bound", "group_test_miss": "realised test miss"})
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("group:N", title=None, sort=None), xOffset="series:N", y=alt.Y("miss_rate:Q", title="Miss rate"),
        color=alt.Color("series:N", title=None), opacity=alt.condition(alt.datum.low_n, alt.value(0.4), alt.value(1.0)))
    pooled = alt.Chart(pd.DataFrame({"y": [d["pooled_bound"].iloc[0]]})).mark_rule(color="black", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(bars + pooled, width="stretch")
    st.caption(f"K = {int(d['K'].iloc[0])} (nearest to slider); dashed line = pooled bound; faded bars = fewer than 30 "
               "calibration positives.")


def elliptic_panel():
    st.subheader("Elliptic: coverage over time")
    df = read("elliptic_spike.csv")
    if df is None:
        return
    q = st.radio("Budget (share of labelled nodes per step)", sorted(df["q"].unique()), horizontal=True,
                 format_func=lambda v: f"{v:.0%}")
    d = df[df["q"] == q]
    x = alt.X("step:Q", title="Time step")
    miss = alt.Chart(d).mark_line(point=True, color="#d9731a").encode(x=x, y=alt.Y("realised_miss:Q", title="Miss rate"))
    bound = alt.Chart(d).mark_line(color="#1f5fa8").encode(x=x, y="bound:Q")
    mark = alt.Chart(pd.DataFrame({"step": [43]})).mark_rule(color="#c0392b", strokeDash=[4, 4]).encode(x="step:Q")
    st.altair_chart(miss + bound + mark, width="stretch")
    st.caption("Orange = realised miss rate per step, blue = bound from steps 25-34, red dashed = step 43 (shutdown).")


def h3_panel():
    st.subheader("H3: direct vs max-over-transaction score")
    df = read("h3_aggregation.csv")
    if df is None:
        return
    d = at_delta(df)[["variant", "K", "certified_bound", "test_miss_realised", "covered", "recall_at_K_test",
                      "control_coverage_vs_pooled"]]
    st.dataframe(d.round(3), hide_index=True, width="stretch")


miss_rate_panel()
left, right = st.columns(2)
with left:
    typology_panel()
with right:
    elliptic_panel()
h3_panel()
