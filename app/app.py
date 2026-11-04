from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results"

st.set_page_config(page_title="AlertBudget", layout="wide")
st.title("AlertBudget")
st.caption("Budget-aware AML alert triage with statistically certified miss rates")

k = st.sidebar.slider("Daily alert budget K", min_value=100, max_value=1000, value=500, step=50)
delay = st.sidebar.selectbox("Label delay (periods)", [0, 2, 4])
st.sidebar.write(f"K = **{k}**, delay = **{delay}** periods")


def chart_panel(title: str, csv_name: str):
    st.subheader(title)
    path = RESULTS / csv_name
    if not path.exists():
        st.info(f"Results not ready yet: results/{csv_name} has not been produced.")
        return
    df = pd.read_csv(path)
    st.line_chart(df.set_index(df.columns[0]))


VALIDITY = ("Validity: the certified bound holds with probability >= 1 - delta only if deployment positives are "
            "exchangeable with the calibration positives; temporal drift or dependence between periods can break it.")


def miss_rate_panel():
    st.subheader("Certified vs realised miss rate")
    path = RESULTS / "miss_rate_certified_vs_realised.csv"
    if not path.exists():
        st.info("Results not ready yet: results/miss_rate_certified_vs_realised.csv has not been produced.")
        return
    df = pd.read_csv(path)
    delta = st.sidebar.selectbox("Confidence 1 - delta", sorted(df["delta"].unique(), reverse=True),
                                 format_func=lambda d: f"{1 - d:.0%}")
    d = df[df["delta"] == delta].sort_values("K")
    row = d.iloc[(d["K"] - k).abs().argmin()]          # nearest budget in the computed grid
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


miss_rate_panel()
left, right = st.columns(2)
with left:
    chart_panel("Coverage by typology", "coverage_by_typology.csv")
with right:
    chart_panel("Elliptic: coverage over time", "elliptic_coverage_over_time.csv")
