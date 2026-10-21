from pathlib import Path

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


chart_panel("Certified vs realised miss rate", "miss_rate_certified_vs_realised.csv")
left, right = st.columns(2)
with left:
    chart_panel("Coverage by typology", "coverage_by_typology.csv")
with right:
    chart_panel("Elliptic: coverage over time", "elliptic_coverage_over_time.csv")
