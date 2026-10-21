from pathlib import Path

import pandas as pd
import streamlit as st

CURVE = Path(__file__).resolve().parents[1] / "results" / "budget_curve.csv"

st.title("AlertBudget")
st.caption("Budget-aware AML alert triage with statistically certified miss rates")

k = st.sidebar.slider("Daily alert budget K", min_value=100, max_value=1000, value=500, step=50)
st.write(f"Selected daily budget: **K = {k}** alerts")

if CURVE.exists():
    curve = pd.read_csv(CURVE)
    st.line_chart(curve.set_index(curve.columns[0]))
else:
    st.info("Results not ready yet: results/budget_curve.csv has not been produced.")
