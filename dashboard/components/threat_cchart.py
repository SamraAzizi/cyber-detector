import streamlit as st
import pandas as pd
import altair as alt

def render_threat_chart():
    # Sample threat data - replace with actual API or data source
    data = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=10),
        "Threats Detected": [3, 5, 1, 7, 6, 4, 8, 2, 6, 5]
    })

    chart = alt.Chart(data).mark_line(point=True).encode(
        x="Date:T",
        y="Threats Detected:Q",
        tooltip=["Date", "Threats Detected"]
    ).properties(title="Threats Over Time")

    st.altair_chart(chart, use_container_width=True)
