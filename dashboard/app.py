import streamlit as st
from components.header import show_header
from components.predict_form import prediction_section
from components.threat_chart import render_threat_chart
from components.threat_summary import show_summary_metrics

st.set_page_config(page_title="Cyber Threat Detector", layout="wide", page_icon="🚨")

def main():
    show_header()

    tabs = st.tabs(["🔎 Threat Detection", "📈 Analytics", "📄 Summary"])

    with tabs[0]:
        prediction_section()

    with tabs[1]:
        st.subheader("Threat Trends")
        render_threat_chart()

    with tabs[2]:
        st.subheader("Summary Metrics")
        show_summary_metrics()

if __name__ == "__main__":
    main()
