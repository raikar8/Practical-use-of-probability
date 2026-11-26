"""Practical Use of Probability — entry point.

Run from the repository root:
    streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Practical Use of Probability", page_icon=":material/insights:",
                   layout="wide")

pages = {
    "Start": [
        st.Page("app_pages/home.py", title="Overview", icon=":material/home:", default=True),
    ],
    "Learn: basics to advanced": [
        st.Page("app_pages/l0_probability.py", title="L0 · Probability toolkit", icon=":material/casino:"),
        st.Page("app_pages/l1_estimation.py", title="L1 · MLE, MAP, CRLB", icon=":material/target:"),
        st.Page("app_pages/l2_adaptive.py", title="L2 · LMS, NLMS, RLS", icon=":material/tune:"),
        st.Page("app_pages/l3_kalman.py", title="L3 · Kalman filter", icon=":material/timeline:"),
        st.Page("app_pages/l4_hmm.py", title="L4 · Hidden Markov models", icon=":material/hub:"),
        st.Page("app_pages/l5_advanced.py", title="L5 · Particles, IMM, change points", icon=":material/scatter_plot:"),
    ],
    "Use it: domain demos": [
        st.Page("app_pages/d_vehicle.py", title="Car GPS + IMU fusion", icon=":material/directions_car:"),
        st.Page("app_pages/d_netcode.py", title="Online game lag prediction", icon=":material/sports_esports:"),
        st.Page("app_pages/d_speech.py", title="Voice activity detection", icon=":material/graphic_eq:"),
        st.Page("app_pages/d_enhance.py", title="Noise suppression & hearing aids", icon=":material/hearing:"),
        st.Page("app_pages/d_markets.py", title="Market regimes & volatility", icon=":material/candlestick_chart:"),
        st.Page("app_pages/d_sports.py", title="Sports ratings & win probability", icon=":material/sports_soccer:"),
    ],
}

st.navigation(pages).run()
