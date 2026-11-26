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
    ],
    "Use it: domain demos": [
        st.Page("app_pages/d_vehicle.py", title="Car GPS + IMU fusion", icon=":material/directions_car:"),
        st.Page("app_pages/d_netcode.py", title="Online game lag prediction", icon=":material/sports_esports:"),
    ],
}

st.navigation(pages).run()
