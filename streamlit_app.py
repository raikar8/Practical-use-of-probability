"""Practical Use of Probability — entry point.

Run from the repository root:
    streamlit run streamlit_app.py

Every topic has two pages: **Theory** (read first) and **Lab** (interactive).
"""
import streamlit as st

from ssp.ui import TOPICS

st.set_page_config(page_title="Practical Use of Probability", page_icon=":material/insights:",
                   layout="wide")

ICONS = {"l0": ":material/casino:", "l1": ":material/target:", "l2": ":material/tune:",
         "l3": ":material/timeline:", "l4": ":material/hub:", "l5": ":material/scatter_plot:",
         "vehicle": ":material/directions_car:", "netcode": ":material/sports_esports:",
         "speech": ":material/graphic_eq:", "enhance": ":material/hearing:",
         "markets": ":material/candlestick_chart:", "sports": ":material/sports_soccer:",
         "l6": ":material/account_tree:", "l7": ":material/bubble_chart:", "l8": ":material/blur_on:",
         "l9": ":material/show_chart:", "bandits": ":material/ads_click:", "l10": ":material/auto_awesome:",
         "l11": ":material/compress:", "l12": ":material/swords:", "l13": ":material/grain:"}

pages = {
    "Start here": [
        st.Page("app_pages/home.py", title="Overview & how to use", icon=":material/home:", default=True),
        st.Page("app_pages/glossary.py", title="Notation & glossary", icon=":material/menu_book:"),
    ],
}
for key, (level, title, theory, lab) in TOPICS.items():
    section = f"{level} · {title}" if level != "Domain" else f"Demo · {title}"
    pages[section] = [
        st.Page(theory, title="Theory", icon=":material/menu_book:", url_path=f"{key}_theory"),
        st.Page(lab, title="Lab", icon=ICONS[key], url_path=f"{key}_lab"),
    ]

st.navigation(pages).run()
