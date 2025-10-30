"""
app.py — What's up Doc? 🤖🩺
Entry point for the Custom Medical Agent.
"""

import os
import sys
import streamlit as st

# -----------------------
# Path Setup
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# -----------------------
# Page Config — only once
# -----------------------
st.set_page_config(
    page_title="What's up Doc? 🤖🩺",
    page_icon="🩺",
    layout="wide"
)
st.title("What's up Doc? ⚕️📖")
st.caption("Empowering multilingual, evidence-based medical search and summarisation.")

# -----------------------
# Load Main UI
# -----------------------
if __name__ == "__main__":
    from interface import streamlit_ui  # Executes the main UI
