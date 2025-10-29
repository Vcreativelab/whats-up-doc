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
# Page Config
# -----------------------
st.set_page_config(page_title="What's up Doc? 🤖🩺", layout="wide")
st.title("What's up Doc? ⚕️📖")
st.caption("Empowering multilingual, evidence-based medical search and summarisation.")

# -----------------------
# Load Streamlit UI
# -----------------------
from interface import streamlit_ui  # This directly runs the full UI logic
