"""
app.py

Main entry point for the Custom Medical Agent (formerly "What's up Doc? 🤖🩺").
This file bootstraps the Streamlit UI and links it to all backend services.
"""

import sys
import os
import streamlit as st

# -----------------------
# Project Path Setup
# -----------------------
# Allow relative imports when running `streamlit run app.py`
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# -----------------------
# Local Imports
# -----------------------
from interface.streamlit_ui import *  # Loads Streamlit UI & backend logic

# -----------------------
# Main Entry
# -----------------------
if __name__ == "__main__":
    st.set_page_config(page_title="What's up Doc? 🤖🩺")
    st.title("What's up Doc? ⚕️📖")
    st.caption("Empowering multilingual, evidence-based medical search and summarisation.")
    
    # Directly trigger UI (Streamlit handles reruns automatically)
    st.write("Initializing Custom Medical Agent interface...")
