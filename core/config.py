"""
core/config.py

Central configuration and constants for the Custom Medical Agent.
"""

import os
import streamlit as st
import google.generativeai as genai

# -----------------------
# Streamlit Page Settings
# -----------------------
st.set_page_config(page_title="What's up Doc? 🤖🩺")

# -----------------------
# Global Constants
# -----------------------
RATE_LIMIT_SECONDS = 60
MAX_TOKENS_PER_MINUTE = 10_000_000
CACHE_TTL = 60 * 60 * 24 * 7  # 7 days

# -----------------------
# API Key Handling
# -----------------------
def get_gemini_api_key() -> str:
    """Fetch Google Gemini API key either from Streamlit Secrets or text input."""
    with st.sidebar:
        use_secrets = st.toggle("Use Streamlit Secrets for API Key", value=True)
        if use_secrets:
            try:
                api_key = st.secrets["GOOGLE_API_KEY"]
            except Exception:
                st.error("Missing GOOGLE_API_KEY in Streamlit secrets.")
                st.stop()
        else:
            api_key = st.text_input("Gemini API Key", type="password")
            if not api_key:
                st.warning("Please enter your Gemini API key!", icon="⚠")
                st.stop()
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    return api_key
