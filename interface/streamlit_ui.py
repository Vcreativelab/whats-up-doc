"""
interface/streamlit_ui.py

Streamlit interface for the What's up Doc? 🤖🩺 medical assistant.
Includes sidebar controls, API key setup, GIF loader, and chat memory.
"""

import os
import time
import streamlit as st
import google.generativeai as genai
from langchain.schema import AIMessage, HumanMessage

from services.medical_agent import get_medical_answer  
from interface.ui_helpers import show_loading_gif
from core.cache_manager import cache  

# -----------------------
# Sidebar: settings + API key
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings")

    k_value = st.number_input("K value", min_value=1, max_value=10, value=3)
    use_secrets = st.toggle("Use Streamlit Secrets for API Key", value=True)

    if use_secrets:
        try:
            gemini_api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            st.error("Missing GOOGLE_API_KEY in Streamlit secrets.")
            st.stop()
    else:
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key!", icon="⚠")
            st.stop()

    # Clear cache button
    if st.button("🧹 Clear Cache"):
        cache.clear()
        st.success("✅ Cache cleared!")

# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = gemini_api_key
genai.configure(api_key=gemini_api_key)

# -----------------------
# Initialize chat memory
# -----------------------
if "memory" not in st.session_state:
    st.session_state.memory = []

# -----------------------
# User Input Form
# -----------------------
with st.form("query_form", clear_on_submit=True):
    user_query = st.text_input("💬 Ask your medical question:")
    st.caption(
        "🌍 Multilingual mode active — non-English questions will be translated "
        "to English for evidence-based sources."
    )
    submit = st.form_submit_button("Submit")

# -----------------------
# Process submission
# -----------------------
if submit and user_query:
    gif_placeholder = show_loading_gif()

    # Call the backend chain
    answer = get_medical_answer(user_query)
    time.sleep(0.5)

    # Clear animation
    gif_placeholder.empty()

    st.markdown("### 🧠 Suggestion")
    st.markdown(answer.replace("\n", "  \n"), unsafe_allow_html=True)

    # Update memory
    st.session_state.memory.append(HumanMessage(content=user_query))
    st.session_state.memory.append(AIMessage(content=answer))

# -----------------------
# Display chat history
# -----------------------
if st.session_state.memory:
    with st.expander("🩺 View Chat History", expanded=False):
        history_md = ""
        for msg in st.session_state.memory[-10:]:
            if isinstance(msg, HumanMessage):
                history_md += f"**You:** {msg.content}  \n"
            else:
                history_md += f"**DocBot:**  \n{msg.content.replace(chr(10), '  \n')}  \n\n"
        st.markdown(history_md, unsafe_allow_html=True)
