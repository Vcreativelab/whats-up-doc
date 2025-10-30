"""
interface/streamlit_ui.py

Streamlit interface for the What's up Doc? 🤖🩺 medical assistant.
"""

import os
import time
import streamlit as st
import google.generativeai as genai
from langchain.schema import AIMessage, HumanMessage

from services.medical_agent import get_medical_answer  
from interface.ui_helpers import show_loading_gif
from core.cache_manager import cache  
from core.memory_manager import init_memory
from core.config import get_gemini_api_key

# -----------------------
# Sidebar: Settings + API Key
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    k_value = st.number_input("K value", min_value=1, max_value=10, value=3)
    gemini_api_key = get_gemini_api_key()
    if st.button("🧹 Clear Cache"):
        cache.clear()
        st.success("✅ Cache cleared!")

# Configure Gemini
os.environ["GOOGLE_API_KEY"] = gemini_api_key
genai.configure(api_key=gemini_api_key)

# -----------------------
# Initialize chat memory
# -----------------------
memory = init_memory(k=k_value)

# -----------------------
# User Input Form
# -----------------------
with st.form("query_form", clear_on_submit=True):
    user_query = st.text_input("💬 Ask your medical question:")
    st.caption("🌍 Multilingual mode active — non-English questions will be translated.")
    submit = st.form_submit_button("Submit")

# -----------------------
# Process submission
# -----------------------
if submit and user_query:
    with st.spinner("🧠 Processing your question..."):
        gif_placeholder = show_loading_gif()
        answer = get_medical_answer(user_query)
        time.sleep(0.5)
        gif_placeholder.empty()

    st.markdown("### 🧠 Suggestion")
    st.markdown(answer.replace("\n", "  \n"), unsafe_allow_html=True)

    st.session_state.memory.append(HumanMessage(content=user_query))
    st.session_state.memory.append(AIMessage(content=answer))

# -----------------------
# Display chat history
# -----------------------
if st.session_state.memory:
    from interface.ui_helpers import show_chat_history
    show_chat_history(st.session_state.memory)
    
