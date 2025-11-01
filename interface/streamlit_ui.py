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
st.write(f"🔑 Using API key (truncated): {gemini_api_key[:6]}***")

genai.configure(api_key=gemini_api_key)

# -----------------------
# Initialize chat memory (LangChain ConversationBufferWindowMemory)
# -----------------------
# init_memory() will set st.session_state.memory to a ConversationBufferWindowMemory
memory = init_memory(k=k_value if 'k_value' in locals() else 3)
# keep a local reference for convenience
memory = st.session_state.memory

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
# -----------------------
# Process submission
# -----------------------
if submit and user_query:
    st.info("[DEBUG] submit pressed", icon="🔘")
    gif_placeholder = show_loading_gif()

    # Show a spinner and a debug message in the UI so we can see progress
    with st.spinner("Calling backend — this may take a few seconds..."):
        st.info("[DEBUG] calling get_medical_answer()", icon="🧭")
        try:
            answer = get_medical_answer(user_query)
            st.success("[DEBUG] backend returned", icon="✅")
        except Exception as e:
            gif_placeholder.empty()
            st.error(f"[ERROR] get_medical_answer failed: {e}")
            raise

    # Clear animation
    gif_placeholder.empty()

    st.markdown("### 🧠 Suggestion")
    st.markdown(answer.replace("\n", "  \n"), unsafe_allow_html=True)

    # Update memory correctly for ConversationBufferWindowMemory
    st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_query))
    st.session_state.memory.chat_memory.add_message(AIMessage(content=answer))


# -----------------------
# Display chat history
# -----------------------
if st.session_state.memory and hasattr(st.session_state.memory, "chat_memory"):
    with st.expander("🩺 View Chat History", expanded=False):
        history_md = ""
        # chat_memory.messages is a list of Message objects
        for msg in st.session_state.memory.chat_memory.messages[-10:]:
            # LangChain message classes don't always match isinstance checks across versions,
            # so compare by class name for robustness:
            if msg.__class__.__name__ == "HumanMessage":
                history_md += f"**You:** {msg.content}  \n"
            else:
                history_md += f"**DocBot:**  \n{msg.content.replace(chr(10), '  \n')}  \n\n"
        st.markdown(history_md, unsafe_allow_html=True)

