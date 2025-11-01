import os
import streamlit as st
import google.generativeai as genai
from langchain.schema import AIMessage, HumanMessage
from services.medical_agent import get_medical_answer  
from interface.ui_helpers import show_loading_gif
from core.cache_manager import cache  
from core.memory_manager import init_memory
from core.config import get_gemini_api_key

def show_ui():
    # Sidebar
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

    # Initialize chat memory
    memory = init_memory(k=k_value)
    memory = st.session_state.memory

    # Form (wrapped inside function = safe!)
    with st.form("query_form", clear_on_submit=True):
        user_query = st.text_input("💬 Ask your medical question:")
        st.caption("🌍 Multilingual mode active — non-English questions will be translated.")
        submit = st.form_submit_button("Submit")

    if submit and user_query:
        gif_placeholder = show_loading_gif()
        with st.spinner("🧠 Processing your question..."):
            try:
                answer = get_medical_answer(user_query)
            except Exception as e:
                gif_placeholder.empty()
                st.error(f"⚠️ get_medical_answer failed: {e}")
                return
        gif_placeholder.empty()
        st.markdown(answer.replace("\n", "  \n"), unsafe_allow_html=True)
        memory.chat_memory.add_message(HumanMessage(content=user_query))
        memory.chat_memory.add_message(AIMessage(content=answer))

    if st.session_state.memory and hasattr(st.session_state.memory, "chat_memory"):
        with st.expander("🩺 View Chat History", expanded=False):
            history_md = ""
            for msg in st.session_state.memory.chat_memory.messages[-10:]:
                if msg.__class__.__name__ == "HumanMessage":
                    history_md += f"**You:** {msg.content}  \n"
                else:
                    history_md += f"**DocBot:**  \n{msg.content.replace(chr(10), '  \n')}  \n\n"
            st.markdown(history_md, unsafe_allow_html=True)
