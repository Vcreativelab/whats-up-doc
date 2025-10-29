"""
app.py

Streamlit UI for the What's up Doc? 🩺 custom medical assistant.
"""

import streamlit as st
from core.config import get_gemini_api_key
from core.cache_manager import clear_all_caches
from services.medical_agent import get_medical_answer
from utils.formatting import format_markdown_response


# -----------------------
# 🎨 Page Configuration
# -----------------------
st.set_page_config(
    page_title="What's up Doc? 🩺",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 What's up Doc?")
st.markdown("Your multilingual, evidence-based medical assistant powered by Gemini and LangChain.")

# -----------------------
# 🔐 API Key & Setup
# -----------------------
get_gemini_api_key()
clear_all_caches()

# -----------------------
# 💬 Chat Interface
# -----------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.markdown("### 💭 Ask me a health-related question:")
user_input = st.text_area("Type your question below:", placeholder="e.g., treatment for bilateral pneumonia")

if st.button("Ask Doc 🩺"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            response = get_medical_answer(user_input.strip())
            st.session_state.conversation.append({
                "question": user_input,
                "answer": response
            })
    else:
        st.warning("Please enter a question first!")

# -----------------------
# 🧠 Conversation History
# -----------------------
if st.session_state.conversation:
    st.markdown("---")
    st.markdown("### 🩵 Conversation History")
    for turn in reversed(st.session_state.conversation):
        st.markdown(f"**🧍‍♂️ You:** {turn['question']}")
        st.markdown(format_markdown_response(turn["answer"]))
        st.markdown("---")

