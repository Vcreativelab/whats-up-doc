"""
core/memory_manager.py

Handles conversation memory (LangChain ConversationBufferWindowMemory).
"""

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory

def init_memory(k: int = 3):
    """Initialize session memory for short conversational context."""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=k)
    return st.session_state.memory
