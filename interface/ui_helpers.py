"""
interface/ui_helpers.py

Handles the loading animation and any other future UI helpers (alerts, banners, etc.).
"""

import streamlit as st


def show_loading_gif() -> "st.delta_generator.DeltaGenerator":
    """
    Display a centered loading GIF while the model processes a query.
    Returns the placeholder so it can be cleared after completion.
    """
    gif_placeholder = st.empty()
    with gif_placeholder.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div style="text-align: center;">
                    <img src="https://github.com/Vcreativelab/whats-up-doc/blob/main/doc.gif?raw=true" 
                         width="220" style="border-radius: 10px; margin-bottom: 0.5rem;">
                    <p style="color: gray; font-size: 0.9rem;">🧠 Processing your question...</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    return gif_placeholder
