import streamlit as st
from interface.streamlit_ui import show_ui

st.set_page_config(
    page_title="What's up Doc? 🤖🩺",
    page_icon="🩺",
    layout="wide"
)

st.title("What's up Doc? ⚕️📖")
st.caption("Empowering multilingual, evidence-based medical search and summarisation.")

if __name__ == "__main__":
    show_ui()
