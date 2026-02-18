"""
services/medical_agent.py

Main medical response generation logic — orchestrates translation,
routing, summarisation, and model responses.

This layer does NOT format responses.
"""

import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import DEFAULT_GEMINI_MODEL, FALLBACK_GEMINI_MODEL
from core.rate_limiter import is_rate_limited
from core.memory_manager import init_memory

from services.translator import (
    detect_and_translate,
    translate_back_to_original_language,
)
from services.router import router_chain
from services.summariser import summarise_medical_sources

from utils.formatting import clean_response_text


# --------------------------------
# Prompt Definition (Direct LLM)
# --------------------------------
medical_prompt = ChatPromptTemplate.from_template("""
You are **DocBot**, a multilingual, evidence-based medical assistant.

Your role is to provide clear, structured, and informative explanations
for medical questions.

Your tone should be:
- Professional and factual
- Clear and simple
- Neutral and evidence-based

Rules:
- Use precise medical language.
- Organize response with Markdown headings
  (e.g. "### Overview", "### Causes", "### Symptoms", "### Treatment").
- Include relevant clinical context when appropriate.
- Avoid speculation and off-label drug advice.
- End with exactly one disclaimer reminding users to consult a healthcare professional.

---

Conversation history:
{history}

User question:
{input}
""")


primary_runnable = (
    medical_prompt
    | ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=0.0)
    | StrOutputParser()
)

fallback_runnable = (
    medical_prompt
    | ChatGoogleGenerativeAI(model=FALLBACK_GEMINI_MODEL, temperature=0.0)
    | StrOutputParser()
)


# --------------------------------
# Main Entry Point
# --------------------------------
def get_medical_answer(query: str) -> str:
    """Generate multilingual, evidence-based medical response."""

    debug_mode = st.sidebar.checkbox("Show debug info", value=False)

    if debug_mode:
        st.info(f"🧩 Processing query: {query[:120]}")

    # ----------------------------
    # Rate limiting
    # ----------------------------
    tokens_this_request = max(len(query) // 4, 1)
    if is_rate_limited(tokens_this_request):
        return "⚠️ Rate limit exceeded. Please wait a bit."

    try:
        # ----------------------------
        # 1️⃣ Language detection
        # ----------------------------
        lang_info = detect_and_translate(query)
        user_lang = lang_info["language"].strip()
        translated_query = lang_info["translation"].strip()

        is_english = user_lang.lower().replace("-", "").startswith("en")

        # ----------------------------
        # 2️⃣ Memory
        # ----------------------------
        memory = init_memory()

        if len(translated_query.split()) < 5 and memory.chat_memory.messages:
            last_user_msg = memory.chat_memory.messages[-1].content
            translated_query = f"{translated_query} (about: {last_user_msg})"

        context = {
            "input": translated_query,
            "history": memory.chat_memory.messages,
        }

        # ----------------------------
        # 3️⃣ Routing
        # ----------------------------
        routed = router_chain.invoke(context)

        if debug_mode:
            st.success(f"✅ Routed via: {routed.get('route')}")

        # ----------------------------
        # 4️⃣ Generate Response
        # ----------------------------
        if routed.get("route") == "search":

            english_response = summarise_medical_sources(
                routed.get("documents"),
                routed.get("question"),
            )

        else:
            try:
                english_response = primary_runnable.invoke({
                    "history": routed.get("history", []),
                    "input": routed.get("question", "")
                })
            except Exception:
                st.warning("⚠️ Primary model failed, using fallback model.")
                english_response = fallback_runnable.invoke({
                    "history": routed.get("history", []),
                    "input": routed.get("question", "")
                })

        english_response = clean_response_text(english_response)

        # ----------------------------
        # 5️⃣ Translate Back (if needed)
        # ----------------------------
        final_response = english_response

        if not is_english:
            translated_back = translate_back_to_original_language(
                english_response,
                user_lang
            )

            final_response = (
                f"*Translated from English to {user_lang}*\n\n"
                f"{translated_back}"
            )

        return final_response.strip()

    except Exception as e:
        st.error(f"⚠️ Error generating answer: {e}")
        return f"⚠️ Failed to generate an answer: {e}"
