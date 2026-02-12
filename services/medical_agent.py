"""
services/medical_agent.py

Main medical response generation logic — orchestrates translation, routing,
summarisation, and model responses.
"""

import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import DEFAULT_GEMINI_MODEL, FALLBACK_GEMINI_MODEL
from core.rate_limiter import is_rate_limited
from core.memory_manager import init_memory

from services.translator import detect_and_translate, translate_back_to_original_language
from services.router import router_chain

from utils.formatting import clean_response_text, format_sources


# --------------------------------
# Prompt Definition
# --------------------------------
medical_prompt = ChatPromptTemplate.from_template("""
You are **DocBot**, a multilingual, evidence-based medical assistant.

Your role is to provide **clear, structured, and informative explanations** for medical questions.

Your tone should be:
- **Professional and factual**, not casual.
- **Clear and simple**, avoiding unnecessary jargon.
- **Neutral** and **evidence-based** — do not speculate or invent facts.

When answering follow these rules strictly:
- Use **simple and precise medical language**.
- Organize the response with **clear Markdown headings** (e.g. "### Overview", "### Causes", "### Symptoms", "### Treatment").
- Include **relevant clinical context**, e.g. risk factors, diagnostic approach, and prevention tips when appropriate.
- Avoid jargon, speculation, and off-label drug advice.
- End with a **single disclaimer** reminding users to consult a healthcare professional.

Use the user’s context to maintain conversational flow.

---

**Conversation history:**
{history}

**User question and context:**
{input}
""")


# Runnable pipeline: prompt → Gemini model → plain text output
medical_runnable = (
    medical_prompt
    | ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=0.0)
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

    tokens_this_request = max(len(query) // 4, 1)
    if is_rate_limited(tokens_this_request):
        return "⚠️ Rate limit exceeded. Please wait a bit."

    try:
        # Step 1: Detect language and translate if needed
        lang_info = detect_and_translate(query)
        user_lang = lang_info["language"].strip()
        translated_query = lang_info["translation"].strip()

        normalized_lang = user_lang.lower().replace("-", "")
        is_english = normalized_lang.startswith("en")

        # Step 2: Initialise short-term memory
        memory = init_memory()
        context = {"input": translated_query, "history": memory.chat_memory.messages}

        # Step 3: Route intelligently
        routed = router_chain.invoke(context)

        if debug_mode:
            st.success(f"✅ Routed via: {routed.get('route')}")

        # Step 4: Handle search-based answer
        if routed.get("route") == "search":
            summary = clean_response_text(routed.get("summary", ""))
            sources = routed.get("sources", {})

            final_response = f"""**Question:** {query}

**Verified medical information (summarised from sources):**
{summary}

---

📚 **Sources referenced:**
{format_sources(sources)}

⚠️ *This information is for educational purposes only and should not replace professional medical advice.*
"""

        # Step 5: Handle direct LLM answer
        else:
            try:
                english_response = medical_runnable.invoke({
                    "history": routed.get("history", []),
                    "input": routed.get("question", "")
                })
            except Exception:
                # Fallback model
                st.warning("⚠️ Primary model failed, using fallback model.")
                fallback_runnable = (
                    medical_prompt
                    | ChatGoogleGenerativeAI(model=FALLBACK_GEMINI_MODEL, temperature=0.0)
                    | StrOutputParser()
                )
                english_response = fallback_runnable.invoke({
                    "history": routed.get("history", []),
                    "input": routed.get("question", "")
                })

            final_response = f"""**Question:** {query}

**Answer:**
{english_response}

---

⚠️ *This information is for educational purposes only and should not replace professional medical advice.*
"""
            final_response = clean_response_text(final_response)

        # Step 6: Translate back only if original language is not English
        if not is_english:
            st.success(f"🌍 Translation completed ({user_lang} → English → {user_lang}).")
            translated_back = translate_back_to_original_language(final_response, user_lang)
            final_response = f"*Translated from English to {user_lang}*\n\n{translated_back}"

        return final_response.strip()

    except Exception as e:
        st.error(f"⚠️ Error generating answer: {e}")
        return f"⚠️ Failed to generate an answer: {e}"
