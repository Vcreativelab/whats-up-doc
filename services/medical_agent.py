import streamlit as st
from core.config import DEFAULT_GEMINI_MODEL, FALLBACK_GEMINI_MODEL
from utils.formatting import clean_response_text, format_sources
from langchain_google_genai import ChatGoogleGenerativeAI


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
            sources = routed.get("sources", [])

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
            except Exception as e:
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
