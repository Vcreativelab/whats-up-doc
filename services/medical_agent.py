"""
services/medical_agent.py

Main medical response generation logic — orchestrates translation,
routing, summarisation, and model responses.
"""

import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core.rate_limiter import is_rate_limited
from services.translator import detect_and_translate, translate_back_to_original_language
from services.router import router_chain
from core.memory_manager import init_memory


# ─────────────────────────────────────────────
# Prompt Definition
# ─────────────────────────────────────────────
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
    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    | StrOutputParser()
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def remove_duplicate_disclaimers(text: str) -> str:
    """Remove repeated disclaimer lines or warnings."""
    seen = set()
    filtered = []
    for line in map(str.strip, text.splitlines()):
        lower_line = line.lower()
        if any(k in lower_line for k in ["⚠️", "disclaimer"]) and lower_line in seen:
            continue
        seen.add(lower_line)
        filtered.append(line)
    return "\n".join(filtered).strip()


# ─────────────────────────────────────────────
# Main Medical Answer Function
# ─────────────────────────────────────────────
def get_medical_answer(query: str) -> str:
    """Generate multilingual, evidence-based medical response."""
    # Optional debugging toggle
    debug_mode = st.sidebar.checkbox("Show debug info", value=False)
    if debug_mode:
        st.info(f"🧩 Processing query: {query[:120]}")

    final_response = None
    tokens_this_request = max(len(query) // 4, 1)
    if is_rate_limited(tokens_this_request):
        return "⚠️ Rate limit exceeded. Please wait a bit."

    try:
        # Step 1: Detect language and translate if needed
        lang_info = detect_and_translate(query)
        user_lang = lang_info["language"]
        translated_query = lang_info["translation"]

        # Step 2: Initialise short-term memory
        memory = init_memory()
        context = {"input": translated_query, "history": memory.chat_memory.messages}

        # Step 3: Route intelligently (decide search vs no-search)
        routed_input = router_chain.invoke(context)
        if debug_mode:
            st.success("✅ Translation and routing completed successfully!")

        # Step 4: If routed to summarised sources
        if isinstance(routed_input, dict) and (
            "Verified medical information" in routed_input.get("input", "") or
            "Sources referenced" in routed_input.get("input", "")
        ):
            final_response = routed_input.get("input")

        # Step 5: Otherwise, generate direct model response
        else:
            english_response = medical_runnable.invoke({
                "history": routed_input.get("history", []),
                "input": routed_input.get("input", "")
            })
            final_response = f"""**Question:** {query}    

**Answer:**  
{english_response}  

---

⚠️ *This information is for educational purposes only and should not replace professional medical advice.*"""
            final_response = remove_duplicate_disclaimers(final_response)

        # Step 6: Translate back if needed
        if user_lang.lower() != "en":
            st.success(f"🌍 Translation completed ({user_lang} → English → {user_lang}).")
            translated_back = translate_back_to_original_language(final_response, user_lang)
            final_response = f"*Translated from English to {user_lang}*\n\n{translated_back}"

    except Exception as e:
        st.error(f"⚠️ Error generating answer: {e}")
        final_response = f"⚠️ Failed to generate an answer: {e}"

    if not final_response:
        final_response = "⚠️ No answer generated."

    return final_response.strip()
