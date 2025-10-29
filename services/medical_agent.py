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


medical_prompt = ChatPromptTemplate.from_template("""
You are **DocBot**, a knowledgeable and empathetic medical assistant.

Your tone should be:
- **Professional and factual**, not casual.
- **Clear and simple**, avoiding unnecessary jargon.
- **Neutral** and **evidence-based** — do not speculate or invent facts.

When answering:
- Give **concise explanations** of the condition or topic.
- Include **possible causes, symptoms, treatments, or precautions** when relevant.
- Avoid recommending specific drugs unless universally accepted.
- End with a **disclaimer** reminding users to consult a healthcare professional.

---

Conversation history:
{history}

User question and context:
{input}
""")

medical_runnable = (
    medical_prompt
    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    | StrOutputParser()
)


def get_medical_answer(query: str) -> str:
    """Generate multilingual, evidence-based medical response."""
    final_response = None
    tokens_this_request = max(len(query) // 4, 1)
    if is_rate_limited(tokens_this_request):
        return "⚠️ Rate limit exceeded. Please wait a bit."

    try:
        lang_info = detect_and_translate(query)
        user_lang = lang_info["language"]
        translated_query = lang_info["translation"]

        memory = init_memory()
        context = {"input": translated_query, "history": memory.chat_memory.messages}
        routed_input = router_chain.invoke(context)

        if isinstance(routed_input, dict) and (
            "Verified medical information" in routed_input.get("input", "") or
            "Sources referenced" in routed_input.get("input", "")
        ):
            final_response = routed_input.get("input")
        else:
            english_response = medical_runnable.invoke(routed_input)
            final_response = f"""**Question:** {query}    

**Answer:**  
{english_response}  

---

⚠️ *This information is for educational purposes only and should not replace professional medical advice.*"""

        # Translate back if non-English
        if user_lang.lower() != "en":
            translated_back = translate_back_to_original_language(final_response, user_lang)
            final_response = f"*Translated from English to {user_lang}*\n\n{translated_back}"

    except Exception as e:
        st.error(f"⚠️ Error generating answer: {e}")
        final_response = f"⚠️ Failed to generate an answer: {e}"

    if not final_response:
        final_response = "⚠️ No answer generated."

    return final_response.strip()
