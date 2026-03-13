"""
services/medical_agent.py

Main medical response generation logic — orchestrates translation,
routing, summarisation, and model responses.

UI-independent service layer.
"""

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
def get_medical_answer(query: str, debug: bool = False) -> dict:
    """
    Generate multilingual, evidence-based medical response.

    Returns structured output:
    {
        "answer": str,
        "route": str,
        "language": str,
        "error": Optional[str],
        "debug_info": Optional[dict]
    }
    """

    debug_info = {}

    # ----------------------------
    # Rate limiting
    # ----------------------------
    tokens_this_request = max(len(query) // 4, 1)
    if is_rate_limited(tokens_this_request):
        return {
            "answer": "⚠️ Rate limit exceeded. Please wait a bit.",
            "route": None,
            "language": None,
            "error": "rate_limited",
        }

    try:
        # ----------------------------
        # 1️⃣ Language detection
        # ----------------------------
        lang_info = detect_and_translate(query)
        user_lang = lang_info["language"].strip()
        translated_query = lang_info["translation"].strip()

        is_english = user_lang.lower().replace("-", "").startswith("en")

        if debug:
            debug_info["detected_language"] = user_lang
            debug_info["translated_query"] = translated_query

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

        route_type = routed.get("route")

        if debug:
            debug_info["route"] = route_type

        # ----------------------------
        # 4️⃣ Generate Response
        # ----------------------------
        if route_type == "search":

            english_response = summarise_medical_sources(
                routed.get("documents"),
                routed.get("question"),
                routed.get("intent", "general")
            )

        else:
            try:
                english_response = primary_runnable.invoke({
                    "history": routed.get("history", []),
                    "input": routed.get("question", "")
                })
            except Exception:
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

        return {
            "answer": final_response.strip(),
            "route": route_type,
            "language": user_lang,
            "error": None,
            "debug_info": debug_info if debug else None,
        }

    except Exception as e:
        return {
            "answer": f"⚠️ Failed to generate an answer: {e}",
            "route": None,
            "language": None,
            "error": str(e),
        }
