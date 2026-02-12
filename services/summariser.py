"""
services/summariser.py

Summarises multi-source medical search results into clear, concise Markdown,
with enforced source grounding and citation.
"""

from core.config import DEFAULT_GEMINI_MODEL
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.formatting import clean_response_text


# --------------------------------
# Prompt Definition
# --------------------------------
summarise_prompt = ChatPromptTemplate.from_template("""
You are a **medical summarisation assistant**.

You will be given multiple medical sources, each with an ID.

Your task:
- Write a **concise, evidence-based summary** in Markdown.
- Use **bullet points** grouped under clear headings (e.g. Overview, Causes, Symptoms, Treatment).
- **Every bullet point MUST end with at least one citation** like: (Source 1) or (Source 1, Source 3)
- **Only use the provided sources. Do NOT invent sources.**
- If multiple sources say the same thing, merge them and cite all relevant sources.
- Be neutral, factual, and medical.
- End with **exactly one disclaimer** reminding users to consult a doctor.

---

**Sources:**
{sources}

**User question:**
{question}

Format your entire answer in Markdown.
""")


# --------------------------------
# Runnable Chain
# --------------------------------
summarise_runnable = (
    summarise_prompt
    | ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=0.0)
    | StrOutputParser()
)


# --------------------------------
# Helpers
# --------------------------------
def format_sources_for_prompt(sources: list) -> str:
    """Turn structured sources into a numbered block for the LLM."""
    blocks = []
    for i, src in enumerate(sources, start=1):
        title = src.get("title", "Untitled")
        snippet = src.get("snippet", "")
        blocks.append(f"[Source {i}]\nTitle: {title}\nContent: {snippet}")
    return "\n\n".join(blocks)


# --------------------------------
# Main Function
# --------------------------------
def summarise_medical_sources(sources: list, question: str) -> str:
    """
    Generate a cleaned, evidence-based medical summary with enforced citations.
    """
    try:
        formatted_sources = format_sources_for_prompt(sources)

        raw_summary = summarise_runnable.invoke({
            "sources": formatted_sources,
            "question": question
        })

        cleaned = clean_response_text(raw_summary)

        # Optional: light validation — ensure at least one citation exists
        if "(Source" not in cleaned:
            cleaned += "\n\n⚠️ *Warning: Sources could not be reliably cited in this summary.*"

        return cleaned

    except Exception as e:
        return f"⚠️ Failed to summarise sources: {e}"
