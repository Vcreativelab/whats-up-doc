"""
services/summariser.py

Summarises multi-source medical search results into clear, concise Markdown,
with enforced citation and a single disclaimer.
"""

import re
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
- Use **clear section headings** (e.g. Overview, Causes, Symptoms, Treatment) only if supported by the sources.
- Use **bullet points** under each heading.
- **Every bullet point MUST end with at least one citation** like: (cdc.gov) or (cdc.gov, nih.gov)
- Only include information directly relevant to the user's question.
- Ignore unrelated medical conditions even if present in the same source.
- Only use the provided sources. Do NOT invent sources.
- Merge overlapping information and cite all relevant sources.
- Be neutral, factual, and medical.
- End with exactly one disclaimer reminding users to consult a doctor.

---

Sources:
{sources}

User question:
{question}

Return your answer in Markdown.
""")


# --------------------------------
# Runnable Chain
# --------------------------------
summarise_runnable = (
    summarise_prompt
    | ChatGoogleGenerativeAI(
        model=DEFAULT_GEMINI_MODEL,
        temperature=0.0
    )
    | StrOutputParser()
)


# --------------------------------
# Helpers
# --------------------------------
def format_sources_for_prompt(sources: dict) -> str:
    """
    Convert {domain: snippet} into domain-labelled blocks.
    """
    blocks = []
    for domain, snippet in sources.items():
        blocks.append(
            f"[{domain}]\n"
            f"{str(snippet).strip()}"
        )
    return "\n\n".join(blocks)


STANDARD_DISCLAIMER = (
    "⚠️ *This information is for educational purposes only and does not "
    "replace professional medical advice. Please consult a doctor.*"
)


def enforce_single_disclaimer(text: str) -> str:
    """
    Remove any existing disclaimer variants and append exactly one.
    """

    text = re.sub(
        r"⚠️?\s*\*?This information[^*]*consult[^*]*doctor\*?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip() + "\n\n" + STANDARD_DISCLAIMER


# --------------------------------
# Main Function
# --------------------------------
def summarise_medical_sources(sources: dict, question: str) -> str:
    """
    Generate a clean, structured, citation-grounded medical summary.
    """

    try:
        if not sources or not isinstance(sources, dict):
            return "⚠️ No reliable sources were found for this query."

        formatted_sources = format_sources_for_prompt(sources)

        raw_summary = summarise_runnable.invoke({
            "sources": formatted_sources,
            "question": question
        })

        cleaned = clean_response_text(raw_summary)
        cleaned = enforce_single_disclaimer(cleaned)

        return cleaned

    except Exception as e:
        return f"⚠️ Failed to summarise sources: {e}"
