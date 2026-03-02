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
You are a **medical evidence summarisation assistant**.

You will receive multiple verified medical sources.

Your task is to produce a **clear, medically accurate summary**
based ONLY on the provided sources.

--------------------------------
WRITING STYLE
--------------------------------
- Write in clear medical prose.
- Use short paragraphs or concise lists when helpful.
- Prefer readability over rigid formatting.
- Avoid repetition between sections.
- Maintain a neutral, educational medical tone.

--------------------------------
STRUCTURE
--------------------------------
- Use section headings ONLY when strongly supported
  by the available evidence (e.g. Overview, Symptoms,
  Causes, Treatment, Risk Factors).
- DO NOT create sections lacking meaningful information.
- If evidence is weak or missing, OMIT the section entirely.

--------------------------------
CITATIONS
--------------------------------
- Every factual statement must be supported by citations.
- Citations should appear at the end of sentences or paragraphs.
- Multiple sources may be cited together:
  (mayoclinic.org, cdc.gov)
- DO NOT invent sources.
- DO NOT reference sources not provided.

--------------------------------
CONTENT RULES
--------------------------------
- Use ONLY the provided sources.
- Merge overlapping information naturally.
- Ignore unrelated conditions appearing in sources.
- Do NOT speculate or add medical advice.
- Do NOT mention "Source 1", "Source 2", etc.
- Integrate source websites naturally as citations.

--------------------------------
DISCLAIMER
--------------------------------
End with exactly ONE short medical disclaimer reminding
users to consult a healthcare professional.

--------------------------------

Sources:
{sources}

User question:
{question}

Return the answer in clean Markdown.
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
    Convert sources into readable evidence blocks.
    """
    blocks = []

    for domain, data in sources.items():
        snippet = data["snippet"] if isinstance(data, dict) else data

        blocks.append(
            f"Website: {domain}\n"
            f"Medical Content:\n{snippet.strip()}"
        )

    return "\n\n---\n\n".join(blocks)


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
