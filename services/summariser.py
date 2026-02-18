"""
services/summariser.py

Summarises multi-source medical search results into clear, concise Markdown,
with enforced source grounding and citation.
"""

import re
from core.config import DEFAULT_GEMINI_MODEL
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.formatting import clean_response_text

REQUIRED_HEADINGS = [
    "Overview",
    "Symptoms",
    "Treatment",
]


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
def format_sources_for_prompt(sources: dict) -> str:
    """
    Turn {domain: snippet} into numbered sources for the LLM.
    """
    blocks = []
    for i, (domain, snippet) in enumerate(sources.items(), start=1):
        snippet = str(snippet).strip()
        blocks.append(
            f"[Source {i}]\n"
            f"Website: {domain}\n"
            f"Content: {snippet}"
        )
    return "\n\n".join(blocks)


def ensure_required_headings(text: str) -> str:
    existing_headings = re.findall(r"^##\s+(.+)", text, flags=re.MULTILINE)
    existing_lower = [h.lower().strip() for h in existing_headings]

    for heading in REQUIRED_HEADINGS:
        if heading.lower() not in existing_lower:
            text += (
                f"\n\n## {heading}\n"
                "- Information not clearly available in the provided sources. ⚠️ *(No source cited)*"
            )

    return text


def validate_bullet_citations(text: str) -> str:
    lines = text.splitlines()
    validated_lines = []

    for line in lines:
        stripped = line.strip()

        if re.match(r"^[-*•]\s+", stripped):
            if not re.search(r"\(Source\s+\d+(?:,\s*Source\s+\d+)*\)", stripped):
                stripped += " ⚠️ *(No source cited)*"

        validated_lines.append(stripped)

    return "\n".join(validated_lines)

STANDARD_DISCLAIMER = (
    "⚠️ *This information is for educational purposes only and does not "
    "replace professional medical advice. Please consult a doctor.*"
)


def enforce_single_disclaimer(text: str) -> str:

    # Remove existing disclaimers completely
    text = re.sub(
    r"⚠️?\s*\*?This information[^*]*consult[^*]*doctor\*?",
    "",
    text,
    flags=re.IGNORECASE,
    )

    # Append one clean disclaimer
    text = text.strip() + "\n\n" + STANDARD_DISCLAIMER

    return text


# --------------------------------
# Main Function
# --------------------------------
def summarise_medical_sources(sources: dict, question: str) -> str:
    """
    Generate a cleaned, evidence-based medical summary with enforced citations.
    """
    try:
        if not sources or not isinstance(sources, dict):
            return "⚠️ No valid sources available to summarise."

        formatted_sources = format_sources_for_prompt(sources)

        raw_summary = summarise_runnable.invoke({
            "sources": formatted_sources,
            "question": question
        })

        cleaned = clean_response_text(raw_summary)

        # Validate structure & citations
        cleaned = ensure_required_headings(cleaned)
        cleaned = validate_bullet_citations(cleaned)
        cleaned = enforce_single_disclaimer(cleaned)

        # Final fallback: ensure at least one citation exists
        if not re.search(r"\(Source\s+\d+", cleaned):
            cleaned += "\n\n⚠️ *Warning: Sources could not be reliably cited in this summary.*"

        return cleaned

    except Exception as e:
        return f"⚠️ Failed to summarise sources: {e}"
