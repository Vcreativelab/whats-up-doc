"""
services/summariser.py

Summarises multi-source medical search results into clear, concise Markdown,
with built-in cleanup for disclaimers and formatting.
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.formatting import clean_response_text


# --------------------------------
# Prompt Definition
# --------------------------------
summarise_prompt = ChatPromptTemplate.from_template("""
You are a **medical summarisation assistant**.

Your goal is to produce a concise, evidence-based summary of verified medical search results.

**Guidelines:**
- Use **short Markdown bullet points** for each key fact.
- Include the **source name in parentheses** for each point.
- Focus on **definitions, causes, symptoms, and treatments/medications**.
- Combine overlapping facts and avoid repetition.
- Write neutrally and factually — do not speculate.
- End with a **single disclaimer** reminding users to consult a doctor.

---

**Collected medical information from verified sources:**
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
    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    | StrOutputParser()
)


# --------------------------------
# Helper Function
# --------------------------------
def summarise_medical_sources(sources: str, question: str) -> str:
    """
    Generate a cleaned, evidence-based medical summary.
    Ensures consistent formatting and single disclaimer.
    """
    try:
        raw_summary = summarise_runnable.invoke({
            "sources": sources,
            "question": question
        })
        cleaned = clean_response_text(raw_summary)
        return cleaned
    except Exception as e:
        return f"⚠️ Failed to summarise sources: {e}"
