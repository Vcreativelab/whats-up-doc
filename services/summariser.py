"""
services/summariser.py

Summarises multi-source medical search results into clear, concise Markdown.
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

summarise_prompt = ChatPromptTemplate.from_template("""
You are a medical summarisation assistant.

Summarise the following **verified medical information** clearly and concisely.

**Instructions:**
- Use **short Markdown bullet points** for key facts.
- For each key fact, include the **source in parentheses**.
- Focus on **definitions, causes, symptoms, and treatments/medications**.
- Combine overlapping information across sources (no repetition).
- Avoid quoting full paragraphs — summarise meaningfully.
- Finish with a one-line disclaimer reminding users to consult a doctor.

---

**Collected information from sources:**
{sources}

**User question:**
{question}

Format your entire answer in Markdown.
""")

summarise_runnable = (
    summarise_prompt
    | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)
    | StrOutputParser()
)
