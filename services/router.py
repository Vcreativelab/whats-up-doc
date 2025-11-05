"""
services/router.py

Routing logic: decides whether a user query should trigger
a search-based summarisation or direct LLM response.
"""

import re
from langchain_core.runnables import RunnableLambda, RunnableBranch
from services.summariser import summarise_medical_sources
from services.search_engine import medical_search
from utils.formatting import format_sources


def should_search(input_text: str) -> bool:
    """Determine if the question requires external medical search."""
    pattern = r"\b(treat(ment)?|symptom(s)?|drug(s)?|medication(s)?|cause(s)?|prevent(ion)?|diagnos(e|is)|test(s)?|therapy|dose|prescrib(e|ed|ing))\b"
    return bool(re.search(pattern, input_text.lower()))


def route(input_text: str) -> str:
    """Return 'search' or 'no_search' depending on query type."""
    return "search" if should_search(input_text) else "no_search"


def enrich_with_question_and_history(prev, original):
    return {
        "sources": prev,
        "question": original["input"],
        "history": original.get("history", []),
        "original": original,
    }


def summarise_with_sources(data):
    # summary = summarise_runnable.invoke({
    #    "sources": data["sources"],
    #    "question": data["question"]
    # })
    summary = summarise_medical_sources(data["sources"], data["question"])
    return {
        "summary": summary,
        "sources": data["sources"],
        "original": data["original"],
    }


def enrich_final_summary(data):
    original = data["original"]
    return {
        "input": f"""**Question:** {original['input']}  

**Verified medical information (summarised from sources):**  
{data['summary']}  

---

📚 **Sources referenced:**  
{format_sources(data['sources'])}""",
        "history": original.get("history", []),
    }


search_branch = (
    RunnableLambda(lambda x: {"original": x, "results": medical_search(x["input"])} )
    | RunnableLambda(lambda d: enrich_with_question_and_history(d["results"], d["original"]))
    | RunnableLambda(lambda d: summarise_with_sources(d))
    | RunnableLambda(lambda d: enrich_final_summary(d))
)

no_search_branch = RunnableLambda(lambda x: {"input": x["input"], "history": x.get("history", [])})

router_chain = RunnableBranch(
    (lambda x: route(x["input"]) == "search", search_branch),
    no_search_branch
)
