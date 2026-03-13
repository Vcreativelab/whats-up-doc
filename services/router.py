"""
services/router.py

Routing logic: decides whether a user query should trigger
a search-based response or direct LLM response.

Returns structured raw data only — no summarisation or formatting.
"""

import re
from langchain_core.runnables import RunnableLambda, RunnableBranch
from services.search_engine import medical_search


# --------------------------------
# Search Decision Logic
# --------------------------------
def should_search(input_text: str) -> bool:
    """Determine if the question likely requires external medical search."""

    expanded_pattern = (
        r"\b("
        r"what is|what are|define|definition|explain|overview|"
        r"treat|treatment|manage|management|"
        r"control|controlled|"
        r"symptom|symptoms|"
        r"drug|drugs|medicine|medication|"
        r"cause|causes|"
        r"prevent|prevention|"
        r"diagnosis|diagnose|"
        r"test|tests|"
        r"therapy|therapies|"
        r"dose|dosing|"
        r"prescribe|prescribed|prescribing"
        r")\b"
    )

    return bool(re.search(expanded_pattern, input_text.lower()))


def route(input_text: str) -> str:
    """Return 'search' or 'direct' depending on query type."""
    return "search" if should_search(input_text) else "direct"


# --------------------------------
# Search Branch (NO summarisation)
# --------------------------------
search_branch = RunnableLambda(
    lambda x: (
        lambda result: {
            "route": "search",
            "question": x["input"],
            "history": x.get("history", []),
            "documents": result.get("sources", {}),
            "intent": result.get("intent", "general"),
            "search_debug": result.get("debug_info", {}),
        }
    )(medical_search(x["input"]))
)


# --------------------------------
# Direct LLM Branch
# --------------------------------
direct_branch = RunnableLambda(
    lambda x: {
        "route": "direct",
        "question": x["input"],
        "history": x.get("history", []),
    }
)


# --------------------------------
# Router Chain
# --------------------------------
router_chain = RunnableBranch(
    (lambda x: route(x["input"]) == "search", search_branch),
    direct_branch,
)
