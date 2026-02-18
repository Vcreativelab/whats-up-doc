"""
services/search_engine.py

Performs cached searches on verified medical websites using DuckDuckGo,
with domain trust weighting, snippet quality scoring,
and overall confidence estimation.
"""

import re
import streamlit as st
from typing import Dict, Any
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from core.cache_manager import (
    cache,
    cache_result,
    get_cached_result,
    normalize_query_key,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

search_engine = DuckDuckGoSearchRun()
MAX_SNIPPET_LEN = 500


DOMAIN_TRUST_WEIGHTS = {
    "cdc.gov": 1.0,
    "nih.gov": 0.95,
    "mayoclinic.org": 0.9,
    "clevelandclinic.org": 0.85,
    "webmd.com": 0.75,
}


BAD_SNIPPET_PATTERNS = [
    "site owner hides",
    "enable javascript",
    "enable cookies",
    "cookie policy",
    "privacy policy",
    "subscribe",
    "sign in",
    "log in",
    "404",
    "not found",
]


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def truncate_snippet(snippet: str) -> str:
    """Trim snippet to a clean readable length."""
    snippet = snippet.strip().replace("\n", " ")
    if len(snippet) > MAX_SNIPPET_LEN:
        snippet = snippet[:MAX_SNIPPET_LEN].rsplit(" ", 1)[0] + "..."
    return snippet


def is_bad_snippet(snippet: str) -> bool:
    """Detect low-quality or irrelevant snippets."""
    if not snippet:
        return True

    s = snippet.lower().strip()

    if len(s) < 50:
        return True

    for bad in BAD_SNIPPET_PATTERNS:
        if bad in s:
            return True

    return False


def compute_snippet_quality(snippet: str, query: str) -> float:
    """
    Compute snippet quality score (0–1).
    Factors:
        - Length adequacy
        - Query term presence
    """
    snippet_lower = snippet.lower()
    query_terms = re.findall(r"\w+", query.lower())

    # Length score
    length_score = min(1.0, len(snippet) / MAX_SNIPPET_LEN)

    # Query relevance score
    term_hits = sum(1 for term in query_terms if term in snippet_lower)
    relevance_score = min(1.0, term_hits / max(1, len(query_terms)))

    return round((0.6 * length_score) + (0.4 * relevance_score), 3)


def compute_confidence(results: Dict[str, Any]) -> int:
    """
    Compute overall confidence score (0–100)
    based on weighted source quality.
    """
    if not results:
        return 0

    scores = [data["final_score"] for data in results.values()]
    avg_score = sum(scores) / len(scores)

    confidence = int(min(100, round(avg_score * 100 + (5 * len(results)))))

    return confidence


# ---------------------------------------------------------------------
# Core Search
# ---------------------------------------------------------------------

def medical_search(query: str) -> Dict[str, Any]:
    """
    Cached, trust-weighted medical search.

    Returns:
        {
            "sources": {
                domain: {
                    "snippet": str,
                    "trust_weight": float,
                    "quality_score": float,
                    "final_score": float
                }
            },
            "confidence": int
        }
    """
    query_key = normalize_query_key(query)

    cached = get_cached_result(cache, query_key)
    if cached:
        return cached

    st.caption(f"🌐 Searching verified medical sources for: **{query}**")

    structured_results = {}

    for domain, trust_weight in DOMAIN_TRUST_WEIGHTS.items():
        try:
            st.caption(f"🔎 Searching {domain} ...")

            raw = search_engine.run(f"site:{domain} {query}")

            if not raw:
                continue

            snippet = truncate_snippet(raw)

            if is_bad_snippet(snippet):
                continue

            quality_score = compute_snippet_quality(snippet, query)
            final_score = round(trust_weight * quality_score, 3)

            structured_results[domain] = {
                "snippet": snippet,
                "trust_weight": trust_weight,
                "quality_score": quality_score,
                "final_score": final_score,
            }

            st.success(f"✅ {domain} scored {final_score}")

        except Exception as e:
            st.error(f"❌ Error searching {domain}: {e}")

    confidence = compute_confidence(structured_results)

    final_payload = {
        "sources": structured_results,
        "confidence": confidence,
    }

    if structured_results:
        cache_result(cache, query_key, final_payload)
    else:
        st.warning(f"⚠️ No high-quality results found for '{query}'")

    return final_payload


# ---------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------

medical_search_tool = StructuredTool.from_function(
    func=medical_search,
    name="MedicalSearch",
    description="Searches reliable medical websites with trust weighting and confidence scoring."
)
