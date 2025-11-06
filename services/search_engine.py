"""
services/search_engine.py

Performs cached searches on verified medical websites using DuckDuckGo.
"""

import streamlit as st
from datetime import datetime
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from core.cache_manager import cache, cache_result, get_cached_result, normalize_query_key

# --------------------------------
# Configuration
# --------------------------------
search_engine = DuckDuckGoSearchRun()
MAX_SNIPPET_LEN = 500  # can raise to 800 if truncation cuts too early

SAFE_SOURCES = [
    "webmd.com",
    "mayoclinic.org",
    "nih.gov",
    "cdc.gov",
    "clevelandclinic.org",
]


# --------------------------------
# Helper
# --------------------------------
def truncate_snippet(snippet: str) -> str:
    """Ensure consistent, readable snippets."""
    snippet = snippet.strip().replace("\n", " ")
    if len(snippet) > MAX_SNIPPET_LEN:
        snippet = snippet[:MAX_SNIPPET_LEN].rsplit(" ", 1)[0] + "..."
    return snippet

# --------------------------------
# Core Search
# --------------------------------
def medical_search(query: str):
    """Cached, source-restricted search for evidence-based medical information."""
    query_key = normalize_query_key(query)

    # Check cache first
    cached = get_cached_result(cache, query_key)
    if cached:
        return cached

    # Indicate live search
    st.caption(f"🌐 Searching verified sources for: **{query}**")

    results = {}
    for src in SAFE_SOURCES:
        try:
            st.caption(f"🔎 Searching {src} ...")
            res = search_engine.run(f"site:{src} {query}")
            if res:
                results[src] = truncate_snippet(res)
                st.success(f"✅ Results found from {src}")
            else:
                st.warning(f"⚠️ No content returned from {src}")
        except Exception as e:
            results[src] = f"Search failed ({e})"
            st.error(f"❌ Error searching {src}: {e}")

    # Save to cache
    cache_result(cache, query_key, results)

    if not results:
        st.warning(f"⚠️ No results found for '{query}'")

    return results
    

# --------------------------------
# Tool registration
# --------------------------------
medical_search_tool = StructuredTool.from_function(
    func=medical_search,
    name="MedicalSearch",
    description="Searches reliable medical websites for evidence-based information."
)
