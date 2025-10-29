"""
services/search_engine.py

Performs cached searches on verified medical websites using DuckDuckGo.
"""

import streamlit as st
from datetime import datetime
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from core.cache_manager import cache, cache_result

search_engine = DuckDuckGoSearchRun()

SAFE_SOURCES = [
    "webmd.com",
    "mayoclinic.org",
    "nih.gov",
    "cdc.gov",
    "clevelandclinic.org",
]


def medical_search(query: str):
    """Cached, source-restricted search for evidence-based medical information."""
    query_key = query.strip().lower()
    if query_key in cache:
        data = cache[query_key]
        st.info(f"🔁 Using cached results for '{query}' (last updated {data['timestamp']}).")
        return data["results"]

    st.info(f"🌐 Searching live sources for '{query}' ...")
    results = {}
    for src in SAFE_SOURCES:
        try:
            res = search_engine.run(f"site:{src} {query}")
            if res:
                results[src] = res[:400]
        except Exception as e:
            results[src] = f"Search failed ({e})"

    cache_result(cache, query_key, results)
    return results


medical_search_tool = StructuredTool.from_function(
    func=medical_search,
    name="MedicalSearch",
    description="Searches reliable medical websites for evidence-based information."
)
