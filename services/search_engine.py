"""
services/search_engine.py

Performs cached searches on verified medical websites using DuckDuckGo,
with domain trust weighting, snippet quality scoring,
and overall confidence estimation.

UI-independent.
"""

import re
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
# Helpers
# ---------------------------------------------------------------------

def truncate_snippet(snippet: str) -> str:
    snippet = snippet.strip().replace("\n", " ")
    if len(snippet) > MAX_SNIPPET_LEN:
        snippet = snippet[:MAX_SNIPPET_LEN].rsplit(" ", 1)[0] + "..."
    return snippet


def is_bad_snippet(snippet: str) -> bool:
    if not snippet:
        return True

    s = snippet.lower().strip()

    if len(s) < 80:
        return True

    for bad in BAD_SNIPPET_PATTERNS:
        if bad in s:
            return True

    return False


def compute_snippet_quality(snippet: str, query: str) -> float:
    snippet_lower = snippet.lower()
    query_terms = re.findall(r"\w+", query.lower())

    length_score = min(1.0, len(snippet) / MAX_SNIPPET_LEN)

    term_hits = sum(1 for term in query_terms if term in snippet_lower)
    relevance_score = min(1.0, term_hits / max(1, len(query_terms)))

    return round((0.6 * length_score) + (0.4 * relevance_score), 3)


def compute_confidence(scores: list) -> int:
    if not scores:
        return 0

    avg_score = sum(scores) / len(scores)
    return int(min(100, round(avg_score * 100)))


# ---------------------------------------------------------------------
# Core Search
# ---------------------------------------------------------------------

def medical_search(query: str) -> Dict[str, Any]:

    query_key = normalize_query_key(query)
    cached = get_cached_result(cache, query_key)
    if cached:
        return cached

    cleaned_sources = {}
    debug_scores = {}
    final_scores = []
    domains_checked = []

    for domain, trust_weight in DOMAIN_TRUST_WEIGHTS.items():
        domains_checked.append(domain)

        try:
            raw = search_engine.run(f"site:{domain} {query}")

            if not raw:
                continue

            snippet = truncate_snippet(raw)

            if is_bad_snippet(snippet):
                continue

            quality_score = compute_snippet_quality(snippet, query)
            final_score = round(trust_weight * quality_score, 3)

            if final_score < 0.25:
                continue

            cleaned_sources[domain] = snippet
            debug_scores[domain] = {
                "trust_weight": trust_weight,
                "quality_score": quality_score,
                "final_score": final_score,
            }

            final_scores.append(final_score)

        except Exception:
            continue

    confidence = compute_confidence(final_scores)

    final_payload = {
        "sources": cleaned_sources,
        "confidence": confidence,
        "debug_info": {
            "scores": debug_scores,
            "domains_checked": domains_checked,
        }
    }

    if cleaned_sources:
        cache_result(cache, query_key, final_payload)

    return final_payload


# ---------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------

medical_search_tool = StructuredTool.from_function(
    func=medical_search,
    name="MedicalSearch",
    description="Searches reliable medical websites with trust weighting and confidence scoring."
)
