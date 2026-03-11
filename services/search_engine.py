"""
services/search_engine.py

Medical Evidence Engine

This module implements a medically-oriented evidence retrieval
pipeline designed to reduce hallucinations and improve factual
grounding when answering healthcare queries.

Moreover, this module performs evidence retrieval only and does not
generate final medical answers. Narrative synthesis is handled
by the summarisation layer.

Main Capabilities
-----------------
✔ Parallel trusted-domain retrieval
✔ Intent-aware domain trust weighting
✔ Semantic reranking using embeddings
✔ Embedding cache for performance
✔ Relaxed consensus evidence fusion
✔ Cross-source contradiction detection
✔ Hallucination safety firewall
✔ LangChain tool integration layer

Pipeline Overview
-----------------
Query
 → Intent Detection
 → Parallel Domain Search
 → Quality Scoring
 → Semantic Reranking
 → Evidence Fusion
 → Contradiction Analysis
 → Safety Firewall
 → Cached Structured Output
"""

import re
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun

from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
import requests
from bs4 import BeautifulSoup

from core.cache_manager import (
    cache,
    cache_result,
    get_cached_result,
    normalize_query_key,
)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

"""
DuckDuckGo search interface used for trusted-domain retrieval.

This search tool queries the web while restricting results to
predefined medical domains. It is used as the primary retrieval
mechanism in the evidence pipeline before semantic filtering
and reranking.
"""
search_engine = DuckDuckGoSearchRun()


@lru_cache(maxsize=1)
def load_embedding_model():
    """
    Load the sentence embedding model.

    The model is cached using LRU caching to ensure it is loaded
    only once during the application lifecycle. This significantly
    reduces startup latency and prevents repeated model loading
    during Streamlit script reruns.

    Returns
    -------
    SentenceTransformer
        Preloaded sentence embedding model used for semantic
        similarity calculations.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_model():
    """
    Retrieve the cached embedding model instance.

    This helper function provides a clean abstraction layer for
    accessing the embedding model while ensuring lazy loading
    via the cached loader.

    Returns
    -------
    SentenceTransformer
        Cached embedding model.
    """
    return load_embedding_model()


@lru_cache(maxsize=2048)
def get_embedding(text: str):
    """
    Compute and cache a normalized embedding vector for text.

    Embeddings are cached using an LRU strategy to avoid repeated
    inference for identical inputs. This dramatically improves
    performance during semantic similarity comparisons where
    the same sentences may be evaluated multiple times.

    Parameters
    ----------
    text : str
        Input sentence or snippet to encode.

    Returns
    -------
    ndarray
        Normalized sentence embedding vector suitable for
        cosine similarity calculations.
    """
    model = get_model()

    return model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
   

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MAX_SNIPPET_LEN = 500
"""Maximum characters retained from retrieved search snippets."""

MAX_WORKERS = 5
"""Maximum parallel domain searches."""

BASE_DOMAIN_TRUST = {
    "cdc.gov": 1.0,
    "nih.gov": 0.95,
    "mayoclinic.org": 0.9,
    "clevelandclinic.org": 0.85,
    "webmd.com": 0.75,
}
"""
Baseline trust scores assigned to medical domains.
Higher values influence ranking and consensus acceptance.
"""

NEGATION_TERMS = {
    "no", "not", "never",
    "without", "avoid",
    "contraindicated"
}
"""Terms used to detect semantic negation in medical claims."""

def similarity(a: str, b: str) -> float:
    """
    Compute cosine semantic similarity between two texts.

    Parameters
    ----------
    a : str
    b : str

    Returns
    -------
    float
        Similarity score in range [0,1].
    """
    emb_a = get_embedding(a)
    emb_b = get_embedding(b)
    return float(np.dot(emb_a, emb_b))


# ---------------------------------------------------------------------
# Intent Detection
# ---------------------------------------------------------------------

DRUG_TERMS = {"dose", "drug", "medication",
              "ibuprofen", "paracetamol",
              "antibiotic", "side effects"}

SYMPTOM_TERMS = {"symptom", "sign",
                 "pain", "fever",
                 "fatigue", "dizziness"}


def detect_intent(query: str) -> str:
    """
    Classify medical query intent.

    Categories
    ----------
    drug
        Medication or pharmacology related.
    symptom
        Symptom or condition exploration.
    general
        Default medical information.

    Returns
    -------
    str
        Detected intent label.
    """
    q = query.lower()

    if any(t in q for t in DRUG_TERMS):
        return "drug"

    if any(t in q for t in SYMPTOM_TERMS):
        return "symptom"

    return "general"


def intent_weighted_trust(intent: str) -> Dict[str, float]:
    """
    Adjust domain trust weights based on detected intent.

    Example:
        Drug queries prioritize NIH and CDC.
        Symptom queries prioritize Mayo Clinic.

    Returns
    -------
    dict
        Domain → adjusted trust score.
    """
    trust = BASE_DOMAIN_TRUST.copy()

    if intent == "drug":
        trust["nih.gov"] *= 1.15
        trust["cdc.gov"] *= 1.10

    elif intent == "symptom":
        trust["mayoclinic.org"] *= 1.15
        trust["clevelandclinic.org"] *= 1.10

    return trust


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def extract_article_text(url: str) -> str:
    """
    Download and extract readable medical text
    from a trusted source page.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = []

        for p in soup.select("article p, main p, p"):
            text = p.get_text(" ", strip=True)

            if (
                len(text) > 60
                and not text.lower().startswith(("cookie", "privacy", "subscribe"))
            ):
                paragraphs.append(text)

        article_text = " ".join(paragraphs)

        return article_text[:3000]  # limit size

    except Exception:
        return ""


def extract_url(raw_result: str) -> str | None:
    match = re.search(r"https?://[^\s]+", raw_result)
    return match.group(0) if match else None


def truncate(snippet):
    """
    Normalize and limit snippet length.

    Prevents extremely long search outputs from
    dominating scoring.
    """
    snippet = snippet.replace("\n", " ").strip()
    return snippet[:MAX_SNIPPET_LEN]


def split_sentences(text):
    """
    Extract meaningful sentences from text.

    Very short fragments are discarded to reduce noise.
    """
    return [
        s.strip()
        for s in re.split(r"[.!?]", text)
        if len(s.strip()) > 40
    ]


def contains_negation(text):
    """
    Detect whether a sentence expresses negation.

    Used for contradiction detection between sources.
    """
    tokens = re.findall(r"\w+", text.lower())
    return any(t in NEGATION_TERMS for t in tokens)


def compute_quality(snippet, query):
    """
    Estimate snippet informational quality.

    Combines:
    - snippet length
    - keyword overlap with query

    Returns
    -------
    float
        Quality score ∈ [0,1].
    """
    terms = re.findall(r"\w+", query.lower())
   
    snippet_tokens = re.findall(r"\w+", snippet.lower())
    hits = sum(t in snippet_tokens for t in terms)

    return min(
        1.0,
        0.6 * len(snippet) / MAX_SNIPPET_LEN +
        0.4 * hits / max(1, len(terms))
    )


def confidence(scores):
    """
    Convert ranking scores into confidence percentage.

    Returns
    -------
    int
        Confidence value (0–100).
    """
    if not scores:
        return 0
    return int(min(100, max(0, sum(scores)/len(scores)*100)))

def recency_boost(text: str) -> float:
    """
    Apply a small score boost if the snippet
    appears to reference recent medical information.
    """
    years = re.findall(r"\b(20[2-3][0-9])\b", text)

    if not years:
        return 1.0

    latest = max(int(y) for y in years)

    if latest >= 2023:
        return 1.1
    if latest >= 2020:
        return 1.05

    return 1.0


# ---------------------------------------------------------------------
# Parallel Search
# ---------------------------------------------------------------------
def search_domain(domain, trust, queries):
    """
    Perform domain-restricted search across query variants.

    Selects the highest scoring snippet retrieved
    from a trusted medical domain.

    Returns
    -------
    tuple
        (domain, best_result_dict)
    """
    best = None
    
    for q in queries:
        try:
            raw = search_engine.run(f"site:{domain} {q}")
        except Exception:
            continue

        if not raw:
            continue

        url = extract_url(raw)

        if url:
            article = extract_article_text(url)
        else:
            article = ""
        
        snippet = truncate(article if article else raw)
     
        # Compute snippet score
        qscore = compute_quality(snippet, q)
        score = trust * qscore * recency_boost(snippet)

        cand = {
            "snippet": snippet,
            "score": score,
            "trust": trust
        }

        if not best or score > best["score"]:
            best = cand

    return domain, best


# ---------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------
def rerank(query, results):
    """
    Apply semantic reranking using embedding similarity.

    Final score =
        60% retrieval score +
        40% semantic similarity
    """
    ranked = {}

    for d, r in results.items():

        sem = similarity(query, r["snippet"])

        ranked[d] = {
            **r,
            "rerank": 0.6 * r["score"] + 0.4 * sem
        }

    return dict(sorted(
        ranked.items(),
        key=lambda x: x[1]["rerank"],
        reverse=True
    ))


# ---------------------------------------------------------------------
# Relaxed Consensus Fusion
# ---------------------------------------------------------------------
def fuse(sources):
    """
    Merge evidence sentences supported by multiple sources.

    Acceptance Rule
    ----------------
    ✔ Agreement across ≥2 domains
    OR
    ✔ Very high trust source (>0.9)

    Returns
    -------
    str
        Consolidated evidence summary.
    """

    sentences = []
    owners = []
    trusts = []

    # Gather sentences from all sources
    for d, data in sources.items():
        txt = data["snippet"]
        trust = data["trust"]

        for s in split_sentences(txt):
            sentences.append(s)
            owners.append(d)
            trusts.append(trust)
    
    # Initialize fused
    fused = []   

    # Precompute embeddings once for efficiency
    if not sentences:
        return ""
    
    embeddings = np.array([get_embedding(s) for s in sentences])

    for i in range(len(sentences)):
        agree = {owners[i]}  # initialize agreement set

        # Prevent empty similarity computation
        if i + 1 >= len(embeddings):
            if trusts[i] > 0.9:
                fused.append(sentences[i])
            continue

        sims = np.dot(embeddings[i], embeddings[i+1:].T)

        for j, sim in enumerate(sims, start=i+1):
            if sim > 0.65:
                agree.add(owners[j])

        if len(agree) >= 2 or trusts[i] > 0.9:
            fused.append(sentences[i])
         
    # Remove duplicates AFTER fusion
    fused = list(dict.fromkeys(fused))
 
    return " ".join(fused[:12])


# ---------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------
def contradictions(sources):
    """
    Detect semantic contradictions between domains.

    A contradiction occurs when:
        • Sentences are semantically similar
        • One contains negation and the other does not

    Returns
    -------
    list
        Pairs of conflicting domains.
    """
    issues = []
    dom = list(sources.keys())

    for i in range(len(dom)):
        for j in range(i + 1, len(dom)):

            s1 = split_sentences(
                sources[dom[i]]["snippet"]
            )
            s2 = split_sentences(
                sources[dom[j]]["snippet"]
            )

            for a in s1:
                for b in s2:

                    if similarity(a, b) < 0.7:
                        continue

                    if contains_negation(a) != \
                       contains_negation(b):

                        issues.append(
                            (dom[i], dom[j])
                        )

    return issues


# ---------------------------------------------------------------------
# Hallucination Firewall
# ---------------------------------------------------------------------
def firewall(fused, conf):
    """
    Apply safety validation before answering.

    Rejects weak or insufficient evidence.

    Returns
    -------
    tuple(bool, int)
        safe_to_answer, adjusted_confidence
    """
    if not fused:
        return False, int(conf * 0.5)

    if len(fused.split()) < 40:
        return False, int(conf * 0.7)

    return True, conf


# ---------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------
def medical_search(query: str) -> Dict[str, Any]:
    """
    Execute full medical evidence retrieval pipeline.

    Steps
    -----
    1. Cache lookup
    2. Intent detection
    3. Parallel trusted search
    4. Semantic reranking
    5. Consensus fusion
    6. Contradiction detection
    7. Hallucination firewall
    8. Result caching

    Output Structure
    -----
    medical_search() returns a dictionary containing:

    	fused_evidence : str
        	Consensus medical evidence extracted from sources.

    	sources : dict
       	 Ranked trusted-domain snippets used for summarisation.

    	confidence : int
       	 Evidence confidence score (0–100).

    	safe_to_answer : bool
        	Indicates whether sufficient evidence was found.

   	 intent : str
       	 Detected medical intent category.

    	debug_info : dict
        	Diagnostic metadata (e.g. contradiction signals).  
    """

    key = normalize_query_key(query)
    cached = get_cached_result(cache, key)
    if cached:
        return cached

    intent = detect_intent(query)
    trust_weights = intent_weighted_trust(intent)

    queries = [
        query,
        f"{query} symptoms",
        f"{query} treatment",
        f"{query} causes",
    ]

    results = {}

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as ex:

        futures = [
            ex.submit(
                search_domain,
                d,
                t,
                queries
            )
            for d, t in trust_weights.items()
        ]

        for f in as_completed(futures):
            d, r = f.result()
            if r:
                results[d] = r

    ranked = rerank(query, results)

    scores = [
        r["rerank"]
        for r in ranked.values()
    ]

    conf = confidence(scores)

    fused = fuse(ranked)

    issues = contradictions(ranked)

    if issues:
        conf = int(conf * 0.75)

    safe, conf = firewall(fused, conf)

    payload = {
        "fused_evidence": fused,
        "sources": ranked,
        "confidence": conf,
        "safe_to_answer": safe,
        "intent": intent,
        "debug_info": {
            "contradictions": issues
        }
    }

    if ranked:
        cache_result(cache, key, payload)

    return payload


# ---------------------------------------------------------------------
# Tool Integration
# ---------------------------------------------------------------------
medical_search_tool = StructuredTool.from_function(
    func=medical_search,
    name="MedicalSearch",
    description="Medical evidence grounded retrieval engine."
)
