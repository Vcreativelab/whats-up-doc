"""
utils/formatting.py

Lightweight text normalization helpers.
Used for small, model-agnostic cleanup only.
"""

import re


def strip_question_answer_labels(text: str) -> str:
    """
    Remove redundant 'Question:' and 'Answer:' markers
    sometimes produced by LLMs.
    """
    text = re.sub(r"\*\*Question:\*\*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*Answer:\*\*", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """
    Normalize excessive blank lines and trailing spaces.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def clean_response_text(text: str) -> str:
    """
    Minimal cleanup pipeline.
    Does NOT handle disclaimers or citations.
    """
    text = strip_question_answer_labels(text)
    text = normalize_whitespace(text)
    return text
