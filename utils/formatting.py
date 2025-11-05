"""
utils/formatting.py

Utility functions to normalize, clean, and deduplicate model responses.
Ensures consistent source formatting and prevents redundant disclaimers or labels
"""
import textwrap
import html
import re


# --- Constants ---
DISCLAIMER_LINE = (
    "⚠️ *This information is for educational purposes only and should not replace professional medical advice.*"
)


# --- Cleaning Helpers ---
def format_sources(sources: dict) -> str:
    """Nicely format a dictionary of {source: snippet} into Markdown."""
    if not sources:
        return "_No sources found or available._"

    lines = []
    for site, snippet in sources.items():
        if not isinstance(snippet, str):
            snippet = str(snippet)
        clean_snippet = html.escape(snippet.replace("\n", " "))
        shortened = textwrap.shorten(clean_snippet, width=180, placeholder="...")
        lines.append(f"- **{site}** — {shortened}")
    return "\n\n".join(lines)


def strip_question_answer(text: str) -> str:
    """Remove redundant 'Question:' and 'Answer:' labels from responses."""
    text = re.sub(r"\*\*Question:\*\*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*Answer:\*\*", "", text, flags=re.IGNORECASE)
    return text.strip()


def remove_duplicate_disclaimers(text: str) -> str:
    """Remove repeated disclaimer or warning lines."""
    seen = set()
    filtered = []
    for line in map(str.strip, text.splitlines()):
        lower_line = line.lower()
        if any(k in lower_line for k in ["⚠️", "disclaimer"]) and lower_line in seen:
            continue
        seen.add(lower_line)
        filtered.append(line)
    return "\n".join(filtered).strip()


def ensure_single_disclaimer(text: str) -> str:
    """Ensure exactly one disclaimer is present; add it if missing."""
    text = remove_duplicate_disclaimers(text)
    if "⚠️" not in text and "disclaimer" not in text.lower():
        text += f"\n\n---\n\n{DISCLAIMER_LINE}"
    return text.strip()


def clean_response_text(text: str) -> str:
    """
    Full cleanup pipeline:
    - Strip redundant question/answer markers
    - Remove duplicate disclaimers
    - Ensure a single final disclaimer
    """
    text = strip_question_answer(text)
    text = remove_duplicate_disclaimers(text)
    text = ensure_single_disclaimer(text)
    return text.strip()
