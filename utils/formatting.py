"""
utils/formatting.py

Helper utilities for consistent Markdown and source formatting.
"""

import textwrap


def format_sources(sources: dict) -> str:
    """Nicely format a dictionary of {source: snippet} into Markdown."""
    if not sources:
        return "_No sources found or available._"

    lines = []
    for site, snippet in sources.items():
        clean_snippet = (
            textwrap.shorten(snippet.replace("\n", " "), width=180, placeholder="...")
            if isinstance(snippet, str)
            else str(snippet)
        )
        lines.append(f"- **{site}** — {clean_snippet}")
    return "\n".join(lines)


def format_markdown_response(response: str) -> str:
    """Ensure consistent Markdown rendering."""
    response = response.strip()
    if not response.startswith("**Question:**"):
        response = f"**Response:**\n\n{response}"
    return response
