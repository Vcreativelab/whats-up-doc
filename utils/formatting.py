"""
utils/formatting.py

Helper utilities for consistent Markdown and source formatting.
"""
import textwrap
import html

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

def format_markdown_response(response: str) -> str:
    """Ensure consistent Markdown rendering."""
    response = response.strip()
    if not any(response.startswith(p) for p in ("**Question:**", "**Response:**", "**Answer:**")):
        response = f"**Response:**\n\n{response}"
    return response
