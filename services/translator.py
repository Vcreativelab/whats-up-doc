"""
services/translator.py

Handles automatic language detection, translation to/from English,
and translation caching for multilingual support.
"""

import re
import json
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langdetect import detect, LangDetectException

from core.cache_manager import translation_cache, back_translation_cache
from core.config import CACHE_TTL, DEFAULT_GEMINI_MODEL


def detect_language_local(text: str) -> str:
    """Detect language using local library (no API cost)."""
    text = text.strip()
    if len(text) < 5:
        return "en"  # assume English for very short inputs

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def detect_and_translate(query: str) -> dict:
    """Detect language and translate non-English input to English."""
    query_key = query.strip().lower()
    if query_key in translation_cache:
        return translation_cache[query_key]

    # 1) Local language detection (cheap, fast, no tokens)
    detected_lang = detect_language_local(query)

    # Normalize language code
    normalized_lang = detected_lang.lower().replace("-", "")

    # 2) If already English → skip translation entirely
    if normalized_lang.startswith("en"):
        data = {"language": detected_lang, "translation": query}
        translation_cache[query_key] = data
        translation_cache.expire(query_key, CACHE_TTL)
        return data

    # 3) Otherwise, translate to English with LLM
    translator_prompt = ChatPromptTemplate.from_template("""
    You are a translation assistant.
    Translate the following text into English.
    Return ONLY the translated text, no explanations.

    Text:
    {text}
    """)

    translator_chain = (
        translator_prompt
        | ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=0)
        | StrOutputParser()
    )

    translation = query  # fallback

    try:
        translation = translator_chain.invoke({"text": query}).strip()
    except Exception as e:
        st.warning(f"⚠️ Translation step failed: {e}")

    data = {"language": detected_lang, "translation": translation}
    translation_cache[query_key] = data
    translation_cache.expire(query_key, CACHE_TTL)
    return data


def translate_back_to_original_language(text: str, target_lang: str) -> str:
    """Translate English text back to user’s original language."""
    if not target_lang or target_lang.lower().startswith("en"):
        return text

    cache_key = f"{target_lang.lower()}::{text.strip()}"
    if cache_key in back_translation_cache:
        return back_translation_cache[cache_key]

    translator_back_prompt = ChatPromptTemplate.from_template("""
    You are a translation assistant.
    Translate the following English text into the language specified below.
    Preserve meaning, tone, and Markdown formatting.

    Target language: {target_lang}
    Text to translate:
    {text}
    """)

    translator_back_chain = (
        translator_back_prompt
        | ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=0)
        | StrOutputParser()
    )

    try:
        translated = translator_back_chain.invoke({
            "target_lang": target_lang,
            "text": text
        }).strip()

        back_translation_cache[cache_key] = translated
        back_translation_cache.expire(cache_key, CACHE_TTL)
        return translated

    except Exception as e:
        st.warning(f"⚠️ Back-translation failed: {e}")
        return text
