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

from core.cache_manager import translation_cache, back_translation_cache
from core.config import CACHE_TTL


def detect_and_translate(query: str) -> dict:
    """Detect language and translate non-English input to English."""
    query_key = query.strip().lower()
    if query_key in translation_cache:
        return translation_cache[query_key]

    translator_prompt = ChatPromptTemplate.from_template("""
    You are a translation assistant.
    Detect the language of this text and, if it's not English, translate it into English.
    Return strictly JSON with keys "language" and "translation".
    Text: {text}
    """)

    translator_chain = (
        translator_prompt
        | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)
        | StrOutputParser()
    )

    lang, translation = "unknown", query  # fallback
    try:
        result = translator_chain.invoke({"text": query}).strip()
        match = re.search(r"\{.*?\}", result, re.DOTALL)
        if match:
            raw_json = match.group(0)
            clean_json = raw_json.replace("'", '"').replace("\n", " ").strip()
            parsed = json.loads(clean_json)
            lang = parsed.get("language", "unknown").strip()
            translation = parsed.get("translation", query).strip()
    except Exception as e:
        st.warning(f"⚠️ Translation step failed: {e}")

    data = {"language": lang, "translation": translation}
    translation_cache[query_key] = data
    translation_cache.expire(query_key, CACHE_TTL)
    return data


def translate_back_to_original_language(text: str, target_lang: str) -> str:
    """Translate English text back to user’s original language."""
    if target_lang.lower() == "en":
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
        | ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)
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
