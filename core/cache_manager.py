"""
core/cache_manager.py

Handles DiskCache initialization, clearing, and TTLs for search and translation results.
"""

import os
import diskcache as dc
import streamlit as st
from datetime import datetime
from core.config import CACHE_TTL

# --- Initialize caches ---
BASE_DIR = os.getcwd()
CACHE_DIR = os.path.join(BASE_DIR, "medical_cache")
TRANSLATION_CACHE_DIR = os.path.join(BASE_DIR, "translation_cache")
BACK_TRANSLATION_CACHE_DIR = os.path.join(BASE_DIR, "back_translation_cache")

cache = dc.Cache(CACHE_DIR)
translation_cache = dc.Cache(TRANSLATION_CACHE_DIR)
back_translation_cache = dc.Cache(BACK_TRANSLATION_CACHE_DIR)

# --- Utilities ---
def clear_all_caches():
    """Utility to clear all caches from the sidebar."""
    if st.sidebar.button("🧹 Clear All Caches"):
        cache.clear()
        translation_cache.clear()
        back_translation_cache.clear()
        st.success("✅ All caches cleared successfully!")

def cache_result(cache_obj, key: str, data: dict):
    """Store a result in cache with timestamp and TTL."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wrapped = {"timestamp": timestamp, "results": data}
    cache_obj.set(key, wrapped, expire=CACHE_TTL)
    return wrapped

def get_cached_result(cache_obj, key: str):
    """Retrieve cached result if available and valid."""
    if key in cache_obj:
        data = cache_obj[key]
        ts = data.get("timestamp", "unknown")
        st.info(f"🔁 Using cached results for '{key}' (last updated {ts}).")
        return data["results"]
    return None
