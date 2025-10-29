"""
core/cache_manager.py

Handles DiskCache initialization, clearing, and TTLs for search and translation results.
"""

import os
import diskcache as dc
import streamlit as st
from datetime import datetime

from core.config import CACHE_TTL

# Initialize caches
CACHE_DIR = os.path.join(os.getcwd(), "medical_cache")
TRANSLATION_CACHE_DIR = os.path.join(os.getcwd(), "translation_cache")
BACK_TRANSLATION_CACHE_DIR = os.path.join(os.getcwd(), "back_translation_cache")

cache = dc.Cache(CACHE_DIR)
translation_cache = dc.Cache(TRANSLATION_CACHE_DIR)
back_translation_cache = dc.Cache(BACK_TRANSLATION_CACHE_DIR)

def clear_all_caches():
    """Utility to clear all caches from sidebar."""
    if st.sidebar.button("Clear Cache"):
        cache.clear()
        translation_cache.clear()
        back_translation_cache.clear()
        st.success("✅ All caches cleared!")

def cache_result(cache_obj, key: str, data: dict):
    """Store result in cache with timestamp and TTL."""
    cache_obj[key] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "results": data,
    }
    cache_obj.expire(key, CACHE_TTL)
