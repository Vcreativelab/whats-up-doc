"""
core/rate_limiter.py

Simple token-based rate limiter to prevent excessive model calls.
"""

import time
from typing import List, Tuple

from core.config import RATE_LIMIT_SECONDS, MAX_TOKENS_PER_MINUTE

# Stores tuples of (timestamp, token_count)
request_tokens: List[Tuple[float, int]] = []

def is_rate_limited(tokens_this_request: int) -> bool:
    """Return True if the request exceeds allowed token rate."""
    global request_tokens
    now = time.time()

    # Remove old requests outside the time window
    request_tokens = [(t, tok) for (t, tok) in request_tokens if now - t < RATE_LIMIT_SECONDS]
    tokens_used = sum(tok for _, tok in request_tokens)

    if tokens_used + tokens_this_request > MAX_TOKENS_PER_MINUTE:
        return True

    request_tokens.append((now, tokens_this_request))
    return False
