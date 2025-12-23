"""Basic text normalization and similarity helpers."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize_simple(text: str) -> str:
    """Lowercase, strip, and keep alphanumeric tokens for comparison."""
    if not text:
        return ""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return " ".join(tokens)


def text_similarity(a: str, b: str) -> float:
    """Return similarity score in [0,1] using difflib's matcher."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# Compatibility alias
similarity = text_similarity
