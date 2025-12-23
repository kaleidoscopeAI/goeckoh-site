from difflib import SequenceMatcher
import re

def normalize_simple(text: str) -> str:
    """
    Converts text to a simple, canonical form for comparison.
    - Lowercases the text.
    - Removes common punctuation.
    - Strips leading/trailing whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation using a regular expression
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def text_similarity(a: str, b: str) -> float:
    """
    Calculates a similarity score between two strings.
    Returns a float between 0.0 (no similarity) and 1.0 (identical).
    Uses the Gestalt pattern matching approach from `difflib`.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
        
    return SequenceMatcher(None, a, b).ratio()-e 


