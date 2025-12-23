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


