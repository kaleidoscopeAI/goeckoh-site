# echo_core/text_normalizer.py
from __future__ import annotations
import re
from typing import Tuple
import language_tool_python

# This tool can require a Java installation and network connection on first run.
# We wrap its initialization to prevent the entire app from crashing if it fails.
try:
    _tool = language_tool_python.LanguageTool("en-US")
except Exception as e:
    print(f"Warning: Could not initialize LanguageTool. Grammar correction will be disabled. Error: {e}")
    _tool = None

# Simple you→I patterns; this can be extended without changing callers.
_PRONOUN_PATTERNS = [
    (re.compile(r"\byou are\b", re.IGNORECASE), "I am"),
    (re.compile(r"\byou're\b", re.IGNORECASE), "I'm"),
    (re.compile(r"\byou\b", re.IGNORECASE), "I"),
    (re.compile(r"\byour\b", re.IGNORECASE), "my"),
    (re.compile(r"\byours\b", re.IGNORECASE), "mine"),
]

def normalize(text: str) -> Tuple[str, str]:
    """Return (grammar_fixed, first_person) versions of text."""
    if _tool:
        # Use a try-except block here as well, in case the tool fails during runtime
        try:
            matches = _tool.check(text)
            corrected = language_tool_python.utils.correct(text, matches)
        except Exception as e:
            print(f"Warning: LanguageTool failed to correct text. Using raw text. Error: {e}")
            corrected = text
    else:
        corrected = text

    # Force first-person framing
    first_person = corrected
    for pattern, repl in _PRONOUN_PATTERNS:
        first_person = pattern.sub(repl, first_person)

    # Capitalize first letter if missing
    if first_person and not first_person[0].isupper():
        first_person = first_person[0].upper() + first_person[1:]

    if not first_person.endswith((".", "!", "?")):
        first_person += "."
        
    return corrected, first_person
