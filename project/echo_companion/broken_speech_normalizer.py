"""
Thin wrapper around the existing broken_speech_tool.py interpreter.
"""

from pathlib import Path
import sys
from typing import Any, Dict

# Try to import the existing tool from the legacy directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent / "project "
if PROJECT_DIR.exists() and str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

try:
    from broken_speech_tool import interpret as interpret_broken  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "broken_speech_tool.py not found. Ensure the legacy 'project ' directory exists "
        "or place broken_speech_tool.py alongside this file."
    ) from exc


def normalize(text: str) -> Dict[str, Any]:
    """
    Wrap the Enhanced Broken Speech Interpreter.
    Returns a dict with normalized text + intent/sentiment/notes.
    """
    result = interpret_broken(text)
    return {
        "normalized_text": result.get("normalized_text", text),
        "intent": result.get("intent", "unknown"),
        "sentiment": result.get("sentiment", "neutral"),
        "entities": result.get("entities", {}),
        "notes": result.get("notes", []),
    }
