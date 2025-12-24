# Cache the LanguageTool instance to avoid expensive re-initialization
_language_tool_instance = None


def correct_text(text: str) -> str:
    """
    Try to run corrections against language_tool_python.
    If it's not installed, return the original text.
    
    Performance: Caches the LanguageTool instance to avoid expensive
    re-initialization on each call.
    """
    global _language_tool_instance
    
    try:
        import language_tool_python
    except Exception:
        # No correction library available; return original
        return text
    try:
        if _language_tool_instance is None:
            _language_tool_instance = language_tool_python.LanguageTool("en-US")
        corrected = _language_tool_instance.correct(text)
        return corrected
    except Exception:
        return text
