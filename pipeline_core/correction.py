def correct_text(text: str) -> str:
    """
    Try to run corrections against language_tool_python.
    If it's not installed, return the original text.
    """
    try:
        import language_tool_python
    except Exception:
        # No correction library available; return original
        return text
    try:
        tool = language_tool_python.LanguageTool("en-US")
        corrected = tool.correct(text)
        return corrected
    except Exception:
        return text
