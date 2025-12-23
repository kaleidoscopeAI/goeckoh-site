"""Escapes text so that it won't be interpreted as markup.

Args:
    markup (str): Content to be inserted in to markup.

Returns:
    str: Markup with square brackets escaped.
"""

def escape_backslashes(match: Match[str]) -> str:
    """Called by re.sub replace matches."""
    backslashes, text = match.groups()
    return f"{backslashes}{backslashes}\\{text}"

markup = _escape(escape_backslashes, markup)
return markup


