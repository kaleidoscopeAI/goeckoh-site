"""Remove control codes from text.

Args:
    text (str): A string possibly contain control codes.

Returns:
    str: String with control codes removed.
"""
return text.translate(_translate_table)


