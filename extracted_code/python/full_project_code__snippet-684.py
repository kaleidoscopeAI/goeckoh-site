"""
Helper parse action for removing quotation marks from parsed
quoted strings.

Example::

    # by default, quotation marks are included in parsed results
    quoted_string.parse_string("'Now is the Winter of our Discontent'") # -> ["'Now is the Winter of our Discontent'"]

    # use remove_quotes to strip quotation marks from parsed results
    quoted_string.set_parse_action(remove_quotes)
    quoted_string.parse_string("'Now is the Winter of our Discontent'") # -> ["Now is the Winter of our Discontent"]
"""
return t[0][1:-1]


