"""Get item from a package name lookup mapping with a resolver identifier.

This extra logic is needed when the target mapping is keyed by package
name, which cannot be directly looked up with an identifier (which may
contain requested extras). Additional logic is added to also look up a value
by "cleaning up" the extras from the identifier.
"""
if identifier in mapping:
    return mapping[identifier]
# HACK: Theoretically we should check whether this identifier is a valid
# "NAME[EXTRAS]" format, and parse out the name part with packaging or
# some regular expression. But since pip's resolver only spits out three
# kinds of identifiers: normalized PEP 503 names, normalized names plus
# extras, and Requires-Python, we can cheat a bit here.
name, open_bracket, _ = identifier.partition("[")
if open_bracket and name in mapping:
    return mapping[name]
return default


