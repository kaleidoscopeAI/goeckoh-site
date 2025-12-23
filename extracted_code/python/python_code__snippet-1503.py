"""
Returns the sequence of tag triples for the running interpreter.

The order of the sequence corresponds to priority order for the
interpreter, from most to least important.
"""

interp_name = interpreter_name()
if interp_name == "cp":
    yield from cpython_tags(warn=warn)
else:
    yield from generic_tags()

if interp_name == "pp":
    yield from compatible_tags(interpreter="pp3")
else:
    yield from compatible_tags()


