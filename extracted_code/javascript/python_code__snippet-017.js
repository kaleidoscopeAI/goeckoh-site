assert isinstance(marker, (list, tuple, str))

# Sometimes we have a structure like [[...]] which is a single item list
# where the single item is itself it's own list. In that case we want skip
# the rest of this function so that we don't get extraneous () on the
# outside.
if (
    isinstance(marker, list)
    and len(marker) == 1
    and isinstance(marker[0], (list, tuple))
):
    return _format_marker(marker[0])

if isinstance(marker, list):
    inner = (_format_marker(m, first=False) for m in marker)
    if first:
        return " ".join(inner)
    else:
        return "(" + " ".join(inner) + ")"
elif isinstance(marker, tuple):
    return " ".join([m.serialize() for m in marker])
else:
    return marker


