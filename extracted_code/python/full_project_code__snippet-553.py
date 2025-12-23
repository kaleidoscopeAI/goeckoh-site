"""
If a position is provided, move file to that point.
Otherwise, we'll attempt to record a position for future use.
"""
if pos is not None:
    rewind_body(body, pos)
elif getattr(body, "tell", None) is not None:
    try:
        pos = body.tell()
    except (IOError, OSError):
        # This differentiates from None, allowing us to catch
        # a failed `tell()` later when trying to rewind the body.
        pos = _FAILEDTELL

return pos


