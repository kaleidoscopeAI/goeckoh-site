try:
    new_pos = src.index(expect, pos)
except ValueError:
    new_pos = len(src)
    if error_on_eof:
        raise suffixed_err(src, new_pos, f"Expected {expect!r}") from None

if not error_on.isdisjoint(src[pos:new_pos]):
    while src[pos] not in error_on:
        pos += 1
    raise suffixed_err(src, pos, f"Found invalid character {src[pos]!r}")
return new_pos


