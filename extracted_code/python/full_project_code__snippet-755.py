"""Stream decodes an iterator."""

if r.encoding is None:
    yield from iterator
    return

decoder = codecs.getincrementaldecoder(r.encoding)(errors="replace")
for chunk in iterator:
    rv = decoder.decode(chunk)
    if rv:
        yield rv
rv = decoder.decode(b"", final=True)
if rv:
    yield rv


