"""
Use this method to apply an iterable of filters to
a stream. If lexer is given it's forwarded to the
filter, otherwise the filter receives `None`.
"""
def _apply(filter_, stream):
    yield from filter_.filter(lexer, stream)
for filter_ in filters:
    stream = _apply(filter_, stream)
return stream


