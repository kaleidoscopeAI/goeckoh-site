"""Build an iterable view from the value returned by `find_matches()`."""
if callable(matches):
    return _FactoryIterableView(matches)
if not isinstance(matches, collections_abc.Sequence):
    matches = list(matches)
return _SequenceIterableView(matches)


