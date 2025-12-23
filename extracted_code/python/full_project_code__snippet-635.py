"""Automatically import formatters."""

def __getattr__(self, name):
    info = FORMATTERS.get(name)
    if info:
        _load_formatters(info[0])
        cls = _formatter_cache[info[1]]
        setattr(self, name, cls)
        return cls
    raise AttributeError(name)


