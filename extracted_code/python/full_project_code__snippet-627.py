"""Automatically import lexers."""

def __getattr__(self, name):
    info = LEXERS.get(name)
    if info:
        _load_lexers(info[0])
        cls = _lexer_cache[info[1]]
        setattr(self, name, cls)
        return cls
    if name in COMPAT:
        return getattr(self, COMPAT[name])
    raise AttributeError(name)


