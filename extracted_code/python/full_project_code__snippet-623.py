"""
Return an instance of a `Lexer` subclass that has `alias` in its
aliases list. The lexer is given the `options` at its
instantiation.

Will raise :exc:`pygments.util.ClassNotFound` if no lexer with that alias is
found.
"""
if not _alias:
    raise ClassNotFound('no lexer for alias %r found' % _alias)

# lookup builtin lexers
for module_name, name, aliases, _, _ in LEXERS.values():
    if _alias.lower() in aliases:
        if name not in _lexer_cache:
            _load_lexers(module_name)
        return _lexer_cache[name](**options)
# continue with lexers from setuptools entrypoints
for cls in find_plugin_lexers():
    if _alias.lower() in cls.aliases:
        return cls(**options)
raise ClassNotFound('no lexer for alias %r found' % _alias)


