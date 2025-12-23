"""
Return the `Lexer` subclass that has `alias` in its aliases list, without
instantiating it.

Like `get_lexer_by_name`, but does not instantiate the class.

Will raise :exc:`pygments.util.ClassNotFound` if no lexer with that alias is
found.

.. versionadded:: 2.2
"""
if not _alias:
    raise ClassNotFound('no lexer for alias %r found' % _alias)
# lookup builtin lexers
for module_name, name, aliases, _, _ in LEXERS.values():
    if _alias.lower() in aliases:
        if name not in _lexer_cache:
            _load_lexers(module_name)
        return _lexer_cache[name]
# continue with lexers from setuptools entrypoints
for cls in find_plugin_lexers():
    if _alias.lower() in cls.aliases:
        return cls
raise ClassNotFound('no lexer for alias %r found' % _alias)


