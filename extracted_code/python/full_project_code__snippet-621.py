"""
Return the `Lexer` subclass that with the *name* attribute as given by
the *name* argument.
"""
if name in _lexer_cache:
    return _lexer_cache[name]
# lookup builtin lexers
for module_name, lname, aliases, _, _ in LEXERS.values():
    if name == lname:
        _load_lexers(module_name)
        return _lexer_cache[name]
# continue with lexers from setuptools entrypoints
for cls in find_plugin_lexers():
    if cls.name == name:
        return cls


