"""Return an iterator over all lexer classes."""
for key in sorted(LEXERS):
    module_name, name = LEXERS[key][:2]
    if name not in _lexer_cache:
        _load_lexers(module_name)
    yield _lexer_cache[name]
if plugins:
    yield from find_plugin_lexers()


