"""Return a generator for all styles by name, both builtin and plugin."""
yield from STYLE_MAP
for name, _ in find_plugin_styles():
    yield name


