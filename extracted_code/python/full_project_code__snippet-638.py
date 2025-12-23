"""Return a generator of all filter names."""
yield from FILTERS
for name, _ in find_plugin_filters():
    yield name


