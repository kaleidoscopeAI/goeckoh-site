"""Parse TOML from a binary file object."""
b = __fp.read()
try:
    s = b.decode()
except AttributeError:
    raise TypeError(
        "File must be opened in binary mode, e.g. use `open('foo.toml', 'rb')`"
    ) from None
return loads(s, parse_float=parse_float)


