"""Get the default representation for the current environment.

This returns an Environment instance from the chosen backend. The default
Environment instance should be built from ``sys.path`` and may use caching
to share instance state accorss calls.
"""
return select_backend().Environment.default()


