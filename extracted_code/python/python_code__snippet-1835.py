"""Get a representation of the environment specified by ``paths``.

This returns an Environment instance from the chosen backend based on the
given import paths. The backend must build a fresh instance representing
the state of installed distributions when this function is called.
"""
return select_backend().Environment.from_paths(paths)


