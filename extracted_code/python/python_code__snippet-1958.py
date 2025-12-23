"""Convert an arbitrary string to a standard 'extra' name

Any runs of non-alphanumeric characters are replaced with a single '_',
and the result is always lowercased.

This function is duplicated from ``pkg_resources``. Note that this is not
the same to either ``canonicalize_name`` or ``_egg_link_name``.
"""
return cast(NormalizedExtra, re.sub("[^A-Za-z0-9.-]+", "_", extra).lower())


