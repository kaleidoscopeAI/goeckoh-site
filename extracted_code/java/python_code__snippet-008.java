"""Find the path to the distribution's metadata directory.

HACK: This relies on importlib.metadata's private ``_path`` attribute. Not
all distributions exist on disk, so importlib.metadata is correct to not
expose the attribute as public. But pip's code base is old and not as clean,
so we do this to avoid having to rewrite too many things. Hopefully we can
eliminate this some day.
"""
return getattr(d, "_path", None)


