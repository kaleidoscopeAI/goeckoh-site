"""
zip manifest builder
"""

@classmethod
def build(cls, path):
    """
    Build a dictionary similar to the zipimport directory
    caches, except instead of tuples, store ZipInfo objects.

    Use a platform-specific path separator (os.sep) for the path keys
    for compatibility with pypy on Windows.
    """
    with zipfile.ZipFile(path) as zfile:
        items = (
            (
                name.replace('/', os.sep),
                zfile.getinfo(name),
            )
            for name in zfile.namelist()
        )
        return dict(items)

load = build


