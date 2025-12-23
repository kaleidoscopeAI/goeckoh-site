"""Convert a legacy installed-files.txt path into modern RECORD path.

The legacy format stores paths relative to the info directory, while the
modern format stores paths relative to the package root, e.g. the
site-packages directory.

:param entry: Path parts of the installed-files.txt entry.
:param info: Path parts of the egg-info directory relative to package root.
:returns: The converted entry.

For best compatibility with symlinks, this does not use ``abspath()`` or
``Path.resolve()``, but tries to work with path parts:

1. While ``entry`` starts with ``..``, remove the equal amounts of parts
   from ``info``; if ``info`` is empty, start appending ``..`` instead.
2. Join the two directly.
"""
while entry and entry[0] == "..":
    if not info or info[-1] == "..":
        info += ("..",)
    else:
        info = info[:-1]
    entry = entry[1:]
return str(pathlib.Path(*info, *entry))


