"""Raises errors or warns if called with an incompatible Wheel-Version.

pip should refuse to install a Wheel-Version that's a major series
ahead of what it's compatible with (e.g 2.0 > 1.1); and warn when
installing a version only minor version ahead (e.g 1.2 > 1.1).

version: a 2-tuple representing a Wheel-Version (Major, Minor)
name: name of wheel or package to raise exception about

:raises UnsupportedWheel: when an incompatible Wheel-Version is given
"""
if version[0] > VERSION_COMPATIBLE[0]:
    raise UnsupportedWheel(
        "{}'s Wheel-Version ({}) is not compatible with this version "
        "of pip".format(name, ".".join(map(str, version)))
    )
elif version > VERSION_COMPATIBLE:
    logger.warning(
        "Installing from a newer Wheel-Version (%s)",
        ".".join(map(str, version)),
    )


