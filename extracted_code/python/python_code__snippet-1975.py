"""Return a distribution object from the given wheel URL.

This uses HTTP range requests to only fetch the portion of the wheel
containing metadata, just enough for the object to be constructed.
If such requests are not supported, HTTPRangeRequestUnsupported
is raised.
"""
with LazyZipOverHTTP(url, session) as zf:
    # For read-only ZIP files, ZipFile only needs methods read,
    # seek, seekable and tell, not the whole IO protocol.
    wheel = MemoryWheel(zf.name, zf)  # type: ignore
    # After context manager exit, wheel.name
    # is an invalid file by intention.
    return get_wheel_distribution(wheel, canonicalize_name(name))


