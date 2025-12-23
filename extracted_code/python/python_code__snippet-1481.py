def _unpack_from(f, b, o=0):
    """Explicit type cast for legacy struct.unpack_from"""
    return struct.unpack_from(f, bytes(b), o)

