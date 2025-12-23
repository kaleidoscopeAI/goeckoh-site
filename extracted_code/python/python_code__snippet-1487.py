"""
Unpack an object from `stream`.

Raises `ExtraData` when `stream` contains extra bytes.
See :class:`Unpacker` for options.
"""
data = stream.read()
return unpackb(data, **kwargs)


