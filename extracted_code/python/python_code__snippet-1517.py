"""
Look for an encoding by its label.
This is the spec’s `get an encoding
<http://encoding.spec.whatwg.org/#concept-encoding-get>`_ algorithm.
Supported labels are listed there.

:param label: A string.
:returns:
    An :class:`Encoding` object, or :obj:`None` for an unknown label.

"""
# Only strip ASCII whitespace: U+0009, U+000A, U+000C, U+000D, and U+0020.
label = ascii_lower(label.strip('\t\n\f\r '))
name = LABELS.get(label)
if name is None:
    return None
encoding = CACHE.get(name)
if encoding is None:
    if name == 'x-user-defined':
        from .x_user_defined import codec_info
    else:
        python_name = PYTHON_NAMES.get(name, name)
        # Any python_name value that gets to here should be valid.
        codec_info = codecs.lookup(python_name)
    encoding = Encoding(name, codec_info)
    CACHE[name] = encoding
return encoding


