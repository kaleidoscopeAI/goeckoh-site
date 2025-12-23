"""
“Push”-based encoder.

:param encoding: An :class:`Encoding` object or a label string.
:param errors: Type of error handling. See :func:`codecs.register`.
:raises: :exc:`~exceptions.LookupError` for an unknown encoding label.

.. method:: encode(input, final=False)

    :param input: An Unicode string.
    :param final:
        Indicate that no more input is available.
        Must be :obj:`True` if this is the last call.
    :returns: A byte string.

"""
def __init__(self, encoding=UTF8, errors='strict'):
    encoding = _get_encoding(encoding)
    self.encode = encoding.codec_info.incrementalencoder(errors).encode


