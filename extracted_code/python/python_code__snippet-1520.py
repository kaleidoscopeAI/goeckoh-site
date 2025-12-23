"""
“Push”-based decoder.

:param fallback_encoding:
    An :class:`Encoding` object or a label string.
    The encoding to use if :obj:`input` does note have a BOM.
:param errors: Type of error handling. See :func:`codecs.register`.
:raises: :exc:`~exceptions.LookupError` for an unknown encoding label.

"""
def __init__(self, fallback_encoding, errors='replace'):
    # Fail early if `encoding` is an invalid label.
    self._fallback_encoding = _get_encoding(fallback_encoding)
    self._errors = errors
    self._buffer = b''
    self._decoder = None
    #: The actual :class:`Encoding` that is being used,
    #: or :obj:`None` if that is not determined yet.
    #: (Ie. if there is not enough input yet to determine
    #: if there is a BOM.)
    self.encoding = None  # Not known yet.

def decode(self, input, final=False):
    """Decode one chunk of the input.

    :param input: A byte string.
    :param final:
        Indicate that no more input is available.
        Must be :obj:`True` if this is the last call.
    :returns: An Unicode string.

    """
    decoder = self._decoder
    if decoder is not None:
        return decoder(input, final)

    input = self._buffer + input
    encoding, input = _detect_bom(input)
    if encoding is None:
        if len(input) < 3 and not final:  # Not enough data yet.
            self._buffer = input
            return ''
        else:  # No BOM
            encoding = self._fallback_encoding
    decoder = encoding.codec_info.incrementaldecoder(self._errors).decode
    self._decoder = decoder
    self.encoding = encoding
    return decoder(input, final)


