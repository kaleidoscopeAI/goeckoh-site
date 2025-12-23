def decode(input, fallback_encoding, errors='replace'):
    """
    Decode a single string.

    :param input: A byte string
    :param fallback_encoding:
        An :class:`Encoding` object or a label string.
        The encoding to use if :obj:`input` does note have a BOM.
    :param errors: Type of error handling. See :func:`codecs.register`.
    :raises: :exc:`~exceptions.LookupError` for an unknown encoding label.
    :return:
        A ``(output, encoding)`` tuple of an Unicode string
        and an :obj:`Encoding`.

    """
    # Fail early if `encoding` is an invalid label.
    fallback_encoding = _get_encoding(fallback_encoding)
    bom_encoding, input = _detect_bom(input)
    encoding = bom_encoding or fallback_encoding
    return encoding.codec_info.decode(input, errors)[0], encoding


def _detect_bom(input):
    """Return (bom_encoding, input), with any BOM removed from the input."""
    if input.startswith(b'\xFF\xFE'):
        return _UTF16LE, input[2:]
    if input.startswith(b'\xFE\xFF'):
        return _UTF16BE, input[2:]
    if input.startswith(b'\xEF\xBB\xBF'):
        return UTF8, input[3:]
    return None, input


def encode(input, encoding=UTF8, errors='strict'):
    """
    Encode a single string.

    :param input: An Unicode string.
    :param encoding: An :class:`Encoding` object or a label string.
    :param errors: Type of error handling. See :func:`codecs.register`.
    :raises: :exc:`~exceptions.LookupError` for an unknown encoding label.
    :return: A byte string.

    """
    return _get_encoding(encoding).codec_info.encode(input, errors)[0]


def iter_decode(input, fallback_encoding, errors='replace'):
    """
    "Pull"-based decoder.

    :param input:
        An iterable of byte strings.

        The input is first consumed just enough to determine the encoding
        based on the precense of a BOM,
        then consumed on demand when the return value is.
    :param fallback_encoding:
        An :class:`Encoding` object or a label string.
        The encoding to use if :obj:`input` does note have a BOM.
    :param errors: Type of error handling. See :func:`codecs.register`.
    :raises: :exc:`~exceptions.LookupError` for an unknown encoding label.
    :returns:
        An ``(output, encoding)`` tuple.
        :obj:`output` is an iterable of Unicode strings,
        :obj:`encoding` is the :obj:`Encoding` that is being used.

    """

    decoder = IncrementalDecoder(fallback_encoding, errors)
    generator = _iter_decode_generator(input, decoder)
    encoding = next(generator)
    return generator, encoding


def _iter_decode_generator(input, decoder):
    """Return a generator that first yields the :obj:`Encoding`,
    then yields output chukns as Unicode strings.

    """
    decode = decoder.decode
    input = iter(input)
    for chunck in input:
        output = decode(chunck)
        if output:
            assert decoder.encoding is not None
            yield decoder.encoding
            yield output
            break
    else:
        # Input exhausted without determining the encoding
        output = decode(b'', final=True)
        assert decoder.encoding is not None
        yield decoder.encoding
        if output:
            yield output
        return

    for chunck in input:
        output = decode(chunck)
        if output:
            yield output
    output = decode(b'', final=True)
    if output:
        yield output


def iter_encode(input, encoding=UTF8, errors='strict'):
    """
    “Pull”-based encoder.

    :param input: An iterable of Unicode strings.
    :param encoding: An :class:`Encoding` object or a label string.
    :param errors: Type of error handling. See :func:`codecs.register`.
    :raises: :exc:`~exceptions.LookupError` for an unknown encoding label.
    :returns: An iterable of byte strings.

    """
    # Fail early if `encoding` is an invalid label.
    encode = IncrementalEncoder(encoding, errors).encode
    return _iter_encode_generator(input, encode)


def _iter_encode_generator(input, encode):
    for chunck in input:
        output = encode(chunck)
        if output:
            yield output
    output = encode('', final=True)
    if output:
        yield output


class IncrementalDecoder(object):
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


class IncrementalEncoder(object):
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


