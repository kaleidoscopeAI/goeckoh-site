def ascii_lower(string):
    r"""Transform (only) ASCII letters to lower case: A-Z is mapped to a-z.

    :param string: An Unicode string.
    :returns: A new Unicode string.

    This is used for `ASCII case-insensitive
    <http://encoding.spec.whatwg.org/#ascii-case-insensitive>`_
    matching of encoding labels.
    The same matching is also used, among other things,
    for `CSS keywords <http://dev.w3.org/csswg/css-values/#keywords>`_.

    This is different from the :meth:`~py:str.lower` method of Unicode strings
    which also affect non-ASCII characters,
    sometimes mapping them into the ASCII range:

        >>> keyword = u'Bac\N{KELVIN SIGN}ground'
        >>> assert keyword.lower() == u'background'
        >>> assert ascii_lower(keyword) != keyword.lower()
        >>> assert ascii_lower(keyword) == u'bac\N{KELVIN SIGN}ground'

    """
    # This turns out to be faster than unicode.translate()
    return string.encode('utf8').lower().decode('utf8')


def lookup(label):
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


def _get_encoding(encoding_or_label):
    """
    Accept either an encoding object or label.

    :param encoding: An :class:`Encoding` object or a label string.
    :returns: An :class:`Encoding` object.
    :raises: :exc:`~exceptions.LookupError` for an unknown label.

    """
    if hasattr(encoding_or_label, 'codec_info'):
        return encoding_or_label

    encoding = lookup(encoding_or_label)
    if encoding is None:
        raise LookupError('Unknown encoding label: %r' % encoding_or_label)
    return encoding


class Encoding(object):
    """Reresents a character encoding such as UTF-8,
    that can be used for decoding or encoding.

    .. attribute:: name

        Canonical name of the encoding

    .. attribute:: codec_info

        The actual implementation of the encoding,
        a stdlib :class:`~codecs.CodecInfo` object.
        See :func:`codecs.register`.

    """
    def __init__(self, name, codec_info):
        self.name = name
        self.codec_info = codec_info

    def __repr__(self):
        return '<Encoding %s>' % self.name


