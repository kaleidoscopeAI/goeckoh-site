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


