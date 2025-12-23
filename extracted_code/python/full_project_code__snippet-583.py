"""
Converts a dNSName SubjectAlternativeName field to the form used by the
standard library on the given Python version.

Cryptography produces a dNSName as a unicode string that was idna-decoded
from ASCII bytes. We need to idna-encode that string to get it back, and
then on Python 3 we also need to convert to unicode via UTF-8 (the stdlib
uses PyUnicode_FromStringAndSize on it, which decodes via UTF-8).

If the name cannot be idna-encoded then we return None signalling that
the name given should be skipped.
"""

def idna_encode(name):
    """
    Borrowed wholesale from the Python Cryptography Project. It turns out
    that we can't just safely call `idna.encode`: it can explode for
    wildcard names. This avoids that problem.
    """
    from pip._vendor import idna

    try:
        for prefix in [u"*.", u"."]:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                return prefix.encode("ascii") + idna.encode(name)
        return idna.encode(name)
    except idna.core.IDNAError:
        return None

# Don't send IPv6 addresses through the IDNA encoder.
if ":" in name:
    return name

name = idna_encode(name)
if name is None:
    return None
elif sys.version_info >= (3, 0):
    name = name.decode("utf-8")
return name


