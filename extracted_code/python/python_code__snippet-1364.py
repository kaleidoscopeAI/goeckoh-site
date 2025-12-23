# Issue #99: on some systems (e.g. containerised),
# sys.getfilesystemencoding() returns None, and we need a real value,
# so fall back to utf-8. From the CPython 2.7 docs relating to Unix and
# sys.getfilesystemencoding(): the return value is "the user’s preference
# according to the result of nl_langinfo(CODESET), or None if the
# nl_langinfo(CODESET) failed."
_fsencoding = sys.getfilesystemencoding() or 'utf-8'
if _fsencoding == 'mbcs':
    _fserrors = 'strict'
else:
    _fserrors = 'surrogateescape'

def fsencode(filename):
    if isinstance(filename, bytes):
        return filename
    elif isinstance(filename, text_type):
        return filename.encode(_fsencoding, _fserrors)
    else:
        raise TypeError("expect bytes or str, not %s" %
                        type(filename).__name__)

def fsdecode(filename):
    if isinstance(filename, text_type):
        return filename
    elif isinstance(filename, bytes):
        return filename.decode(_fsencoding, _fserrors)
    else:
        raise TypeError("expect bytes or str, not %s" %
                        type(filename).__name__)


