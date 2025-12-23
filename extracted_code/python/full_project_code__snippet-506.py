string_types = (basestring,)
integer_types = (int, long)
class_types = (type, types.ClassType)
text_type = unicode
binary_type = str

if sys.platform.startswith("java"):
    # Jython always uses 32 bits.
    MAXSIZE = int((1 << 31) - 1)
else:
    # It's possible to have sizeof(long) != sizeof(Py_ssize_t).
    class X(object):
        def __len__(self):
            return 1 << 31

    try:
        len(X())
    except OverflowError:
        # 32-bit
        MAXSIZE = int((1 << 31) - 1)
    else:
        # 64-bit
        MAXSIZE = int((1 << 63) - 1)
    del X

