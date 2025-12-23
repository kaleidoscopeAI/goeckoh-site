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


