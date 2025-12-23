class BrotliDecoder(object):
    # Supports both 'brotlipy' and 'Brotli' packages
    # since they share an import name. The top branches
    # are for 'brotlipy' and bottom branches for 'Brotli'
    def __init__(self):
        self._obj = brotli.Decompressor()
        if hasattr(self._obj, "decompress"):
            self.decompress = self._obj.decompress
        else:
            self.decompress = self._obj.process

    def flush(self):
        if hasattr(self._obj, "flush"):
            return self._obj.flush()
        return b""


