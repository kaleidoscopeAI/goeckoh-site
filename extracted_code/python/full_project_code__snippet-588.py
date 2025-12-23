def __init__(self):
    self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
    self._state = GzipDecoderState.FIRST_MEMBER

def __getattr__(self, name):
    return getattr(self._obj, name)

def decompress(self, data):
    ret = bytearray()
    if self._state == GzipDecoderState.SWALLOW_DATA or not data:
        return bytes(ret)
    while True:
        try:
            ret += self._obj.decompress(data)
        except zlib.error:
            previous_state = self._state
            # Ignore data after the first error
            self._state = GzipDecoderState.SWALLOW_DATA
            if previous_state == GzipDecoderState.OTHER_MEMBERS:
                # Allow trailing garbage acceptable in other gzip clients
                return bytes(ret)
            raise
        data = self._obj.unused_data
        if not data:
            return bytes(ret)
        self._state = GzipDecoderState.OTHER_MEMBERS
        self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)


