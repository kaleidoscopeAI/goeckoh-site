orig_stream: TextIO

@classmethod
def from_stream(cls, orig_stream: TextIO) -> "StreamWrapper":
    ret = cls()
    ret.orig_stream = orig_stream
    return ret

# compileall.compile_dir() needs stdout.encoding to print to stdout
# type ignore is because TextIOBase.encoding is writeable
@property
def encoding(self) -> str:  # type: ignore
    return self.orig_stream.encoding


