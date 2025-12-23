"""A utility class to handle a context for both a reader and a progress."""

def __init__(self, progress: "Progress", reader: _I) -> None:
    self.progress = progress
    self.reader: _I = reader

def __enter__(self) -> _I:
    self.progress.start()
    return self.reader.__enter__()

def __exit__(
    self,
    exc_type: Optional[Type[BaseException]],
    exc_val: Optional[BaseException],
    exc_tb: Optional[TracebackType],
) -> None:
    self.progress.stop()
    self.reader.__exit__(exc_type, exc_val, exc_tb)


