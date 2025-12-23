"""Context manager to capture the result of printing to the console.
See :meth:`~rich.console.Console.capture` for how to use.

Args:
    console (Console): A console instance to capture output.
"""

def __init__(self, console: "Console") -> None:
    self._console = console
    self._result: Optional[str] = None

def __enter__(self) -> "Capture":
    self._console.begin_capture()
    return self

def __exit__(
    self,
    exc_type: Optional[Type[BaseException]],
    exc_val: Optional[BaseException],
    exc_tb: Optional[TracebackType],
) -> None:
    self._result = self._console.end_capture()

def get(self) -> str:
    """Get the result of the capture."""
    if self._result is None:
        raise CaptureError(
            "Capture result is not available until context manager exits."
        )
    return self._result


