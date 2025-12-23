"""Encapsulates a (future or past) attempted call to a target function."""

def __init__(self, attempt_number: int) -> None:
    super().__init__()
    self.attempt_number = attempt_number

@property
def failed(self) -> bool:
    """Return whether a exception is being held in this future."""
    return self.exception() is not None

@classmethod
def construct(cls, attempt_number: int, value: t.Any, has_exception: bool) -> "Future":
    """Construct a new Future object."""
    fut = cls(attempt_number)
    if has_exception:
        fut.set_exception(value)
    else:
        fut.set_result(value)
    return fut


