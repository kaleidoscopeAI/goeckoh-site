"""Encapsulates the last attempt instance right before giving up."""

def __init__(self, last_attempt: "Future") -> None:
    self.last_attempt = last_attempt
    super().__init__(last_attempt)

def reraise(self) -> "t.NoReturn":
    if self.last_attempt.failed:
        raise self.last_attempt.result()
    raise self

def __str__(self) -> str:
    return f"{self.__class__.__name__}[{self.last_attempt}]"


