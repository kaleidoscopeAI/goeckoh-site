"""When there are errors while loading a configuration file"""

def __init__(
    self,
    reason: str = "could not be loaded",
    fname: Optional[str] = None,
    error: Optional[configparser.Error] = None,
) -> None:
    super().__init__(error)
    self.reason = reason
    self.fname = fname
    self.error = error

def __str__(self) -> str:
    if self.fname is not None:
        message_part = f" in {self.fname}."
    else:
        assert self.error is not None
        message_part = f".\n{self.error}\n"
    return f"Configuration file {self.reason}{message_part}"


