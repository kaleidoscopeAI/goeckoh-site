"""Wait strategy that doesn't wait at all before retrying."""

def __init__(self) -> None:
    super().__init__(0)


