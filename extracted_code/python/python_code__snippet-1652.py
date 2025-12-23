"""A renderable to generate new line(s)"""

def __init__(self, count: int = 1) -> None:
    self.count = count

def __rich_console__(
    self, console: "Console", options: "ConsoleOptions"
) -> Iterable[Segment]:
    yield Segment("\n" * self.count)


