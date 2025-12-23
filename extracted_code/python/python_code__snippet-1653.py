"""Render a list of lines at a given offset."""

def __init__(self, lines: List[List[Segment]], x: int, y: int) -> None:
    self._lines = lines
    self.x = x
    self.y = y

def __rich_console__(
    self, console: "Console", options: ConsoleOptions
) -> RenderResult:
    x = self.x
    move_to = Control.move_to
    for offset, line in enumerate(self._lines, self.y):
        yield move_to(x, offset)
        yield from line


