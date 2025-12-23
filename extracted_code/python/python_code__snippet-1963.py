renderable: RenderableType
indent: int

def __rich_console__(
    self, console: Console, options: ConsoleOptions
) -> RenderResult:
    segments = console.render(self.renderable, options)
    lines = Segment.split_lines(segments)
    for line in lines:
        yield Segment(" " * self.indent)
        yield from line
        yield Segment("\n")


