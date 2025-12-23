"""Takes a group of renderables and returns a renderable object that renders the group.

Args:
    renderables (Iterable[RenderableType]): An iterable of renderable objects.
    fit (bool, optional): Fit dimension of group to contents, or fill available space. Defaults to True.
"""

def __init__(self, *renderables: "RenderableType", fit: bool = True) -> None:
    self._renderables = renderables
    self.fit = fit
    self._render: Optional[List[RenderableType]] = None

@property
def renderables(self) -> List["RenderableType"]:
    if self._render is None:
        self._render = list(self._renderables)
    return self._render

def __rich_measure__(
    self, console: "Console", options: "ConsoleOptions"
) -> "Measurement":
    if self.fit:
        return measure_renderables(console, options, self.renderables)
    else:
        return Measurement(options.max_width, options.max_width)

def __rich_console__(
    self, console: "Console", options: "ConsoleOptions"
) -> RenderResult:
    yield from self.renderables


