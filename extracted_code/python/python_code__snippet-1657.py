"""A context manager that enables an alternative screen. See :meth:`~rich.console.Console.screen` for usage."""

def __init__(
    self, console: "Console", hide_cursor: bool, style: StyleType = ""
) -> None:
    self.console = console
    self.hide_cursor = hide_cursor
    self.screen = Screen(style=style)
    self._changed = False

def update(
    self, *renderables: RenderableType, style: Optional[StyleType] = None
) -> None:
    """Update the screen.

    Args:
        renderable (RenderableType, optional): Optional renderable to replace current renderable,
            or None for no change. Defaults to None.
        style: (Style, optional): Replacement style, or None for no change. Defaults to None.
    """
    if renderables:
        self.screen.renderable = (
            Group(*renderables) if len(renderables) > 1 else renderables[0]
        )
    if style is not None:
        self.screen.style = style
    self.console.print(self.screen, end="")

def __enter__(self) -> "ScreenContext":
    self._changed = self.console.set_alt_screen(True)
    if self._changed and self.hide_cursor:
        self.console.show_cursor(False)
    return self

def __exit__(
    self,
    exc_type: Optional[Type[BaseException]],
    exc_val: Optional[BaseException],
    exc_tb: Optional[TracebackType],
) -> None:
    if self._changed:
        self.console.set_alt_screen(False)
        if self.hide_cursor:
            self.console.show_cursor(True)


