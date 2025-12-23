"""A context manager to use a temporary theme. See :meth:`~rich.console.Console.use_theme` for usage."""

def __init__(self, console: "Console", theme: Theme, inherit: bool = True) -> None:
    self.console = console
    self.theme = theme
    self.inherit = inherit

def __enter__(self) -> "ThemeContext":
    self.console.push_theme(self.theme)
    return self

def __exit__(
    self,
    exc_type: Optional[Type[BaseException]],
    exc_val: Optional[BaseException],
    exc_tb: Optional[TracebackType],
) -> None:
    self.console.pop_theme()


