"""An object that supports the console protocol."""

def __rich_console__(
    self, console: "Console", options: "ConsoleOptions"
) -> "RenderResult":  # pragma: no cover
    ...


