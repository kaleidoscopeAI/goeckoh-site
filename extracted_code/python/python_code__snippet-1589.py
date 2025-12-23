"""A renderable that inserts a control code (non printable but may move cursor).

Args:
    *codes (str): Positional arguments are either a :class:`~rich.segment.ControlType` enum or a
        tuple of ControlType and an integer parameter
"""

__slots__ = ["segment"]

def __init__(self, *codes: Union[ControlType, ControlCode]) -> None:
    control_codes: List[ControlCode] = [
        (code,) if isinstance(code, ControlType) else code for code in codes
    ]
    _format_map = CONTROL_CODES_FORMAT
    rendered_codes = "".join(
        _format_map[code](*parameters) for code, *parameters in control_codes
    )
    self.segment = Segment(rendered_codes, None, control_codes)

@classmethod
def bell(cls) -> "Control":
    """Ring the 'bell'."""
    return cls(ControlType.BELL)

@classmethod
def home(cls) -> "Control":
    """Move cursor to 'home' position."""
    return cls(ControlType.HOME)

@classmethod
def move(cls, x: int = 0, y: int = 0) -> "Control":
    """Move cursor relative to current position.

    Args:
        x (int): X offset.
        y (int): Y offset.

    Returns:
        ~Control: Control object.

    """

    def get_codes() -> Iterable[ControlCode]:
        control = ControlType
        if x:
            yield (
                control.CURSOR_FORWARD if x > 0 else control.CURSOR_BACKWARD,
                abs(x),
            )
        if y:
            yield (
                control.CURSOR_DOWN if y > 0 else control.CURSOR_UP,
                abs(y),
            )

    control = cls(*get_codes())
    return control

@classmethod
def move_to_column(cls, x: int, y: int = 0) -> "Control":
    """Move to the given column, optionally add offset to row.

    Returns:
        x (int): absolute x (column)
        y (int): optional y offset (row)

    Returns:
        ~Control: Control object.
    """

    return (
        cls(
            (ControlType.CURSOR_MOVE_TO_COLUMN, x),
            (
                ControlType.CURSOR_DOWN if y > 0 else ControlType.CURSOR_UP,
                abs(y),
            ),
        )
        if y
        else cls((ControlType.CURSOR_MOVE_TO_COLUMN, x))
    )

@classmethod
def move_to(cls, x: int, y: int) -> "Control":
    """Move cursor to absolute position.

    Args:
        x (int): x offset (column)
        y (int): y offset (row)

    Returns:
        ~Control: Control object.
    """
    return cls((ControlType.CURSOR_MOVE_TO, x, y))

@classmethod
def clear(cls) -> "Control":
    """Clear the screen."""
    return cls(ControlType.CLEAR)

@classmethod
def show_cursor(cls, show: bool) -> "Control":
    """Show or hide the cursor."""
    return cls(ControlType.SHOW_CURSOR if show else ControlType.HIDE_CURSOR)

@classmethod
def alt_screen(cls, enable: bool) -> "Control":
    """Enable or disable alt screen."""
    if enable:
        return cls(ControlType.ENABLE_ALT_SCREEN, ControlType.HOME)
    else:
        return cls(ControlType.DISABLE_ALT_SCREEN)

@classmethod
def title(cls, title: str) -> "Control":
    """Set the terminal window title

    Args:
        title (str): The new terminal window title
    """
    return cls((ControlType.SET_WINDOW_TITLE, title))

def __str__(self) -> str:
    return self.segment.text

def __rich_console__(
    self, console: "Console", options: "ConsoleOptions"
) -> "RenderResult":
    if self.segment.text:
        yield self.segment


