"""A stack of styles."""

__slots__ = ["_stack"]

def __init__(self, default_style: "Style") -> None:
    self._stack: List[Style] = [default_style]

def __repr__(self) -> str:
    return f"<stylestack {self._stack!r}>"

@property
def current(self) -> Style:
    """Get the Style at the top of the stack."""
    return self._stack[-1]

def push(self, style: Style) -> None:
    """Push a new style on to the stack.

    Args:
        style (Style): New style to combine with current style.
    """
    self._stack.append(self._stack[-1] + style)

def pop(self) -> Style:
    """Pop last style and discard.

    Returns:
        Style: New current style (also available as stack.current)
    """
    self._stack.pop()
    return self._stack[-1]


