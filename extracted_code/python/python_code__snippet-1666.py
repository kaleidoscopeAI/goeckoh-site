"""A marked up region in some text."""

start: int
"""Span start index."""
end: int
"""Span end index."""
style: Union[str, Style]
"""Style associated with the span."""

def __repr__(self) -> str:
    return f"Span({self.start}, {self.end}, {self.style!r})"

def __bool__(self) -> bool:
    return self.end > self.start

def split(self, offset: int) -> Tuple["Span", Optional["Span"]]:
    """Split a span in to 2 from a given offset."""

    if offset < self.start:
        return self, None
    if offset >= self.end:
        return self, None

    start, end, style = self
    span1 = Span(start, min(end, offset), style)
    span2 = Span(span1.end, end, style)
    return span1, span2

def move(self, offset: int) -> "Span":
    """Move start and end by a given offset.

    Args:
        offset (int): Number of characters to add to start and end.

    Returns:
        TextSpan: A new TextSpan with adjusted position.
    """
    start, end, style = self
    return Span(start + offset, end + offset, style)

def right_crop(self, offset: int) -> "Span":
    """Crop the span at the given offset.

    Args:
        offset (int): A value between start and end.

    Returns:
        Span: A new (possibly smaller) span.
    """
    start, end, style = self
    if offset >= end:
        return self
    return Span(start, min(offset, end), style)


