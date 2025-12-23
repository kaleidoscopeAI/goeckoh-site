"""A context manager that 'pages' content. See :meth:`~rich.console.Console.pager` for usage."""

def __init__(
    self,
    console: "Console",
    pager: Optional[Pager] = None,
    styles: bool = False,
    links: bool = False,
) -> None:
    self._console = console
    self.pager = SystemPager() if pager is None else pager
    self.styles = styles
    self.links = links

def __enter__(self) -> "PagerContext":
    self._console._enter_buffer()
    return self

def __exit__(
    self,
    exc_type: Optional[Type[BaseException]],
    exc_val: Optional[BaseException],
    exc_tb: Optional[TracebackType],
) -> None:
    if exc_type is None:
        with self._console._lock:
            buffer: List[Segment] = self._console._buffer[:]
            del self._console._buffer[:]
            segments: Iterable[Segment] = buffer
            if not self.styles:
                segments = Segment.strip_styles(segments)
            elif not self.links:
                segments = Segment.strip_links(segments)
            content = self._console._render_buffer(segments)
        self.pager.show(content)
    self._console._exit_buffer()


