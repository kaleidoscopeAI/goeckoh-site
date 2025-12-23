KEYWORDS: ClassVar[Optional[List[str]]] = []

def __init__(self, stream: Optional[TextIO], no_color: bool) -> None:
    super().__init__(
        console=Console(file=stream, no_color=no_color, soft_wrap=True),
        show_time=False,
        show_level=False,
        show_path=False,
        highlighter=NullHighlighter(),
    )

# Our custom override on Rich's logger, to make things work as we need them to.
def emit(self, record: logging.LogRecord) -> None:
    style: Optional[Style] = None

    # If we are given a diagnostic error to present, present it with indentation.
    assert isinstance(record.args, tuple)
    if getattr(record, "rich", False):
        (rich_renderable,) = record.args
        assert isinstance(
            rich_renderable, (ConsoleRenderable, RichCast, str)
        ), f"{rich_renderable} is not rich-console-renderable"

        renderable: RenderableType = IndentedRenderable(
            rich_renderable, indent=get_indentation()
        )
    else:
        message = self.format(record)
        renderable = self.render_message(record, message)
        if record.levelno is not None:
            if record.levelno >= logging.ERROR:
                style = Style(color="red")
            elif record.levelno >= logging.WARNING:
                style = Style(color="yellow")

    try:
        self.console.print(renderable, overflow="ignore", crop=False, style=style)
    except Exception:
        self.handleError(record)

def handleError(self, record: logging.LogRecord) -> None:
    """Called when logging is unable to log some output."""

    exc_class, exc = sys.exc_info()[:2]
    # If a broken pipe occurred while calling write() or flush() on the
    # stdout stream in logging's Handler.emit(), then raise our special
    # exception so we can handle it in main() instead of logging the
    # broken pipe error and continuing.
    if (
        exc_class
        and exc
        and self.console.file is sys.stdout
        and _is_broken_pipe_error(exc_class, exc)
    ):
        raise BrokenStdoutLoggingError()

    return super().handleError(record)


