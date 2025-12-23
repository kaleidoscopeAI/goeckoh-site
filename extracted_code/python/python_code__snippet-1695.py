"""A rich renderable that pretty prints an object.

Args:
    _object (Any): An object to pretty print.
    highlighter (HighlighterType, optional): Highlighter object to apply to result, or None for ReprHighlighter. Defaults to None.
    indent_size (int, optional): Number of spaces in indent. Defaults to 4.
    justify (JustifyMethod, optional): Justify method, or None for default. Defaults to None.
    overflow (OverflowMethod, optional): Overflow method, or None for default. Defaults to None.
    no_wrap (Optional[bool], optional): Disable word wrapping. Defaults to False.
    indent_guides (bool, optional): Enable indentation guides. Defaults to False.
    max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
        Defaults to None.
    max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to None.
    max_depth (int, optional): Maximum depth of nested data structures, or None for no maximum. Defaults to None.
    expand_all (bool, optional): Expand all containers. Defaults to False.
    margin (int, optional): Subtrace a margin from width to force containers to expand earlier. Defaults to 0.
    insert_line (bool, optional): Insert a new line if the output has multiple new lines. Defaults to False.
"""

def __init__(
    self,
    _object: Any,
    highlighter: Optional["HighlighterType"] = None,
    *,
    indent_size: int = 4,
    justify: Optional["JustifyMethod"] = None,
    overflow: Optional["OverflowMethod"] = None,
    no_wrap: Optional[bool] = False,
    indent_guides: bool = False,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
    max_depth: Optional[int] = None,
    expand_all: bool = False,
    margin: int = 0,
    insert_line: bool = False,
) -> None:
    self._object = _object
    self.highlighter = highlighter or ReprHighlighter()
    self.indent_size = indent_size
    self.justify: Optional["JustifyMethod"] = justify
    self.overflow: Optional["OverflowMethod"] = overflow
    self.no_wrap = no_wrap
    self.indent_guides = indent_guides
    self.max_length = max_length
    self.max_string = max_string
    self.max_depth = max_depth
    self.expand_all = expand_all
    self.margin = margin
    self.insert_line = insert_line

def __rich_console__(
    self, console: "Console", options: "ConsoleOptions"
) -> "RenderResult":
    pretty_str = pretty_repr(
        self._object,
        max_width=options.max_width - self.margin,
        indent_size=self.indent_size,
        max_length=self.max_length,
        max_string=self.max_string,
        max_depth=self.max_depth,
        expand_all=self.expand_all,
    )
    pretty_text = Text.from_ansi(
        pretty_str,
        justify=self.justify or options.justify,
        overflow=self.overflow or options.overflow,
        no_wrap=pick_bool(self.no_wrap, options.no_wrap),
        style="pretty",
    )
    pretty_text = (
        self.highlighter(pretty_text)
        if pretty_text
        else Text(
            f"{type(self._object)}.__repr__ returned empty string",
            style="dim italic",
        )
    )
    if self.indent_guides and not options.ascii_only:
        pretty_text = pretty_text.with_indent_guides(
            self.indent_size, style="repr.indent"
        )
    if self.insert_line and "\n" in pretty_text:
        yield ""
    yield pretty_text

def __rich_measure__(
    self, console: "Console", options: "ConsoleOptions"
) -> "Measurement":
    pretty_str = pretty_repr(
        self._object,
        max_width=options.max_width,
        indent_size=self.indent_size,
        max_length=self.max_length,
        max_string=self.max_string,
        max_depth=self.max_depth,
        expand_all=self.expand_all,
    )
    text_width = (
        max(cell_len(line) for line in pretty_str.splitlines()) if pretty_str else 0
    )
    return Measurement(text_width, text_width)


