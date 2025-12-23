"""Install automatic pretty printing in the Python REPL.

Args:
    console (Console, optional): Console instance or ``None`` to use global console. Defaults to None.
    overflow (Optional[OverflowMethod], optional): Overflow method. Defaults to "ignore".
    crop (Optional[bool], optional): Enable cropping of long lines. Defaults to False.
    indent_guides (bool, optional): Enable indentation guides. Defaults to False.
    max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
        Defaults to None.
    max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to None.
    max_depth (int, optional): Maximum depth of nested data structures, or None for no maximum. Defaults to None.
    expand_all (bool, optional): Expand all containers. Defaults to False.
    max_frames (int): Maximum number of frames to show in a traceback, 0 for no maximum. Defaults to 100.
"""
from pip._vendor.rich import get_console

console = console or get_console()
assert console is not None

def display_hook(value: Any) -> None:
    """Replacement sys.displayhook which prettifies objects with Rich."""
    if value is not None:
        assert console is not None
        builtins._ = None  # type: ignore[attr-defined]
        console.print(
            value
            if _safe_isinstance(value, RichRenderable)
            else Pretty(
                value,
                overflow=overflow,
                indent_guides=indent_guides,
                max_length=max_length,
                max_string=max_string,
                max_depth=max_depth,
                expand_all=expand_all,
            ),
            crop=crop,
        )
        builtins._ = value  # type: ignore[attr-defined]

if "get_ipython" in globals():
    ip = get_ipython()  # type: ignore[name-defined]
    from IPython.core.formatters import BaseFormatter

    class RichFormatter(BaseFormatter):  # type: ignore[misc]
        pprint: bool = True

        def __call__(self, value: Any) -> Any:
            if self.pprint:
                return _ipy_display_hook(
                    value,
                    console=get_console(),
                    overflow=overflow,
                    indent_guides=indent_guides,
                    max_length=max_length,
                    max_string=max_string,
                    max_depth=max_depth,
                    expand_all=expand_all,
                )
            else:
                return repr(value)

    # replace plain text formatter with rich formatter
    rich_formatter = RichFormatter()
    ip.display_formatter.formatters["text/plain"] = rich_formatter
else:
    sys.displayhook = display_hook


