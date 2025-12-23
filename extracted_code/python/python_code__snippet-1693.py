# needed here to prevent circular import:
from .console import ConsoleRenderable

# always skip rich generated jupyter renderables or None values
if _safe_isinstance(value, JupyterRenderable) or value is None:
    return None

console = console or get_console()

with console.capture() as capture:
    # certain renderables should start on a new line
    if _safe_isinstance(value, ConsoleRenderable):
        console.line()
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
            margin=12,
        ),
        crop=crop,
        new_line_start=True,
        end="",
    )
# strip trailing newline, not usually part of a text repr
# I'm not sure if this should be prevented at a lower level
return capture.get().rstrip("\n")


